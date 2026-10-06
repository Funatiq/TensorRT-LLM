#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audit the evaluated C++ build graph without importing TensorRT-LLM.

Hard boundaries and policy-selected duplicate compilation are checked on every
invocation. Other duplicate compilation, cycles, and baseline growth are advisory
unless --strict is requested. Reports describe
CMake build dependencies, including custom/build-order dependencies, rather than
inferring an architectural layer from a source directory.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shlex
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO_ROOT / "cpp/cmake/build_graph_policy.json"
DEFAULT_BASELINES = REPO_ROOT / "cpp/cmake/build_graph_baselines"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Target:
    name: str
    kind: str
    directory: str
    owned: bool
    sources: tuple[str, ...]
    dependencies: tuple[str, ...]
    torch_link: bool = False
    facade_link: bool = False


@dataclass
class BuildGraph:
    targets: dict[str, Target]
    configuration: dict
    tests: dict[str, str]
    source_root: Path
    build_root: Path

    def paths(self, start: str) -> dict[str, list[str]]:
        """Return shortest dependency paths, traversing vendor targets as well."""
        paths = {start: [start]}
        pending = deque([start])
        while pending:
            current = pending.popleft()
            for dependency in self.targets[current].dependencies:
                if dependency not in paths:
                    paths[dependency] = [*paths[current], dependency]
                    pending.append(dependency)
        return paths


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_path(path: Path, source: Path, build: Path) -> str:
    path = path.resolve()
    if "_deps" in path.parts:
        dependency_index = path.parts.index("_deps")
        return "dependency/" + "/".join(path.parts[dependency_index + 1 :])
    thirdparty = source.parent / "3rdparty"
    if path.is_relative_to(thirdparty):
        return "thirdparty/" + path.relative_to(thirdparty).as_posix()
    # Out-of-source build trees can be nested inside the source tree.
    for root, prefix in ((build, "build"), (source, "source")):
        if path.is_relative_to(root):
            return f"{prefix}/{path.relative_to(root).as_posix()}"
    return f"external/{path.as_posix().lstrip('/')}"


def load_graph(build_dir: Path, configuration: str | None = None) -> BuildGraph:
    build_dir = build_dir.resolve()
    reply_dir = build_dir / ".cmake/api/v1/reply"
    indexes = sorted(reply_dir.glob("index-*.json"))
    if not indexes:
        raise ValueError(f"No CMake File API reply in {build_dir}; configure the project first")
    index = read_json(indexes[-1])
    replies = {}
    for reference in index["objects"]:
        if reference["kind"] in ("codemodel", "cache", "toolchains"):
            replies[reference["kind"]] = read_json(reply_dir / reference["jsonFile"])
    if set(replies) != {"codemodel", "cache", "toolchains"}:
        raise ValueError("Configure with CODEMODEL 2, CACHE 2 and TOOLCHAINS 1 queries enabled")

    model = replies["codemodel"]
    configurations = model["configurations"]
    if configuration is None:
        if len(configurations) != 1:
            raise ValueError("Multi-config build: select --config explicitly")
        selected = configurations[0]
    else:
        matches = [item for item in configurations if item["name"] == configuration]
        if not matches:
            raise ValueError(f"Configuration {configuration!r} is not in the CMake reply")
        selected = matches[0]
    source_root = Path(model["paths"]["source"]).resolve()
    build_root = Path(model["paths"]["build"]).resolve()
    names = {item["id"]: item["name"] for item in selected.get("targets", [])}
    targets = {}
    for reference in selected.get("targets", []):
        target = read_json(reply_dir / reference["jsonFile"])
        source_dir = (source_root / target["paths"]["source"]).resolve()
        owned = (
            source_dir.is_relative_to(source_root)
            and not source_dir.is_relative_to(build_root)
            and "_deps" not in source_dir.parts
        )
        sources = []
        for entry in target.get("sources", []):
            if "compileGroupIndex" in entry:
                # Relative source paths are relative to the top-level source,
                # including generated paths that use '..' to reach the build.
                sources.append(
                    normalized_path(source_root / entry["path"], source_root, build_root)
                )
        dependencies = tuple(sorted(names[item["id"]] for item in target.get("dependencies", [])))
        libraries = " ".join(
            item["fragment"]
            for item in target.get("link", {}).get("commandFragments", [])
            if item["role"] == "libraries"
        )
        library_names = [
            fragment.strip("\"'").replace("\\", "/").rsplit("/", 1)[-1]
            for fragment in shlex.split(libraries, posix=False)
        ]
        torch_link = any(
            re.match(r"(?:lib|-l)?(?:torch|c10)(?:[_.-]|$)", name) for name in library_names
        )
        facade_link = any(
            re.match(r"(?:lib|-l)?(?:tensorrt_llm|th_common)(?:\.|$)", name)
            for name in library_names
        )
        targets[target["name"]] = Target(
            target["name"],
            target["type"],
            normalized_path(source_dir, source_root, build_root),
            owned,
            tuple(sorted(sources)),
            dependencies,
            torch_link,
            facade_link,
        )

    cache = {entry["name"]: entry["value"] for entry in replies["cache"]["entries"]}
    options = {
        name: value
        for name, value in sorted(cache.items())
        if name.startswith(("BUILD_", "ENABLE_", "USING_OSS_"))
        and not name.endswith(("_DIR", "_ROOT", "_PATH", "_TARGETS"))
        or name
        in {
            "FAST_BUILD",
            "FAST_MATH",
            "NVTX_DISABLE",
            "NVRTC_DYNAMIC_LINKING",
            "CUBLAS_DYNAMIC_LINKING",
            "CURAND_DYNAMIC_LINKING",
            "COMPRESS_FATBIN",
            "INDEX_RANGE_CHECK",
            "SKIP_SOFTMAX_STAT",
            "USE_CXX11_ABI",
            "CMAKE_CUDA_ARCHITECTURES",
            "CMAKE_SYSTEM_NAME",
            "CMAKE_SYSTEM_PROCESSOR",
            "CMAKE_CXX_COMPILER_LAUNCHER",
            "CMAKE_CUDA_COMPILER_LAUNCHER",
            "CMAKE_CXX_FLAGS",
            "CMAKE_CUDA_FLAGS",
            "CMAKE_C_FLAGS",
            f"CMAKE_CXX_FLAGS_{selected['name'].upper()}",
            f"CMAKE_CUDA_FLAGS_{selected['name'].upper()}",
            f"CMAKE_C_FLAGS_{selected['name'].upper()}",
        }
    }
    effective_manifest = build_dir / "tllm_build_graph_config.tsv"
    if not effective_manifest.is_file():
        raise ValueError("Missing effective configuration manifest; reconfigure the project")
    for line in effective_manifest.read_text(encoding="utf-8").splitlines():
        name, value = line.split("\t", 1)
        if name in ("NIXL_ROOT", "MOONCAKE_ROOT", "INTERNAL_CUTLASS_KERNELS_PATH"):
            options[name] = bool(value and not value.endswith("NOTFOUND"))
        elif name in options or name in (
            "CMAKE_SYSTEM_NAME",
            "CMAKE_SYSTEM_PROCESSOR",
            "CMAKE_CXX_COMPILER_LAUNCHER",
            "CMAKE_CUDA_COMPILER_LAUNCHER",
            "CMAKE_CXX_FLAGS",
            "CMAKE_CUDA_FLAGS",
            "CMAKE_C_FLAGS",
            f"CMAKE_CXX_FLAGS_{selected['name'].upper()}",
            f"CMAKE_CUDA_FLAGS_{selected['name'].upper()}",
            f"CMAKE_C_FLAGS_{selected['name'].upper()}",
            "CUDAToolkit_VERSION",
        ):
            options[name] = value
    options = {
        name: value.replace(str(build_root), "<build>").replace(str(source_root), "<source>")
        if isinstance(value, str)
        else value
        for name, value in options.items()
    }
    toolchains = {
        item["language"]: {
            "id": item.get("compiler", {}).get("id", ""),
            "version": item.get("compiler", {}).get("version", ""),
        }
        for item in replies["toolchains"]["toolchains"]
    }
    profile = {
        "cmake_version": index["cmake"]["version"]["string"],
        "generator": index["cmake"]["generator"]["name"],
        "configuration": selected["name"],
        "options": options,
        "toolchains": toolchains,
    }
    manifest = build_dir / "tllm_build_graph_tests.tsv"
    if not manifest.is_file():
        raise ValueError("Missing test-boundary manifest; reconfigure with build-graph support")
    tests = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        name, mode = line.split("\t")
        if mode not in ("component", "full_stack") or name in tests:
            raise ValueError(f"Invalid test manifest entry: {line}")
        if name not in targets:
            raise ValueError(f"Test manifest target {name} is absent from configuration")
        tests[name] = mode
    return BuildGraph(targets, profile, tests, source_root, build_root)


def matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def find_cycles(graph: BuildGraph) -> list[list[str]]:
    """Find strongly connected components, including paths through vendor targets."""
    counter = 0
    indexes: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    cycles = []

    def visit(name: str) -> None:
        nonlocal counter
        indexes[name] = low[name] = counter
        counter += 1
        stack.append(name)
        active.add(name)
        for dependency in graph.targets[name].dependencies:
            if dependency not in indexes:
                visit(dependency)
                low[name] = min(low[name], low[dependency])
            elif dependency in active:
                low[name] = min(low[name], indexes[dependency])
        if low[name] == indexes[name]:
            members = []
            while True:
                member = stack.pop()
                active.remove(member)
                members.append(member)
                if member == name:
                    break
            if (len(members) > 1 or name in graph.targets[name].dependencies) and any(
                graph.targets[member].owned for member in members
            ):
                cycles.append(sorted(members))

    for name in sorted(graph.targets):
        if name not in indexes:
            visit(name)
    return sorted(cycles)


def audit(graph: BuildGraph, policy: dict) -> tuple[list[str], list[str]]:
    """Separate enforced architectural boundaries from advisory findings."""
    errors = []
    warnings = []
    forbidden_test = policy["component_test_forbidden"]
    for name, mode in sorted(graph.tests.items()):
        if mode == "component":
            for dependency, path in graph.paths(name).items():
                if graph.targets[dependency].facade_link:
                    errors.append(
                        f"Component test links a facade/operator library: {' -> '.join(path)}"
                    )
                if (
                    dependency != name
                    and matches(dependency, forbidden_test)
                    and not matches(dependency, policy.get("component_test_allowed", []))
                ):
                    errors.append(f"Component test requires FULL_STACK: {' -> '.join(path)}")
    for rule in policy["boundaries"]:
        starts = sorted(
            name
            for name, target in graph.targets.items()
            if target.owned
            and (rule.get("include_tests") or name not in graph.tests)
            and (
                matches(name, rule["from"])
                or matches(target.directory, rule.get("from_directories", []))
            )
            and not matches(name, rule.get("exclude_from", []))
            and not matches(target.directory, rule.get("exclude_from_directories", []))
        )
        for name in starts:
            for dependency, path in graph.paths(name).items():
                if rule.get("forbid_facade_libraries") and graph.targets[dependency].facade_link:
                    errors.append(
                        f"{rule['name']}: {' -> '.join(path)} links a facade/operator library"
                    )
                if dependency != name and (
                    matches(dependency, rule["to"])
                    or matches(graph.targets[dependency].directory, rule.get("to_directories", []))
                ):
                    errors.append(f"{rule['name']}: {' -> '.join(path)}")

    for name in policy["lightweight_tests"]:
        if name not in graph.targets:
            continue
        for dependency, path in graph.paths(name).items():
            target = graph.targets[dependency]
            if matches(dependency, policy["lightweight_forbidden"]) or target.torch_link:
                errors.append(f"Lightweight test dependency: {' -> '.join(path)}")

    for cycle in find_cycles(graph):
        warnings.append(f"Dependency cycle: {', '.join(cycle)}")
    for source, owners in source_owners(graph).items():
        if len(owners) > 1 and not any(
            fnmatch.fnmatchcase(source, exception["source"])
            and set(owners) <= set(exception["targets"])
            for exception in policy.get("duplicate_source_allowlist", [])
        ):
            kernel_duplicate = matches(source, policy.get("duplicate_source_errors", [])) or any(
                matches(
                    graph.targets[owner].directory, policy.get("duplicate_owner_directories", [])
                )
                for owner in owners
            )
            findings = errors if kernel_duplicate else warnings
            findings.append(f"Source compiled by multiple targets: {source}: {', '.join(owners)}")
    return sorted(set(errors)), sorted(set(warnings))


def source_owners(graph: BuildGraph) -> dict[str, list[str]]:
    owners: dict[str, set[str]] = defaultdict(set)
    for name, target in graph.targets.items():
        if target.owned:
            for source in target.sources:
                owners[source].add(name)
    return {source: sorted(names) for source, names in sorted(owners.items())}


def snapshot(graph: BuildGraph) -> dict:
    paths = {name: graph.paths(name) for name in graph.targets}
    targets = {}
    for name, target in sorted(graph.targets.items()):
        if not target.owned:
            continue
        dependencies = sorted(
            dependency
            for dependency in paths[name]
            if dependency != name and graph.targets[dependency].owned
        )
        targets[name] = {
            "type": target.kind,
            "directory": target.directory,
            "sources": list(target.sources),
            "compile_units": len(target.sources),
            "dependencies": sorted(
                dependency for dependency in target.dependencies if graph.targets[dependency].owned
            ),
            "transitive_dependency_count": len(dependencies),
            "transitive_dependency_hash": hashlib.sha256(
                "\n".join(dependencies).encode()
            ).hexdigest()[:16],
            "reverse_fan_in": sum(name in path and other != name for other, path in paths.items()),
        }
    return {
        "_copyright": "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "_license": "Apache-2.0",
        "schema_version": SCHEMA_VERSION,
        "configuration": graph.configuration,
        "tests": dict(sorted(graph.tests.items())),
        "targets": targets,
        "source_owners": source_owners(graph),
        "totals": {
            "targets": len(targets),
            "compile_units": sum(target["compile_units"] for target in targets.values()),
            "unique_sources": len(source_owners(graph)),
        },
    }


def compare_baseline(current: dict, baseline: dict) -> list[str]:
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported baseline schema version")
    if baseline["configuration"] != current["configuration"]:
        raise ValueError("Baseline configuration differs; use a baseline for this exact profile")
    warnings = []
    for name, target in current["targets"].items():
        previous = baseline["targets"].get(name)
        if previous is None:
            warnings.append(f"New repository target needs review: {name}")
            continue
        if previous["type"] != target["type"]:
            warnings.append(f"Target type changed: {name}: {previous['type']} -> {target['type']}")
        for metric in ("compile_units", "reverse_fan_in", "transitive_dependency_count"):
            if target[metric] > previous[metric]:
                warnings.append(f"Growth: {name} {metric}: {previous[metric]} -> {target[metric]}")
        if (
            target["transitive_dependency_count"] == previous["transitive_dependency_count"]
            and target["transitive_dependency_hash"] != previous["transitive_dependency_hash"]
        ):
            warnings.append(f"Dependency closure changed at the same size: {name}")
        for metric in ("dependencies",):
            added = sorted(set(target[metric]) - set(previous[metric]))
            if added:
                warnings.append(f"Growth: {name} {metric}: {', '.join(added)}")
        added_sources = sorted(set(target["sources"]) - set(previous["sources"]))
        if added_sources:
            warnings.append(f"Source ownership changed: {name}: {', '.join(added_sources)}")
    return sorted(warnings)


def profile_id(configuration: dict) -> str:
    encoded = json.dumps(configuration, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--config", help="CMake configuration for multi-config generators")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--baseline", type=Path, help="Compare against this exact configuration")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--report", type=Path, help="Write a portable graph and audit report")
    parser.add_argument(
        "--report-only", action="store_true", help="Report hard boundary violations too"
    )
    parser.add_argument("--strict", action="store_true", help="Fail on advisory findings as well")
    args = parser.parse_args(argv)
    try:
        graph = load_graph(args.build_dir, args.config)
        policy = read_json(args.policy)
        current = snapshot(graph)
        errors, warnings = audit(graph, policy)
        identity = profile_id(graph.configuration)
        baseline_path = args.baseline or DEFAULT_BASELINES / f"{identity}.json"
        if args.update_baseline:
            if errors:
                raise ValueError("Cannot capture a baseline with architectural boundary violations")
            write_json(baseline_path, current)
            print(f"Captured graph baseline: {baseline_path}")
        elif baseline_path.is_file():
            warnings.extend(compare_baseline(current, read_json(baseline_path)))
        elif args.baseline:
            raise ValueError(f"Baseline not found: {baseline_path}")
        else:
            warnings.append(
                f"No reviewed baseline for profile {identity}; growth checks are advisory"
            )
        if args.report:
            write_json(
                args.report,
                {**current, "profile": identity, "errors": errors, "warnings": warnings},
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Build graph audit failed: {error}", file=sys.stderr)
        return 2
    for finding in errors:
        print(f"ERROR: {finding}")
    for finding in warnings:
        print(f"ADVISORY: {finding}")
    totals = current["totals"]
    print(
        f"Build graph {identity}: {totals['targets']} repository targets, "
        f"{totals['compile_units']} compile units, "
        f"{len(errors)} boundary violations, {len(warnings)} advisory findings"
    )
    return int(not args.report_only and (bool(errors) or (args.strict and bool(warnings))))


if __name__ == "__main__":
    sys.exit(main())
