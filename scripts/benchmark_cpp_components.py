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
"""Measure selected Ninja component builds with an isolated compiler cache.

The default only measures existing incremental builds. --clean-dependencies
explicitly enables target/dependency cleaning and cold/warm cache measurements;
it never cleans the complete build tree. Source/header invalidation uses mtime
changes and restores their timestamps in a finally block.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

from check_cpp_build_graph import load_graph, normalized_path, profile_id, write_json


def run(command: list[str], env: dict[str, str], log: Path) -> dict:
    """Time one operation, retaining diagnostics without flooding the report."""
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=1800
        )
    actions = {"compile": 0, "link": 0, "other": 0}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.match(r"^\[\d+/\d+\]", line):
            action = (
                "compile"
                if "Building " in line and " object " in line
                else ("link" if "Linking " in line else "other")
            )
            actions[action] += 1
    return {
        "actions": actions,
        "seconds": round(time.perf_counter() - start, 3),
        "returncode": result.returncode,
        "log": log.name,
    }


def benchmark(
    build_dir: Path,
    targets: list[str],
    output: Path,
    jobs: int,
    clean_dependencies: bool,
    leaf_sources: list[Path],
    headers: list[Path],
    configuration: str | None = None,
) -> int:
    graph = load_graph(build_dir, configuration)
    if graph.configuration["generator"] != "Ninja":
        raise ValueError("Component cleaning measurements currently require the Ninja generator")
    for target in targets:
        if target not in graph.targets:
            raise ValueError(f"Unknown configured target: {target}")
    touches = [("leaf", path.resolve()) for path in leaf_sources]
    touches.extend(("header", path.resolve()) for path in headers)
    for _, path in touches:
        if not path.is_file() or not path.is_relative_to(graph.source_root):
            raise ValueError(f"Invalid source/header invalidation path: {path}")
    if clean_dependencies and not shutil.which("ccache"):
        raise ValueError("Cold/warm cache measurements require ccache")
    if clean_dependencies:
        for language in ("CXX", "CUDA"):
            # Validate actual cache entries rather than silently measuring an
            # uncached compiler while labeling the result a warm cache build.
            cache_text = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
            if not any(
                line.startswith(f"CMAKE_{language}_COMPILER_LAUNCHER:") and "=ccache" in line
                for line in cache_text.splitlines()
            ):
                raise ValueError(f"Configure CMAKE_{language}_COMPILER_LAUNCHER=ccache first")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = output.parent / f"{output.stem}-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = log_dir / "ccache"
    if clean_dependencies and cache_dir.exists():
        raise ValueError(f"Use a fresh output name; benchmark cache already exists: {cache_dir}")
    env = os.environ.copy()
    env["CCACHE_DIR"] = str(cache_dir.resolve())
    env.pop("CCACHE_DISABLE", None)
    env["CCACHE_REMOTE_STORAGE"] = ""
    for name in (
        "CCACHE_RECACHE",
        "CCACHE_READONLY",
        "CCACHE_READONLY_DIRECT",
        "CCACHE_REMOTE_ONLY",
    ):
        env.pop(name, None)
    cache_config = log_dir / "ccache.conf"
    cache_config.write_text("max_size = 2G\n", encoding="utf-8")
    env["CCACHE_CONFIGPATH"] = str(cache_config.resolve())
    results = []
    report = {
        "configuration": graph.configuration,
        "profile": profile_id(graph.configuration),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
        },
        "jobs": jobs,
        "cache": "isolated, initially empty" if clean_dependencies else "existing build artifacts",
        "results": results,
    }
    failed = False
    for target in targets:
        phases = (
            ["uncached", "cache-fill", "warm-cache", "no-op"]
            if clean_dependencies
            else ["incremental"]
        )
        changes = [(phase, None) for phase in phases]
        changes.extend((kind, path) for kind, path in touches)
        for index, (phase, path) in enumerate(changes):
            if phase in ("uncached", "cache-fill", "warm-cache"):
                subprocess.run(
                    ["ninja", "-C", str(build_dir), "-t", "clean", target],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    check=True,
                    timeout=60,
                )
            saved = path.stat() if path else None
            if path:
                os.utime(path, ns=(saved.st_atime_ns, time.time_ns()))
            command = [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                graph.configuration["configuration"],
                "--target",
                target,
                "--parallel",
                str(jobs),
            ]
            phase_env = {**env, "CCACHE_DISABLE": "1"} if phase == "uncached" else env
            try:
                timing = run(command, phase_env, log_dir / f"{target}-{index}-{phase}.log")
            finally:
                if path:
                    os.utime(path, ns=(saved.st_atime_ns, saved.st_mtime_ns))
            measurement = {"target": target, "phase": phase, **timing}
            if path:
                measurement["invalidated"] = normalized_path(
                    path, graph.source_root, graph.build_root
                )
            if clean_dependencies:
                stats = subprocess.run(
                    ["ccache", "--print-stats"],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                measurement["cache_stats"] = {
                    name: int(value)
                    for name, value in (line.split() for line in stats.stdout.splitlines())
                }
            results.append(measurement)
            write_json(output, report)
            print(
                f"{target} {phase}: {timing['seconds']}s (exit {timing['returncode']})", flush=True
            )
            if timing["returncode"]:
                failed = True
                break
        if failed:
            break
    return int(failed)


def interrupted(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--clean-dependencies", action="store_true")
    parser.add_argument("--leaf-source", type=Path, action="append", default=[])
    parser.add_argument("--header", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        return benchmark(
            args.build_dir.resolve(),
            args.targets,
            args.output,
            args.jobs,
            args.clean_dependencies,
            args.leaf_source,
            args.header,
            args.config,
        )
    except KeyboardInterrupt:
        print("Component benchmark interrupted; source timestamps restored", file=sys.stderr)
        return 130
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"Component benchmark failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
