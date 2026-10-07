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
"""CPU-only regression coverage for evaluated CMake build boundaries."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "check_cpp_build_graph", REPO_ROOT / "scripts/check_cpp_build_graph.py"
)
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def target(
    name: str,
    dependencies: tuple[str, ...] = (),
    sources: tuple[str, ...] = (),
    directory: str = "source/tensorrt_llm/common",
    owned: bool = True,
    torch_link: bool = False,
) -> checker.Target:
    return checker.Target(
        name, "STATIC_LIBRARY", directory, owned, sources, dependencies, torch_link
    )


def graph(*targets: checker.Target, tests: dict[str, str] | None = None) -> checker.BuildGraph:
    return checker.BuildGraph(
        {item.name: item for item in targets},
        {"configuration": "Debug"},
        tests or {},
        Path("/source"),
        Path("/build"),
    )


@pytest.fixture
def policy() -> dict:
    return checker.read_json(checker.DEFAULT_POLICY)


def test_transitive_facade_dependency_through_vendor_is_rejected(policy: dict) -> None:
    build = graph(
        target("tinyTest", ("vendor",)),
        target("vendor", ("tensorrt_llm",), owned=False),
        target("tensorrt_llm"),
        tests={"tinyTest": "component"},
    )
    errors, _ = checker.audit(build, policy)
    assert any("tinyTest -> vendor -> tensorrt_llm" in error for error in errors)


def test_full_stack_opt_in_is_respected(policy: dict) -> None:
    build = graph(
        target("integrationTest", ("tensorrt_llm",)),
        target("tensorrt_llm"),
        tests={"integrationTest": "full_stack"},
    )
    assert checker.audit(build, policy)[0] == []


def test_component_to_binding_edge_is_rejected(policy: dict) -> None:
    build = graph(target("tllm_runtime_buffers", ("th_common",)), target("th_common"))
    assert any(
        "tllm_runtime_buffers -> th_common" in error for error in checker.audit(build, policy)[0]
    )


def test_production_cannot_reach_test_with_arbitrary_name(policy: dict) -> None:
    build = graph(
        target("tllm_common_error", ("helper",)),
        target("helper", directory="source/tests/unit_tests/common"),
    )
    assert any("Production" in error for error in checker.audit(build, policy)[0])


@pytest.mark.parametrize(
    "dependency", ["_context_attention_kernels_80", "kernels_src", "th_common"]
)
def test_lightweight_closure_excludes_kernel_aggregates(policy: dict, dependency: str) -> None:
    build = graph(target("executorConfigTest", (dependency,)), target(dependency))
    assert any("Lightweight" in error for error in checker.audit(build, policy)[0])


def test_lightweight_closure_detects_imported_torch_link(policy: dict) -> None:
    build = graph(
        target("stringUtilsTest", ("helper",)),
        target("helper", torch_link=True),
    )
    assert any("stringUtilsTest -> helper" in error for error in checker.audit(build, policy)[0])


def test_runtime_kernel_edges_are_permitted(policy: dict) -> None:
    build = graph(
        target("tllm_kernel_all_reduce_fusion", ("tllm_runtime_ipc",)),
        target("tllm_runtime_ipc", ("tllm_kernel_custom_all_reduce",)),
        target("tllm_kernel_custom_all_reduce"),
    )
    assert checker.audit(build, policy) == ([], [])


def test_cycles_are_reported_once_and_traversal_terminates() -> None:
    build = graph(target("a", ("b",)), target("b", ("a", "c")), target("c", ("c",)))
    assert checker.find_cycles(build) == [["a", "b"], ["c"]]
    assert build.paths("a")["c"] == ["a", "b", "c"]


def test_shortest_path_is_used() -> None:
    build = graph(
        target("a", ("long", "short")),
        target("long", ("extra",)),
        target("extra", ("destination",)),
        target("short", ("destination",)),
        target("destination"),
    )
    assert build.paths("a")["destination"] == ["a", "short", "destination"]


def test_duplicate_allowlist_is_source_and_owner_specific(policy: dict) -> None:
    source = "source/kernel.cu"
    build = graph(target("a", sources=(source,)), target("b", sources=(source,)))
    assert len(checker.audit(build, policy)[1]) == 1
    policy["duplicate_source_allowlist"] = [{"source": source, "targets": ["a", "b"]}]
    assert checker.audit(build, policy)[1] == []
    build.targets["c"] = target("c", sources=(source,))
    assert len(checker.audit(build, policy)[1]) == 1


def test_baseline_rejects_same_count_dependency_replacement() -> None:
    before = checker.snapshot(graph(target("a", ("b",)), target("b"), target("c")))
    after = checker.snapshot(graph(target("a", ("c",)), target("b"), target("c")))
    assert any(
        "a dependencies: c" in finding for finding in checker.compare_baseline(after, before)
    )


def test_baseline_requires_matching_configuration() -> None:
    before = checker.snapshot(graph(target("a")))
    after = checker.snapshot(graph(target("a")))
    after["configuration"] = {"configuration": "Release"}
    with pytest.raises(ValueError, match="configuration differs"):
        checker.compare_baseline(after, before)


def test_source_ownership_change_is_reported_at_same_size() -> None:
    before = checker.snapshot(graph(target("a", sources=("source/old.cpp",))))
    after = checker.snapshot(graph(target("a", sources=("source/new.cpp",))))
    assert any("source/new.cpp" in finding for finding in checker.compare_baseline(after, before))


def test_decreases_do_not_trigger_growth() -> None:
    before = checker.snapshot(graph(target("a", ("b",), ("source/a.cpp",)), target("b")))
    after = checker.snapshot(graph(target("a")))
    assert checker.compare_baseline(after, before) == []


def write_reply(tmp_path: Path, configurations: tuple[str, ...] = ("Debug",)) -> Path:
    source = tmp_path / "source"
    build = source / "build"
    source.mkdir()
    build.mkdir()
    reply = build / ".cmake/api/v1/reply"
    reply.mkdir(parents=True)
    checker.write_json(
        reply / "model.json",
        {
            "paths": {"source": str(source), "build": str(build)},
            "configurations": [
                {
                    "name": name,
                    "targets": [{"id": "test-id", "name": "tinyTest", "jsonFile": "test.json"}],
                }
                for name in configurations
            ],
        },
    )
    checker.write_json(
        reply / "test.json",
        {
            "name": "tinyTest",
            "type": "EXECUTABLE",
            "paths": {"source": "tests"},
            "sources": [
                {"path": "tests/test.cpp", "compileGroupIndex": 0},
                {"path": "tests/test.h"},
                {"path": str(build / "generated.cpp"), "compileGroupIndex": 0, "isGenerated": True},
            ],
        },
    )
    checker.write_json(reply / "cache.json", {"entries": [{"name": "ENABLE_UCX", "value": "ON"}]})
    checker.write_json(reply / "toolchains.json", {"toolchains": []})
    checker.write_json(
        reply / "index-001.json",
        {
            "objects": [
                {"kind": kind, "jsonFile": file}
                for kind, file in (
                    ("codemodel", "model.json"),
                    ("cache", "cache.json"),
                    ("toolchains", "toolchains.json"),
                )
            ],
            "cmake": {"version": {"string": "3.27.0"}, "generator": {"name": "Ninja"}},
        },
    )
    (build / "tllm_build_graph_tests.tsv").write_text("tinyTest\tcomponent\n")
    (build / "tllm_build_graph_config.tsv").write_text("ENABLE_UCX\t0\n")
    return build


def test_file_api_headers_generated_paths_and_effective_options(tmp_path: Path) -> None:
    build_dir = write_reply(tmp_path)
    build = checker.load_graph(build_dir)
    assert build.targets["tinyTest"].sources == ("build/generated.cpp", "source/tests/test.cpp")
    assert build.targets["tinyTest"].owned
    assert build.configuration["options"]["ENABLE_UCX"] == "0"
    assert build.tests == {"tinyTest": "component"}


def test_multiconfig_requires_selection(tmp_path: Path) -> None:
    build_dir = write_reply(tmp_path, ("Debug", "Release"))
    with pytest.raises(ValueError, match="select --config"):
        checker.load_graph(build_dir)
    assert checker.load_graph(build_dir, "Release").configuration["configuration"] == "Release"


def test_missing_manifest_cannot_silently_bypass_boundary_checks(tmp_path: Path) -> None:
    build_dir = write_reply(tmp_path)
    (build_dir / "tllm_build_graph_tests.tsv").unlink()
    with pytest.raises(ValueError, match="Missing test-boundary manifest"):
        checker.load_graph(build_dir)


def test_vendor_build_directory_is_not_repository_owned(tmp_path: Path) -> None:
    build_dir = write_reply(tmp_path)
    reply = build_dir / ".cmake/api/v1/reply/test.json"
    definition = checker.read_json(reply)
    definition["paths"]["source"] = str(build_dir / "_deps/vendor-src")
    checker.write_json(reply, definition)
    assert not checker.load_graph(build_dir).targets["tinyTest"].owned


@pytest.mark.parametrize(
    "libraries", ['"/opt/torch/lib/libtorch_cpu.so"', "-ltorch", '"C:/Torch/lib/c10.lib"']
)
def test_imported_torch_library_detection(tmp_path: Path, libraries: str) -> None:
    build_dir = write_reply(tmp_path)
    reply = build_dir / ".cmake/api/v1/reply/test.json"
    definition = checker.read_json(reply)
    definition["link"] = {"commandFragments": [{"role": "libraries", "fragment": libraries}]}
    checker.write_json(reply, definition)
    assert checker.load_graph(build_dir).targets["tinyTest"].torch_link


def test_cli_baseline_report_and_strict_rollout(tmp_path: Path) -> None:
    build_dir = write_reply(tmp_path)
    baseline = tmp_path / "baseline.json"
    report = tmp_path / "report.json"
    arguments = [
        "--build-dir",
        str(build_dir),
        "--baseline",
        str(baseline),
        "--report",
        str(report),
    ]
    assert checker.main([*arguments, "--update-baseline"]) == 0
    assert checker.main([*arguments, "--strict"]) == 0
    reply = build_dir / ".cmake/api/v1/reply/test.json"
    definition = checker.read_json(reply)
    definition["sources"].append({"path": "tests/extra.cpp", "compileGroupIndex": 0})
    checker.write_json(reply, definition)
    assert checker.main(arguments) == 0
    assert checker.main([*arguments, "--strict"]) == 1
    assert checker.read_json(report)["warnings"]
    assert checker.main([*arguments, "--strict", "--report-only"]) == 0


def test_cmake_query_and_transitive_guard_end_to_end(tmp_path: Path) -> None:
    if not shutil.which("cmake") or not shutil.which("ninja"):
        pytest.skip("CMake and Ninja are needed for the evaluated-graph regression")
    source = tmp_path / "fixture"
    source.mkdir()
    (source / "file.cpp").write_text("int main() { return 0; }\n")
    (source / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.27)
project(fixture LANGUAGES CXX)
include("{REPO_ROOT}/cpp/cmake/modules/tllm_build_graph.cmake")
set(Python3_EXECUTABLE "{sys.executable}")
add_library(tensorrt_llm STATIC file.cpp)
add_library(tllm_common_error INTERFACE)
target_link_libraries(tllm_common_error INTERFACE $<$<CONFIG:Debug>:tensorrt_llm>)
add_executable(tinyTest file.cpp)
target_link_libraries(tinyTest PRIVATE tllm_common_error)
set_property(GLOBAL PROPERTY TLLM_BUILD_GRAPH_TEST_MANIFEST "tinyTest\\tcomponent\\n")
tllm_add_build_graph_targets()
''')
    build_dir = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Debug",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    build = checker.load_graph(build_dir)
    assert any(
        "tinyTest -> tensorrt_llm" in finding
        for finding in checker.audit(build, checker.read_json(checker.DEFAULT_POLICY))[0]
    )
    assert checker.main(["--build-dir", str(build_dir), "--report-only"]) == 0
    assert checker.main(["--build-dir", str(build_dir)]) == 1
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        checker.audit(checker.load_graph(build_dir), checker.read_json(checker.DEFAULT_POLICY))[0]
        == []
    )


BENCH_SPEC = importlib.util.spec_from_file_location(
    "benchmark_cpp_components", REPO_ROOT / "scripts/benchmark_cpp_components.py"
)
benchmark_tool = importlib.util.module_from_spec(BENCH_SPEC)
BENCH_SPEC.loader.exec_module(benchmark_tool)


def test_benchmark_restores_timestamps_on_build_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir = write_reply(tmp_path)
    source = build_dir.parent / "leaf.cpp"
    source.write_text("int leaf;\n")
    original = source.stat().st_mtime_ns
    output = tmp_path / "measurements.json"

    def fail(command: list[str], env: dict[str, str], log: Path) -> dict:
        assert source.stat().st_mtime_ns != original
        assert env["CCACHE_DIR"].startswith(str(tmp_path))
        return {"seconds": 0.01, "returncode": 1, "log": log.name}

    monkeypatch.setattr(
        benchmark_tool, "run", lambda *args: {"seconds": 0.01, "returncode": 0, "log": "noop"}
    )
    # The first phase is the existing incremental build; fail only when the
    # leaf invalidation is reached.
    initial_run = benchmark_tool.run

    def dispatch(command: list[str], env: dict[str, str], log: Path) -> dict:
        return fail(command, env, log) if "leaf" in log.name else initial_run(command, env, log)

    monkeypatch.setattr(benchmark_tool, "run", dispatch)
    assert benchmark_tool.benchmark(build_dir, ["tinyTest"], output, 1, False, [source], []) == 1
    assert source.stat().st_mtime_ns == original
    assert checker.read_json(output)["results"][-1]["returncode"] == 1


def test_benchmark_restores_timestamps_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir = write_reply(tmp_path)
    source = build_dir.parent / "header.h"
    source.write_text("#pragma once\n")
    original = source.stat().st_mtime_ns

    def timeout(command: list[str], env: dict[str, str], log: Path) -> dict:
        if "header" in log.name:
            raise subprocess.TimeoutExpired(command, 1)
        return {"seconds": 0.01, "returncode": 0, "log": log.name}

    monkeypatch.setattr(benchmark_tool, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        benchmark_tool.benchmark(
            build_dir, ["tinyTest"], tmp_path / "report.json", 1, False, [], [source]
        )
    assert source.stat().st_mtime_ns == original


def test_benchmark_counts_compilation_and_link_actions(tmp_path: Path) -> None:
    result = benchmark_tool.run(
        [
            sys.executable,
            "-c",
            "print('[1/3] Building CXX object file.o\\n[2/3] Linking CXX executable test\\n[3/3] Running audit')",
        ],
        {},
        tmp_path / "actions.log",
    )
    assert result["actions"] == {"compile": 1, "link": 1, "other": 1}


def test_shared_dependency_checkout_is_not_repository_owned(tmp_path: Path) -> None:
    build_dir = write_reply(tmp_path)
    shared = build_dir.parent / "other-build/_deps/vendor-src"
    reply = build_dir / ".cmake/api/v1/reply/test.json"
    definition = checker.read_json(reply)
    definition["paths"]["source"] = str(shared)
    checker.write_json(reply, definition)
    assert not checker.load_graph(build_dir).targets["tinyTest"].owned
    assert (
        checker.normalized_path(shared / "file.cpp", build_dir.parent, build_dir)
        == "dependency/vendor-src/file.cpp"
    )


def test_distinct_external_sources_are_not_conflated(tmp_path: Path) -> None:
    source = tmp_path / "checkout"
    build = tmp_path / "build"
    first = checker.normalized_path(tmp_path / "one/file.cpp", source, build)
    second = checker.normalized_path(tmp_path / "two/file.cpp", source, build)
    assert first != second


def test_component_rename_cannot_bypass_facade_boundary(policy: dict) -> None:
    build = graph(target("renamed_library", ("tensorrt_llm",)), target("tensorrt_llm"))
    assert any(
        "renamed_library -> tensorrt_llm" in error for error in checker.audit(build, policy)[0]
    )


@pytest.mark.parametrize("library", ["/opt/lib/libtensorrt_llm.so", "-lth_common"])
def test_raw_imported_facade_links_cannot_bypass_test_boundary(
    tmp_path: Path, policy: dict, library: str
) -> None:
    build_dir = write_reply(tmp_path)
    reply = build_dir / ".cmake/api/v1/reply/test.json"
    definition = checker.read_json(reply)
    definition["link"] = {"commandFragments": [{"role": "libraries", "fragment": library}]}
    checker.write_json(reply, definition)
    assert any(
        "facade/operator library" in error
        for error in checker.audit(checker.load_graph(build_dir), policy)[0]
    )


def test_transport_library_is_not_mistaken_for_facade(tmp_path: Path, policy: dict) -> None:
    build_dir = write_reply(tmp_path)
    reply = build_dir / ".cmake/api/v1/reply/test.json"
    definition = checker.read_json(reply)
    definition["link"] = {
        "commandFragments": [
            {"role": "libraries", "fragment": "/opt/lib/libtensorrt_llm_ucx_wrapper.so"}
        ]
    }
    checker.write_json(reply, definition)
    assert checker.audit(checker.load_graph(build_dir), policy)[0] == []


@pytest.mark.parametrize(
    "source",
    ["source/tensorrt_llm/kernels/example.cu", "build/tensorrt_llm/kernels/generated/example.cu"],
)
def test_kernel_duplicates_are_errors_without_strict_mode(policy: dict, source: str) -> None:
    build = graph(target("owner", sources=(source,)), target("kernels_src", sources=(source,)))
    errors, warnings = checker.audit(build, policy)
    assert len(errors) == 1
    assert source in errors[0]
    assert not warnings
    policy["duplicate_source_allowlist"] = [{"source": source, "targets": ["owner", "kernels_src"]}]
    assert checker.audit(build, policy) == ([], [])


def configure_kernel_ownership_fixture(tmp_path: Path, extra: str = "") -> Path:
    if not shutil.which("cmake") or not shutil.which("ninja"):
        pytest.skip("CMake and Ninja are required for ownership integration coverage")
    module = REPO_ROOT / "cpp/tensorrt_llm/kernels/kernelSourceOwnership.cmake"
    (tmp_path / "owned.cpp").write_text("int owned() { return 1; }\n")
    (tmp_path / "CMakeLists.txt").write_text(
        f'''cmake_minimum_required(VERSION 3.27)
project(kernel_ownership LANGUAGES CXX)
include("{module.as_posix()}")
add_library(owner STATIC owned.cpp)
{extra}
tllm_check_kernel_source_ownership(TARGETS owner ${{other_targets}}
  ${{delegation}})
'''
    )
    return tmp_path / "build"


def test_new_kernel_file_fails_reconfiguration_before_compilation(tmp_path: Path) -> None:
    build_dir = configure_kernel_ownership_fixture(tmp_path)
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    (tmp_path / "unowned.cpp").write_text("#error This file must not be compiled\n")
    result = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "owner"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode
    assert "Kernel source has no explicit owner: unowned.cpp" in result.stdout + result.stderr
    assert not list(build_dir.rglob("*.o"))


def test_duplicate_kernel_owners_fail_configuration(tmp_path: Path) -> None:
    build_dir = configure_kernel_ownership_fixture(
        tmp_path, "add_library(other STATIC owned.cpp)\nset(other_targets other)"
    )
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode
    assert "Kernel source has multiple owners: owned.cpp: owner, other" in result.stderr


@pytest.mark.parametrize("source", ["backend/optional.cpp", "backend_extra/unowned.cpp"])
def test_backend_delegation_has_directory_boundaries(tmp_path: Path, source: str) -> None:
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend/CMakeLists.txt").write_text("# Optional backend owns its selection.\n")
    path = tmp_path / source
    path.parent.mkdir(exist_ok=True)
    path.write_text("int optional() { return 0; }\n")
    build_dir = configure_kernel_ownership_fixture(
        tmp_path, "set(delegation DELEGATED_DIRECTORIES backend)"
    )
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if source.startswith("backend/"):
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode
        assert f"Kernel source has no explicit owner: {source}" in result.stderr


def test_missing_backend_definition_is_rejected(tmp_path: Path) -> None:
    build_dir = configure_kernel_ownership_fixture(
        tmp_path, "set(delegation DELEGATED_DIRECTORIES missing_backend)"
    )
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode
    assert "Missing delegated kernel build: missing_backend" in result.stderr


def test_kernel_helper_keeps_implementation_usage_private(tmp_path: Path) -> None:
    if not shutil.which("cmake") or not shutil.which("ninja"):
        pytest.skip("CMake and Ninja are required for kernel helper integration coverage")
    module = (REPO_ROOT / "cpp/tensorrt_llm/kernels/kernelComponents.cmake").read_text()
    start = module.index("function(add_tllm_kernel_library target)")
    end = module.index("endfunction()", start) + len("endfunction()")
    (tmp_path / "helper.cmake").write_text(module[start:end] + "\n")
    (tmp_path / "public.h").write_text("int component();\n")
    (tmp_path / "private.h").write_text("int implementation();\n")
    (tmp_path / "implementation.cpp").write_text("int implementation() { return 7; }\n")
    (tmp_path / "component.cpp").write_text(
        '#include "private.h"\n#ifndef PRIVATE_USAGE\n#error Missing private usage\n#endif\n'
        "int component() { return implementation(); }\n"
    )
    (tmp_path / "consumer.cpp").write_text(
        '#include "public.h"\n#ifdef PRIVATE_USAGE\n#error Private usage leaked\n#endif\n'
        "#ifndef PUBLIC_USAGE\n#error Missing public usage\n#endif\n"
        "int main() { return component() == 7 ? 0 : 1; }\n"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.27)
project(kernel_helper LANGUAGES CXX)
function(add_cuda_architectures)
endfunction()
add_library(common INTERFACE)
add_library(tllm::common_cuda ALIAS common)
add_library(public_headers INTERFACE)
target_compile_definitions(public_headers INTERFACE PUBLIC_USAGE)
add_library(implementation STATIC implementation.cpp)
target_compile_definitions(implementation INTERFACE PRIVATE_USAGE)
include(helper.cmake)
add_tllm_kernel_library(tllm_kernel_example SOURCES component.cpp
  LINK_LIBRARIES implementation PUBLIC_LINK_LIBRARIES public_headers)
add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE tllm::kernel_example)
"""
    )
    build_dir = tmp_path / "build"
    subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "consumer"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    executable = "consumer.exe" if sys.platform == "win32" else "consumer"
    subprocess.run([str(build_dir / executable)], check=True, timeout=30)


def test_delegation_names_build_definition_outside_source_directory(tmp_path: Path) -> None:
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend/optional.cpp").write_text("int optional() { return 0; }\n")
    (tmp_path / "backend_build").mkdir()
    (tmp_path / "backend_build/CMakeLists.txt").write_text("# Backend source selection.\n")
    build_dir = configure_kernel_ownership_fixture(
        tmp_path, "set(delegation DELEGATED_DIRECTORIES backend=backend_build)"
    )
    result = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build_dir), "-G", "Ninja"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_generated_kernel_duplicates_outside_kernel_build_directory_are_errors(
    policy: dict,
) -> None:
    source = "external/generated/example.cu"
    build = graph(
        target("kernel_owner", sources=(source,), directory="source/tensorrt_llm/kernels/backend"),
        target("other_owner", sources=(source,)),
    )
    errors, warnings = checker.audit(build, policy)
    assert len(errors) == 1
    assert source in errors[0]
    assert not warnings


@pytest.mark.parametrize("family", ["th_common_sequence", "th_common_visual_gen"])
@pytest.mark.parametrize(
    "dependency",
    [
        "tensorrt_llm",
        "th_common_attention",
        "pg_utils",
        "tllm_kernel_grouped_gemm",
        "tllm_kernel_attention_decode",
        "tllm_runtime_nccl",
        "tllm_kernel_ulysses",
        "tensorrt_llm_ucx_wrapper",
    ],
)
def test_focused_torch_families_reject_broad_transitive_links(
    policy: dict, family: str, dependency: str
) -> None:
    build = graph(
        target(family, ("helper",), directory="source/tensorrt_llm/thop"),
        target("helper", (dependency,), directory="source/tensorrt_llm/thop"),
        target(dependency, directory="source/tensorrt_llm/thop"),
    )
    assert any(
        f"{family} -> helper -> {dependency}" in error for error in checker.audit(build, policy)[0]
    )


@pytest.mark.parametrize(
    ("family", "kernel"),
    [("th_common_sequence", "tllm_kernel_recurrent"), ("th_common_visual_gen", "tllm_kernel_dit")],
)
def test_component_test_can_consume_narrow_torch_family(
    policy: dict, family: str, kernel: str
) -> None:
    build = graph(
        target("focusedTest", (family,), directory="source/tests/unit_tests/thop"),
        target(family, (kernel, "th_utils"), directory="source/tensorrt_llm/thop"),
        target(kernel, directory="source/tensorrt_llm/kernels"),
        target("th_utils", directory="source/tensorrt_llm/thop", torch_link=True),
        tests={"focusedTest": "component"},
    )
    assert checker.audit(build, policy) == ([], [])


@pytest.mark.parametrize("family", ["th_common_new_family", "th_common_attention"])
def test_unmigrated_torch_family_still_requires_full_stack(policy: dict, family: str) -> None:
    build = graph(
        target("focusedTest", (family,), directory="source/tests/unit_tests/thop"),
        target(family, directory="source/tensorrt_llm/thop"),
        tests={"focusedTest": "component"},
    )
    assert any("requires FULL_STACK" in error for error in checker.audit(build, policy)[0])


def test_focused_torch_test_cannot_add_unrelated_kernel_directly(policy: dict) -> None:
    build = graph(
        target(
            "torchSequenceRegistrationTest",
            ("tllm_kernel_grouped_gemm",),
            directory="source/tests/unit_tests/thop",
        ),
        target("tllm_kernel_grouped_gemm", directory="source/tensorrt_llm/kernels"),
        tests={"torchSequenceRegistrationTest": "component"},
    )
    assert any("Focused Torch" in error for error in checker.audit(build, policy)[0])


@pytest.mark.parametrize(
    "dependency",
    [
        "tensorrt_llm",
        "th_common",
        "pg_utils",
        "tllm_kernel_attention_decode",
        "tllm_kernel_moe_prepare",
        "tllm_runtime_distributed",
    ],
)
def test_decoding_torch_family_rejects_unrelated_closures(policy: dict, dependency: str) -> None:
    build = graph(
        target("th_common_decoding", ("helper",), directory="source/tensorrt_llm/thop"),
        target("helper", (dependency,), directory="source/tensorrt_llm/thop"),
        target(dependency, directory="source/tensorrt_llm/thop"),
    )
    assert any("Decoding Torch" in error for error in checker.audit(build, policy)[0])


def test_decoding_consumer_can_use_its_lora_and_sampling_backends(policy: dict) -> None:
    build = graph(
        target(
            "torchDecodingRegistrationTest",
            ("th_common_decoding",),
            directory="source/tests/unit_tests/thop",
        ),
        target(
            "th_common_decoding",
            ("tllm_kernel_grouped_gemm", "fusedSamplingKernels_src"),
            directory="source/tensorrt_llm/thop",
        ),
        target("tllm_kernel_grouped_gemm", directory="source/tensorrt_llm/kernels"),
        target("fusedSamplingKernels_src", directory="source/tensorrt_llm/kernels"),
        tests={"torchDecodingRegistrationTest": "component"},
    )
    assert checker.audit(build, policy) == ([], [])
