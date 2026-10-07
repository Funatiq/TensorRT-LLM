<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Continuous Integration Overview

This page explains how TensorRT‑LLM's CI is organized and how individual tests map to Jenkins stages. Most stages execute integration tests defined in YAML files, while unit tests run as part of a merge‑request pipeline. The sections below describe how to locate a test and trigger the stage that runs it.

## Table of Contents
1. [CI pipelines](#ci-pipelines)
2. [Test definitions](#test-definitions)
3. [Unit tests](#unit-tests)
4. [Jenkins stage names](#jenkins-stage-names)
5. [Finding the stage for a test](#finding-the-stage-for-a-test)
6. [Waiving tests](#waiving-tests)
7. [Multi-GPU Tests](#multi-gpu-tests)
8. [Triggering CI Best Practices](#triggering-ci-best-practices)

## CI pipelines

Pull requests do not start testing by themselves. Developers trigger the CI by commenting `/bot run` (optionally with arguments) on the pull request (see [Pull Request Template](source:.github/pull_request_template.md) for more details). That kicks off the **merge-request pipeline** (defined in `jenkins/L0_MergeRequest.groovy`), which runs unit tests and integration tests whose YAML entries specify `stage: pre_merge`. Once a pull request is merged, a separate **post-merge pipeline** (defined in `jenkins/L0_Test.groovy`) runs every test marked `post_merge` across all supported GPU configurations.

`stage` tags live in the YAML files under `tests/integration/test_lists/test-db/`. Searching those files for `stage: pre_merge` shows exactly which tests the merge-request pipeline covers.

## Test definitions

Integration tests are listed under `tests/integration/test_lists/test-db/`. Most YAML files are named after the GPU or configuration they run on (for example `l0_a100.yml`). Some files, like `l0_sanity_check.yml`, use wildcards and can run on multiple hardware types. Entries contain conditions and a list of tests. Two important terms in each entry are:

- `stage`: either `pre_merge` or `post_merge`.
- `backend`: for example `pytorch`, `cpp`, or `fmha`. Grep the YAML files for `backend:` to see the values currently in use.

Example from `l0_a30.yml`:

```yaml
      terms:
        stage: pre_merge
        backend: pytorch
  tests:
  - unittest/_torch/sampler/test_beam_search.py
```

## Unit tests

Unit tests live under `tests/unittest/` and run during the merge-request pipeline. They are invoked from `jenkins/L0_MergeRequest.groovy` and do not require mapping to specific hardware stages.

### C++ build graph checks

C++ component test executables use explicit libraries and are excluded from the
default build. Select their owner aggregate, such as `common-tests`, or an
individual executable. CTest schedules tests after the selected executables have
been built.

CMake requests its File API codemodel, cache, and toolchain data during every
configuration. `check-build-graph` audits the resulting evaluated graph, including
transitive dependencies and configuration-specific generator expressions. Every
`add_tllm_gtest` executable depends on this check, and `scripts/build_wheel.py`
runs it before building, including configure-only invocations. This gives the
Jenkins wheel-build paths the same checks as local component builds.

```bash
cmake --build cpp/build_RelWithDebInfo --target check-build-graph
cmake --build cpp/build_RelWithDebInfo --target common-tests
ctest --test-dir cpp/build_RelWithDebInfo -L common --output-on-failure
```

The initial policy enforces these boundaries:

- Component tests cannot reach the main facade or Torch operator aggregates unless
  they explicitly request `FULL_STACK`.
- Component libraries cannot reach the facade, language bindings, tests, or
  benchmarks.
- Representative lightweight tests (`stringUtilsTest`, `executorConfigTest`, and
  `contextTransferCoordinatorTest`) cannot reach Torch libraries or unrelated
  kernel/attention aggregates.

Violations print the shortest dependency path and fail the build. Unexplained
kernel source duplication also fails without `--strict`. Cycles, duplicate
compilation outside the kernel tree, new targets, source ownership changes, and
positive dependency/compile growth are advisory during rollout. `--strict` makes
these findings fail too. Advisory findings must be resolved or reviewed as intentional before enabling
strict mode. `--report-only` reports hard
violations without failing and is intended for investigation.

```bash
cmake --build cpp/build_RelWithDebInfo --target report-build-graph
python3 scripts/check_cpp_build_graph.py \
  --build-dir cpp/build_RelWithDebInfo \
  --report /tmp/build-graph.json
```

Reports include repository-owned native targets, compiled source ownership,
compile-unit counts, dependency fan-out, transitive counts, reverse fan-in, and
configuration metadata. The graph represents CMake build dependencies, which
include build-order edges. Interface/imported targets are not separate native
nodes; CMake resolves their contributions into native dependencies and linker
fragments. Imported Torch libraries are checked through the evaluated linker
fragments. CUDA source counts describe translation units, not individual GPU
architecture compiler passes.

Source/build paths are normalized. Profiles include the generator, configuration,
architecture, effective feature options and compiler flags, toolkit/compiler
versions, and compiler launchers. Baselines are stored under
`cpp/cmake/build_graph_baselines/<profile-hash>.json`; mismatched explicit baselines
fail, and uncaptured profiles receive an advisory finding. Windows, release, and
other architecture profiles need their own captures rather than reusing a Linux
SM80 baseline.

After reviewing an intentional graph change, regenerate its baseline:

```bash
python3 scripts/check_cpp_build_graph.py \
  --build-dir cpp/build_RelWithDebInfo --update-baseline
```

Baseline updates do not bypass architectural violations. The policy in
`cpp/cmake/build_graph_policy.json` allows intentional duplicate sources only by
source pattern and an explicit owner set. Do not add exceptions merely to hide
residual-glob duplication. A fresh graph capture has to succeed before the
checker can run; missing replies/manifests fail instead of silently skipping the
audit. CPU regression tests live in
`tests/unittest/scripts/test_check_cpp_build_graph.py`, covered by the existing
`unittest/scripts` entry in `l0_cpu.yml`.

### Kernel source ownership

Ordinary kernel families use explicit `SOURCES` lists in
`cpp/tensorrt_llm/kernels/kernelComponents.cmake` and
`residualKernelComponents.cmake`. `add_tllm_kernel_library` treats
`LINK_LIBRARIES` as implementation dependencies; use `PUBLIC_LINK_LIBRARIES`
when a public header exposes dependency types or requires its include paths.
CUDA/common header requirements remain public. Static implementation dependencies
still propagate for linking through CMake's `LINK_ONLY` semantics.

`kernels_src` is a compatibility archive. It packages objects from Marlin, causal
convolution, group RMS normalization, and the DSV3/Llama4 minimum-latency owners,
then links the former residual families. These sources compile once under their
owners' architecture and compilation settings. The facade continues to retain
component exports through its existing whole-archive assembly.

`kernelSourceOwnership.cmake` inventories source files solely to validate
ownership. Adding an undeclared `.cpp` or `.cu` in the audited kernel scope
triggers reconfiguration and fails before the file can enter an aggregate.
Duplicate declared owners also fail configuration. Add each ordinary source to
its component's explicit list; add architecture settings to that owner.

Specialized backend directories retain their local build definitions, generation,
and conditional source selection. Their explicitly delegated scopes are listed
at the end of the kernel `CMakeLists.txt`; each scope must name an existing build
definition. CUTLASS owns `moe/cutlass` from `cutlass_kernels/CMakeLists.txt`.
Delegation handles inactive/generated backend sources; it does not exempt
compiled sources from the evaluated graph's duplicate checks. Intentional
multi-architecture reuse requires a reviewed source/owner exception in the graph
policy. CPU regressions cover unknown-source reconfiguration, duplicate owners,
backend scope boundaries, and public/private usage propagation.

### Focused Torch operator consumers

`th_common_sequence` (`tllm::torch_sequence`) and `th_common_visual_gen`
(`tllm::torch_visual_gen`), and `th_common_decoding` (`tllm::torch_decoding`)
use direct kernel/runtime dependencies. Link these
object targets with `target_link_libraries` to retain operator registration
objects and propagate final-link requirements. The sequence family forwards
its specialized kernel objects through `INTERFACE_SOURCES`; consuming only
`$<TARGET_OBJECTS:th_common_sequence>` does not propagate those requirements.
The DiT kernel component excludes the separate Ulysses permutation component.

Focused host consumers validate every family schema and CUDA dispatch
registration, and assert that unrelated operators are absent:

```bash
cmake --build cpp/build_RelWithDebInfo --parallel --target \
  torchSequenceRegistrationTest torchVisualGenRegistrationTest
ctest --test-dir cpp/build_RelWithDebInfo -L thop --output-on-failure
```

The decoding consumer also checks its existing Composite dispatch registrations
for host queries and LoRA capability checks. Its dependency closure includes
LoRA/grouped GEMM, speculative decoding, and specialized sampling/MHC/compressor
objects, while excluding attention and MoE backends.

These checks do not execute GPU kernels. They join `runtime-tests` and
`google-tests`. The build graph policy permits these migrated families in
component tests and rejects transitive facade, aggregate, attention, GEMM, MoE,
and transport dependencies. Other Torch families still require `FULL_STACK`
consumers. `th_common` retains its existing packaging and loading contract.

### Measuring C++ component builds

`scripts/benchmark_cpp_components.py` records wall time, Ninja compilation/link
actions, cache statistics, and build metadata. Its default measures an existing
incremental build. `--clean-dependencies` cleans the selected target's dependency
artifacts between uncached, cache-fill, and warm-cache builds, then measures a
steady-state build. This does not clean the entire build directory. Use a build
directory with no other builds running during these measurements.

```bash
python3 scripts/benchmark_cpp_components.py \
  --build-dir cpp/build_RelWithDebInfo \
  --targets stringUtilsTest executorConfigTest contextTransferCoordinatorTest \
  --jobs 16 --clean-dependencies \
  --leaf-source cpp/tensorrt_llm/common/stringUtils.cpp \
  --header cpp/include/tensorrt_llm/common/assert.h \
  --output /tmp/component-builds.json
```

Measurements use an isolated ccache shared across the targets in that invocation;
cache-fill builds may reuse dependencies cached by earlier targets. Uncached
builds disable ccache. Source/header invalidation changes only modification times
and restores them after the build, including failures and handled interruptions.
Reports are saved after each completed phase, with logs beside the report. Use a
fresh output name for each clean measurement series. Steady-state timings include
the always-run graph audit. These are build measurements, not GPU test execution
or model-quality/performance results.

## Jenkins stage names

`jenkins/L0_Test.groovy` maps stage names to these YAML files.  For A100 the mapping includes:

```groovy
    "A100X-PyTorch-1": ["a100x", "l0_a100", 1, 1],
    "A100X-PyTorch-Post-Merge-1": ["a100x", "l0_a100", 1, 1],
```

The array elements are: GPU type, YAML file (without extension), shard index, and total number of shards. Only tests with `stage: post_merge` from that YAML file are selected when a `Post-Merge` stage runs.

Stage names are unbracketed, with `Post-Merge` embedded directly in the name. Copy the name exactly as it appears in the Groovy map. Jenkins `stage_list` matches names without a `*` wildcard exactly, so a stale spelling such as `A100X-Triton-[Post-Merge]-1` silently selects nothing. The mapping script's `--stages` option is stricter: it reports unknown names and near-match suggestions on stderr.

## Finding the stage for a test

1. Locate the test in the appropriate YAML file under `tests/integration/test_lists/test-db/` and note its `stage` and `backend` values.
2. Search `jenkins/L0_Test.groovy` for a stage whose YAML file matches (for example `l0_a100`) and whose name contains `Post-Merge` if the YAML entry uses `stage: post_merge`.
3. The resulting stage name(s) are what you pass to Jenkins via the `stage_list` parameter when triggering a job.

### Using `test_to_stage_mapping.py`

Manually searching YAML and Groovy files can be tedious.  The helper script
`scripts/test_to_stage_mapping.py` automates the lookup:

```bash
python scripts/test_to_stage_mapping.py --tests "unittest/_torch/sampler/test_beam_search.py"
python scripts/test_to_stage_mapping.py --tests test_beam_search
python scripts/test_to_stage_mapping.py --stages A100X-PyTorch-Post-Merge-1
python scripts/test_to_stage_mapping.py --test-list my_tests.txt
python scripts/test_to_stage_mapping.py --test-list my_tests.yml
```

The first two commands print the Jenkins stages that run the specified tests or
patterns. Patterns are matched by substring, so partial test names are
supported out of the box. The third lists every test executed in the given stage. When
providing tests on the command line, quote each test string so the shell does
not interpret the `[` and `]` characters as globs. Alternatively, store the
tests in a newline‑separated text file or a YAML list and supply it with
`--test-list`.


To run the same tests on your pull request, comment:

```bash
/bot run --stage-list "A100X-PyTorch-Post-Merge-1"
```

This executes the same tests that run post-merge for this hardware/backend.


## Waiving tests

Sometimes a test is known to fail due to a bug or unsupported feature. Instead
of removing it from the YAML test lists, add the test name to
`tests/integration/test_lists/waives.txt`. Every CI run passes this file to
pytest via `--waives-file`, so the listed tests are skipped automatically.

Each line contains the fully qualified test name followed by an optional
`SKIP (reason)` marker. A `full:GPU_TYPE/` prefix restricts the waive to a
specific hardware family. Example:

```text
accuracy/test_disaggregated_serving.py::TestDeepSeekV32Exp::test_auto_dtype[False] SKIP (https://nvbugs/6120535)
full:A100/accuracy/test_llm_api_pytorch_multimodal.py::TestExaone4_5_33B::test_auto_dtype[full_budget] SKIP (https://nvbugs/6422318)
```

Changes to `waives.txt` should include a bug link or brief explanation so other
developers understand why the test is disabled.

## Multi-GPU Tests

Running multi-GPU tests (`--add-multi-gpu-test`, `--only-multi-gpu-test`) requires the
`ci: full pre-merge approved` label on the PR. Without this label the
`[Test-x86_64-Multi-GPU] Remote Run` and `[Test-SBSA-Multi-GPU] Remote Run` stages will
fail with an explanatory error.

To obtain the label, ask a member of `NVIDIA/trt-llm-ci-approvers` to apply it. Only
members of that team can add the label; unauthorized additions are automatically removed
by the `Guard Full Pre-Merge Approval Label` GitHub Actions workflow.

Once the label is present, re-trigger CI with the same command you used
originally. For example:

```bash
# Run the normal pipeline plus multi-GPU stages
/bot run --add-multi-gpu-test

# Run only multi-GPU stages
/bot run --only-multi-gpu-test
```

Post-merge pipelines and GitLab MR builds are exempt from this label check.

## Triggering CI Best Practices

### Triggering Post-merge tests

Full `/bot run --post-merge` runs require the `ci: post-merge approved` PR
label because they can consume substantial shared GPU resources. The label is
intended to be applied by an active member of the
`NVIDIA/trt-llm-ci-approvers` GitHub team. A GitHub workflow validates the label
actor and normally removes invalid approvals. This is a best-effort resource
governance guard, not a strict authorization boundary. The label remains in
place when new commits are pushed and can be removed manually when the approval
no longer applies.

When you only need to verify a handful of post-merge tests, specify exactly
which stages to run:

```bash
/bot run --stage-list "stage-A,stage-B"
```

This runs **only** the stages listed. You can also add stages on top of the
default pre-merge set:

```bash
/bot run --extra-stage "stage-A,stage-B"
```

Both options accept stage names and wildcard patterns defined in
`jenkins/L0_Test.groovy`. Explicit Post-merge stage test doesn't need
`ci: post-merge approved` PR label to run. However, the `"*"`,
`"*Post-Merge*"`, and `"*PerfSanity*"` selectors require the same approval
label, including when they appear in a comma-separated list. Equivalent escaped
or repeated-star forms are treated the same. Other stage selectors, including
explicit stage names and other limited wildcard patterns, retain their existing
behavior.

Being selective keeps CI turnaround fast and conserves hardware resources.

### Avoiding unnecessary `--disable-fail-fast` usage

Avoid habitually using `--disable-fail-fast` as it wastes scarce hardware resources. The CI system automatically reuses successful test stages when commits remain unchanged, and subsequent `/bot run` commands only retry failed stages. Overusing `--disable-fail-fast` keeps failed pipelines consuming resources (like DGX-H100s), increasing queue backlogs and reducing team efficiency.
