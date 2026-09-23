# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end inference and CUDA graph coverage for native V2 KV pool sleep."""

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    ExecutorMemoryType,
    KvCacheConfig,
    SamplingParams,
    SleepConfig,
)

from ..conftest import llm_models_root

MODEL = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
PROMPT = [[1] + [42] * 63 + [43]]


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("mode", ["NONE", "MEMSET", "CPU", "PINNED"])
def test_v2_pool_sleep_inference_and_cuda_graph(mode: str, monkeypatch: pytest.MonkeyPatch):
    # Keep the worker in-process so graph replay can be observed directly.
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    with LLM(
        model=MODEL,
        max_batch_size=1,
        max_num_tokens=128,
        kv_cache_config=KvCacheConfig(
            use_kv_cache_manager_v2=True, enable_block_reuse=True, max_tokens=1024
        ),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1], enable_padding=False),
        sleep_config=SleepConfig(restore_modes={ExecutorMemoryType.KV_CACHE: mode}),
    ) as llm:
        runner = llm._executor.engine.model_engine.cuda_graph_runner
        replay_count = 0
        original_replay = runner.replay

        def counted_replay(*args, **kwargs):
            nonlocal replay_count
            replay_count += 1
            return original_replay(*args, **kwargs)

        monkeypatch.setattr(runner, "replay", counted_replay)
        sampling = SamplingParams(max_tokens=4, end_id=-1, temperature=0)
        first = llm.generate(PROMPT, sampling)[0]
        free_before = torch.cuda.mem_get_info()[0]

        llm.release([ExecutorMemoryType.KV_CACHE])
        free_parked = torch.cuda.mem_get_info()[0]
        assert free_parked - free_before >= 16 * 1024 * 1024

        llm.resume([ExecutorMemoryType.KV_CACHE])
        replay_count = 0
        second = llm.generate(PROMPT, sampling)[0]

        assert second.outputs[0].token_ids == first.outputs[0].token_ids
        if mode in ("NONE", "MEMSET"):
            assert second.cached_tokens == 0
        else:
            assert second.cached_tokens > 0
        assert replay_count > 0
