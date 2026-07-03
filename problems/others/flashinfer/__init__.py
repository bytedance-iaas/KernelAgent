# Copyright (c) Meta Platforms, Inc. and affiliates.
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

"""KernelAgent × FlashInfer-Bench integration.

This package bridges KernelAgent's optimization pipeline with the
flashinfer-bench benchmark/submission format used by the MLSys 2026
FlashInfer contest.

Typical usage:
    from flashinfer.adapter import FlashInferProblemAdapter
    from flashinfer.solution_builder import SolutionBuilder
    from flashinfer.definitions import DEFINITIONS, list_definitions

    spec = DEFINITIONS["moe_fp8_block_scale_e256_h7168_i2048_topk8"]
    adapter = FlashInferProblemAdapter(spec)
    adapter.write_problem_dir(Path("/tmp/my_problem"))
"""

from flashinfer.adapter import FlashInferProblemAdapter
from flashinfer.solution_builder import SolutionBuilder
from flashinfer.definitions import DEFINITIONS, list_definitions

__all__ = [
    "FlashInferProblemAdapter",
    "SolutionBuilder",
    "DEFINITIONS",
    "list_definitions",
]
