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

"""Optimization Manager for parallel kernel optimization.

This module provides the OptimizationManager class that orchestrates
parallel kernel optimization using pluggable search strategies:
- beam_search: Maintain top-N kernels, explore M bottlenecks each
- greedy: Simple single-best optimization

Example:
    >>> manager = OptimizationManager(
    ...     strategy="beam_search",
    ...     num_workers=4,
    ...     strategy_config={"num_top_kernels": 2, "num_bottlenecks": 2},
    ... )
    >>> result = manager.run_optimization(
    ...     initial_kernel=kernel_code,
    ...     problem_file=Path("problem.py"),
    ...     test_code=test_file.read_text(),
    ...     max_rounds=20,
    ... )
"""

import difflib
import logging
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Any

from triton_kernel_agent.opt_worker_component.searching.history.json_db import (
    JSONProgramDatabase,
)
from triton_kernel_agent.opt_worker_component.searching.history.models import (
    ProgramEntry,
    ProgramMetrics,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.strategy import (
    SearchStrategy,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.beam_search import (
    BeamSearchStrategy,
)
from triton_kernel_agent.opt_worker_component.searching.strategy.greedy import (
    GreedyStrategy,
)
from utils.config_injectable import config_injectable

# Manager-level component keys resolved by the registry
_MANAGER_LEVEL_KEYS = {"verifier", "benchmarker", "worker_runner"}


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    """Print a labeled metrics block to stdout."""
    print(f"\n{label}")
    print("-" * 60)
    for key, value in metrics.items():
        print(f"  {key:<32} {value}")


def _kernel_diff_lines(old_code: str, new_code: str, max_lines: int = 20) -> list[str]:
    """Return a compact list of changed lines between two kernel codes.

    Only +/- lines are returned (no @@ hunks, no context lines).
    Blank-only changes are suppressed.  At most max_lines entries.
    """
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
    result = []
    for line in diff:
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith(("+", "-")) and line[1:].strip():
            result.append(line.rstrip())
            if len(result) >= max_lines:
                result.append(f"  ... (diff truncated at {max_lines} lines)")
                break
    return result


# NCU metrics to surface in the round results table, in display order.
# Each entry: (label_suffix, ncu_key, format_fn)
_NCU_DISPLAY: list[tuple[str, str, Any]] = [
    ("DRAM throughput",  "dram__throughput.avg.pct_of_peak_sustained_elapsed",              lambda v: f"{v:.1f}%"),
    ("DRAM BW",          "dram__bytes.sum.per_second",                                      lambda v: f"{v/1e9:.1f} GB/s"),
    ("Warp active",      "sm__warps_active.avg.pct_of_peak_sustained_active",               lambda v: f"{v:.1f}%"),
    ("Grid X",           "launch__grid_dim_x",                                              lambda v: f"{int(v)}"),
    ("Block X",          "launch__block_dim_x",                                             lambda v: f"{int(v)}"),
    ("Blocks/SM",        "launch__blocks_per_multiprocessor",                               lambda v: f"{v:.2f}"),
    ("L1 hit rate",      "l1tex__t_sector_hit_rate.pct",                                    lambda v: f"{v:.1f}%"),
    ("L2 hit rate",      "lts__t_sector_hit_rate.pct",                                      lambda v: f"{v:.1f}%"),
    ("Mem coalescing",   "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",   lambda v: f"{v:.1f}%"),
    ("Long SB stalls",   "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",    lambda v: f"{v:.1f}%"),
]


def _add_ncu_submetrics(metrics: dict[str, Any], prefix: str, ncu: dict[str, Any]) -> None:
    """Append NCU profiler sub-rows into a print_metrics dict under the given prefix."""
    for label, key, fmt in _NCU_DISPLAY:
        val = ncu.get(key)
        if val is not None:
            try:
                metrics[f"{prefix} {label}"] = fmt(float(val))
            except (TypeError, ValueError):
                pass


@config_injectable
class OptimizationManager:
    """Manages parallel kernel optimization with pluggable strategies.

    Supports:
    - beam_search: Current default (top-N kernels × M bottlenecks)
    - greedy: Simple single-best optimization

    Platform-specific behaviour (verification, benchmarking, worker
    orchestration) is delegated to injectable components that implement
    :class:`KernelVerifier`, :class:`KernelBenchmarker`, and
    :class:`WorkerRunner`.  When these are not supplied the default
    NVIDIA / CUDA implementations are used.
    """

    def __init__(
        self,
        strategy: str = "beam_search",
        num_workers: int = 4,
        max_rounds: int = 10,
        log_dir: Path | str | None = None,
        database_path: Path | str | None = None,
        strategy_config: dict[str, Any] | None = None,
        openai_model: str = "claude-opus-4.5",
        high_reasoning_effort: bool = True,
        bottleneck_override: str | None = None,
        platform: dict[str, str] | str | None = None,
        kernel_language: str = "triton",
        **worker_kwargs: Any,
    ):
        """Initialize the optimization manager.

        Args:
            strategy: Search strategy name ("beam_search" or "greedy")
            num_workers: Number of parallel workers
            max_rounds: Maximum optimization rounds
            log_dir: Directory for logs and artifacts
            database_path: Path for program database JSON file
            strategy_config: Strategy-specific configuration
            openai_model: Model name for LLM optimization
            high_reasoning_effort: Whether to use high reasoning effort
            bottleneck_override: Pre-computed bottleneck category to skip LLM analysis
            platform: Platform component config.  Can be:
                - ``None`` — use ``"nvidia"`` for all components (default)
                - a string like ``"nvidia"`` — shorthand for all components
                - a dict like ``{"verifier": "nvidia", ...}`` — per-component
            kernel_language: Kernel DSL ("triton" or "cutedsl"). Controls
                benchmarking backend, LLM guidelines, and config extraction.
            **worker_kwargs: Additional kwargs passed to OptimizationWorker
        """
        self.max_rounds = max_rounds
        self.log_dir = (
            Path(log_dir) if log_dir else Path(tempfile.mkdtemp(prefix="opt_"))
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.openai_model = openai_model
        self.high_reasoning_effort = high_reasoning_effort
        self.bottleneck_override = bottleneck_override
        self.kernel_language = kernel_language
        # Ensure kernel_language flows to workers via worker_kwargs
        worker_kwargs.setdefault("kernel_language", kernel_language)
        self.worker_kwargs = worker_kwargs

        # Store template overrides (also stays in worker_kwargs for forwarding)
        self.templates_config = worker_kwargs.get("templates")

        # Setup logging
        self.logger = self._setup_logging()

        # Initialize database
        db_path = (
            Path(database_path)
            if database_path
            else self.log_dir / "program_database.json"
        )
        self.database = JSONProgramDatabase(db_path)

        # Initialize strategy
        self.strategy = self._create_strategy(
            strategy, strategy_config or {}, num_workers
        )

        # Validate worker count
        if num_workers != self.strategy.num_workers_needed:
            raise ValueError(
                f"Strategy '{strategy}' requires {self.strategy.num_workers_needed} "
                f"workers, got {num_workers}. Adjust num_workers or strategy_config."
            )

        self.num_workers = num_workers
        self.benchmark_lock = mp.Lock()
        # Semaphore to serialize NCU profiling - NCU requires exclusive GPU access
        # and has high memory overhead, so only one worker should profile at a time
        self.profiling_semaphore = mp.Semaphore(1)

        # Shared history across beam search iterations
        self.shared_history: list[
            dict
        ] = []  # List of serialized OptimizationAttempt dicts
        self.shared_reflexions: list[dict] = []  # List of serialized Reflexion dicts
        self.history_size: int = 10  # Max history entries to pass to workers

        # ── Platform components (resolved from registry) ─────────
        self._resolve_platform(platform)

        self.logger.info(
            f"OptimizationManager initialized: strategy={strategy}, workers={num_workers}"
        )

    # ------------------------------------------------------------------
    # Platform resolution
    # ------------------------------------------------------------------

    def _resolve_platform(self, platform: dict[str, str] | str | None) -> None:
        """Resolve platform components from the :mod:`platform.registry`.

        Manager-level components (``verifier``, ``benchmarker``,
        ``worker_runner``) are instantiated and stored on *self*.
        Worker-level component names are forwarded to worker processes
        via ``self.worker_kwargs["platform_config"]`` so each worker
        can resolve its own instances from the registry.
        """
        from triton_kernel_agent.platform.registry import registry

        # Expand shorthand → full per-component dict
        if platform is None or isinstance(platform, str):
            impl = platform or "nvidia"
            config = {k: impl for k in registry.list_components()}
        else:
            config = dict(platform)

        # Split manager vs worker keys
        mgr_config = {k: v for k, v in config.items() if k in _MANAGER_LEVEL_KEYS}
        worker_config = {
            k: v for k, v in config.items() if k not in _MANAGER_LEVEL_KEYS
        }

        # Resolve manager-level components (shared kwargs bag is
        # filtered per-factory by the registry)
        components = registry.create_from_config(
            mgr_config,
            log_dir=self.log_dir,
            logger=self.logger,
            benchmark_lock=self.benchmark_lock,
            profiling_semaphore=self.profiling_semaphore,
            openai_model=self.openai_model,
            high_reasoning_effort=self.high_reasoning_effort,
            bottleneck_override=self.bottleneck_override,
            worker_kwargs=self.worker_kwargs,
            kernel_language=self.kernel_language,
        )
        self.verifier = components["verifier"]
        self.benchmarker = components["benchmarker"]
        self.worker_runner = components["worker_runner"]

        # Propagate worker-level config (string names) to worker
        # processes — each worker resolves its own instances via the
        # registry so there are no pickling issues.
        if worker_config:
            self.worker_kwargs["platform_config"] = worker_config

    # ------------------------------------------------------------------
    # Logging / strategy helpers (unchanged)
    # ------------------------------------------------------------------

    def _setup_logging(self) -> logging.Logger:
        """Setup manager logging."""
        logger = logging.getLogger("OptimizationManager")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler(self.log_dir / "manager.log")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)

            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(console)

        return logger

    def _create_strategy(
        self, name: str, config: dict[str, Any], num_workers: int
    ) -> SearchStrategy:
        """Create the search strategy.

        Args:
            name: Strategy name
            config: Strategy-specific configuration
            num_workers: Number of workers

        Returns:
            Configured SearchStrategy instance

        Raises:
            ValueError: If strategy name is unknown
        """
        if name == "beam_search":
            return BeamSearchStrategy(
                num_top_kernels=config.get("num_top_kernels", 2),
                num_bottlenecks=config.get("num_bottlenecks", 2),
                database=self.database,
                logger=self.logger,
            )
        elif name == "greedy":
            return GreedyStrategy(
                database=self.database,
                max_no_improvement=config.get("max_no_improvement", 5),
                logger=self.logger,
            )
        else:
            raise ValueError(f"Unknown strategy: {name}. Use 'beam_search' or 'greedy'")

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def run_optimization(
        self,
        initial_kernel: str,
        problem_file: Path | str,
        test_code: str | list[str],
        max_rounds: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run optimization with the configured strategy.

        Args:
            initial_kernel: Starting kernel code
            problem_file: Path to problem.py defining Model and get_inputs()
            test_code: Test code for correctness verification. Can be a single
                string or a list.
            max_rounds: Override max_rounds (optional)
            **kwargs: Additional kwargs (reserved for future use)

        Returns:
            Dict with:
                - success: bool
                - kernel_code: str | None
                - best_time_ms: float
                - total_rounds: int
                - top_kernels: list[dict]
        """
        max_rounds = max_rounds or self.max_rounds
        problem_file = Path(problem_file)

        # Normalize test_code to list
        if isinstance(test_code, str):
            test_code = [test_code]

        self.logger.info("=" * 80)
        self.logger.info("STARTING OPTIMIZATION")
        self.logger.info("=" * 80)

        # Initialize strategy with starting kernel
        initial_entry = ProgramEntry(
            program_id="initial",
            kernel_code=initial_kernel,
            metrics=ProgramMetrics(time_ms=float("inf")),
            problem_id=str(problem_file),
        )
        self.strategy.initialize(initial_entry)

        # Verify initial kernel correctness before investing in benchmarks/optimization
        if not self._verify_initial_kernel(initial_kernel, problem_file, test_code):
            return {
                "success": False,
                "kernel_code": None,
                "best_time_ms": float("inf"),
                "total_rounds": 0,
                "top_kernels": [],
                "error": "Initial kernel failed correctness verification",
            }

        # Benchmark PyTorch baseline once (before spawning workers)
        pytorch_baseline = self._benchmark_pytorch_baseline(problem_file)

        # Benchmark torch.compile baseline
        pytorch_compile_time = self._benchmark_pytorch_compile(problem_file)

        # Benchmark the initial kernel
        initial_kernel_time = self._benchmark_initial_kernel(
            initial_kernel, problem_file
        )

        def _fmt(t: float) -> str:
            return f"{t:.4f} ms" if t != float("inf") else "N/A"

        def _speedup(ref: float, t: float) -> str:
            if ref == float("inf") or t == float("inf") or t == 0:
                return "N/A"
            return f"{ref / t:.2f}x"

        baselines: dict[str, Any] = {"PyTorch eager": _fmt(pytorch_baseline)}
        if pytorch_compile_time != float("inf"):
            baselines["torch.compile"] = (
                f"{_fmt(pytorch_compile_time)}  ({_speedup(pytorch_baseline, pytorch_compile_time)} vs eager)"
            )
        baselines["Initial kernel"] = (
            f"{_fmt(initial_kernel_time)}  ({_speedup(pytorch_baseline, initial_kernel_time)} vs eager)"
        )
        print_metrics("Baselines", baselines)

        # Round loop
        baseline_ncu_printed = False
        round_num = 0
        for round_num in range(1, max_rounds + 1):
            self.logger.info("")
            self.logger.info(f"{'=' * 20} ROUND {round_num}/{max_rounds} {'=' * 20}")

            # 1. Get candidates from strategy
            candidates = self.strategy.select_candidates(round_num)
            if not candidates:
                self.logger.warning("No candidates to explore, terminating")
                break

            # 2. Spawn workers
            results = self._run_workers(
                candidates,
                round_num,
                problem_file,
                test_code,
                pytorch_baseline,
            )

            # 3. Update strategy with results
            self.strategy.update_with_results(results, round_num)

            # Print per-round results table
            successful = [r for r in results if r.get("success")]
            failed = [r for r in results if not r.get("success")]
            round_metrics: dict[str, Any] = {}
            for r in sorted(successful, key=lambda r: r.get("time_ms", float("inf"))):
                wid = r.get("worker_id", "?")
                t = r.get("time_ms", float("inf"))
                round_metrics[f"Worker {wid}"] = (
                    f"{_fmt(t)}  (vs PyTorch: {_speedup(pytorch_baseline, t)}"
                    f",  vs initial: {_speedup(initial_kernel_time, t)})"
                )
                attempt = r.get("attempt") or {}
                sol = attempt.get("combined_sol_pct", 0.0)
                comp = attempt.get("compute_sol_pct", 0.0)
                mem = attempt.get("memory_sol_pct", 0.0)
                if sol or comp or mem:
                    round_metrics[f"  W{wid} SOL"] = (
                        f"{sol:.1f}% combined  (compute: {comp:.1f}%, memory: {mem:.1f}%)"
                    )
                ncu = r.get("ncu_flat") or {}
                if ncu:
                    _add_ncu_submetrics(round_metrics, f"  W{wid}", ncu)
            for r in failed:
                wid = r.get("worker_id", "?")
                attempt = r.get("attempt") or {}
                if r.get("error"):
                    reason = str(r["error"])[:120]
                elif r.get("early_stop_reason"):
                    reason = r["early_stop_reason"][:120]
                elif attempt and not attempt.get("passed_verification", True):
                    err = attempt.get("error_message", "").strip()
                    reason = f"verification failed: {err[:100]}" if err else "generated kernel failed correctness check"
                elif r.get("time_ms") == float("inf"):
                    reason = "benchmark failed (time=inf)"
                else:
                    reason = "failed (no result)"
                round_metrics[f"Worker {wid} (failed)"] = reason
            print_metrics(f"Round {round_num}/{max_rounds} Results", round_metrics)

            # Print baseline NCU metrics once (from the first worker that has them)
            if not baseline_ncu_printed:
                for r in results:
                    pytorch_ncu = r.get("pytorch_ncu_flat") or {}
                    initial_ncu = r.get("baseline_ncu_flat") or {}
                    if pytorch_ncu or initial_ncu:
                        baseline_ncu: dict[str, Any] = {}
                        if pytorch_ncu:
                            _add_ncu_submetrics(baseline_ncu, "PyTorch eager", pytorch_ncu)
                        if initial_ncu:
                            _add_ncu_submetrics(baseline_ncu, "Initial kernel", initial_ncu)
                        if baseline_ncu:
                            print_metrics("Baseline Profiling", baseline_ncu)
                            baseline_ncu_printed = True
                        break

            # Print per-worker code diffs (parent → what the LLM actually generated)
            for r in sorted(results, key=lambda r: r.get("worker_id", 0)):
                wid = r.get("worker_id", "?")
                old_code = r.get("parent_kernel_code", "")
                new_code = r.get("generated_kernel_code") or r.get("kernel_code", "")
                if not old_code or not new_code or old_code == new_code:
                    continue
                diff_lines = _kernel_diff_lines(old_code, new_code)
                if diff_lines:
                    status = "" if r.get("success") else " (failed)"
                    print(f"\n  Worker {wid}{status} code changes:")
                    for line in diff_lines:
                        print(f"    {line}")

            # Log per-round winner summary (for log file)
            if successful:
                best = min(successful, key=lambda r: r.get("time_ms", float("inf")))
                self.logger.info(
                    f"Round {round_num} best: worker {best['worker_id']} at {best['time_ms']:.4f} ms"
                )
            else:
                self.logger.info(f"Round {round_num}: no successful workers")

            # 4. Check termination
            if self.strategy.should_terminate(round_num, max_rounds):
                self.logger.info("Strategy signaled termination")
                break

        # Return best result
        best = self.strategy.get_best_program()

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("OPTIMIZATION COMPLETE")
        self.logger.info("=" * 80)

        if best:
            self.logger.info(f"Best time: {best.metrics.time_ms:.4f}ms")
            if initial_kernel_time != float("inf") and best.metrics.time_ms > 0:
                speedup = initial_kernel_time / best.metrics.time_ms
                self.logger.info(f"Speedup vs initial kernel: {speedup:.2f}x")
            if pytorch_baseline != float("inf") and best.metrics.time_ms > 0:
                speedup_pt = pytorch_baseline / best.metrics.time_ms
                self.logger.info(f"Speedup vs PyTorch eager: {speedup_pt:.2f}x")

        return {
            "success": best is not None and best.metrics.time_ms != float("inf"),
            "kernel_code": best.kernel_code if best else None,
            "best_time_ms": best.metrics.time_ms if best else float("inf"),
            "total_rounds": round_num,
            "pytorch_baseline_ms": pytorch_baseline,
            "pytorch_compile_ms": pytorch_compile_time,
            "initial_kernel_time_ms": initial_kernel_time,
            "top_kernels": [
                {
                    "kernel_code": p.kernel_code,
                    "time_ms": p.metrics.time_ms,
                    "generation": p.generation,
                    "program_id": p.program_id,
                }
                for p in self.database.get_top_k(5)
            ],
        }

    # ------------------------------------------------------------------
    # Thin delegates to platform components
    # ------------------------------------------------------------------

    def _benchmark_pytorch_baseline(self, problem_file: Path) -> float:
        """Benchmark the eager reference implementation."""
        return self.benchmarker.benchmark_reference(problem_file)

    def _verify_initial_kernel(
        self,
        initial_kernel: str,
        problem_file: Path,
        test_code: list[str],
    ) -> bool:
        """Verify the initial kernel passes correctness before optimization."""
        return self.verifier.verify(initial_kernel, problem_file, test_code)

    def _benchmark_initial_kernel(
        self, initial_kernel: str, problem_file: Path
    ) -> float:
        """Benchmark the initial kernel before optimization begins."""
        return self.benchmarker.benchmark_kernel(initial_kernel, problem_file)

    def _benchmark_pytorch_compile(self, problem_file: Path) -> float:
        """Benchmark the compiler-optimized reference."""
        return self.benchmarker.benchmark_reference_compiled(problem_file)

    def _run_workers(
        self,
        candidates: list[dict[str, Any]],
        round_num: int,
        problem_file: Path,
        test_code: list[str],
        pytorch_baseline: float,
    ) -> list[dict[str, Any]]:
        """Spawn workers for each candidate and collect results."""
        results = self.worker_runner.run_workers(
            candidates=candidates,
            round_num=round_num,
            problem_file=problem_file,
            test_code=test_code,
            pytorch_baseline=pytorch_baseline,
            shared_history=(
                self.shared_history[-self.history_size :] if self.shared_history else []
            ),
            shared_reflexions=(
                self.shared_reflexions[-self.history_size :]
                if self.shared_reflexions
                else []
            ),
        )

        # Collect history and reflexions from worker results
        for r in results:
            if r.get("attempt"):
                self.shared_history.append(r["attempt"])
            if r.get("reflexion"):
                self.shared_reflexions.append(r["reflexion"])

        # Log errors from failed workers
        for r in results:
            if not r.get("success") and r.get("error"):
                self.logger.error(
                    f"Worker {r.get('worker_id')} failed: {r.get('error')}"
                )
                if r.get("traceback"):
                    self.logger.debug(f"Traceback:\n{r.get('traceback')}")

        return results
