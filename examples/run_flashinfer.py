#!/usr/bin/env python3
"""FlashInfer contest kernel optimizer.

Generates a KernelAgent problem directory from a flashinfer-bench Definition,
runs KernelAgent's beam-search optimization, then packages the best kernel
into a flashinfer-bench solution.json.

Usage:
    # List available contest kernels
    python examples/run_flashinfer.py --list

    # Optimize a specific definition
    python examples/run_flashinfer.py \\
        --definition moe_fp8_block_scale_e256_h7168_i2048_topk8 \\
        --max-rounds 10 \\
        --output-dir /tmp/flashinfer_solutions

    # Use a custom strategy config
    python examples/run_flashinfer.py \\
        --definition gdn_decode_b16_qh32_kvh8_qkd128_vd128_kv4096 \\
        --strategy flashinfer_beam_search \\
        --max-rounds 5
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _do_list() -> None:
    from flashinfer.definitions import DEFINITIONS, list_definitions

    print("Available FlashInfer contest kernel definitions:")
    print()
    for key in list_definitions():
        spec = DEFINITIONS[key]
        print(f"  {key}")
        print(f"    {spec.description}")
        print()


def _optimize(
    definition_key: str,
    strategy: str,
    max_rounds: int,
    output_dir: Path,
    author: str,
    work_dir: Path | None,
) -> None:
    from flashinfer.adapter import FlashInferProblemAdapter
    from flashinfer.definitions import DEFINITIONS
    from flashinfer.solution_builder import SolutionBuilder
    from triton_kernel_agent.opt_manager import OptimizationManager, print_metrics

    if definition_key not in DEFINITIONS:
        print(f"ERROR: unknown definition '{definition_key}'")
        print(f"Run with --list to see available definitions.")
        sys.exit(1)

    spec = DEFINITIONS[definition_key]

    # ------------------------------------------------------------------
    # 1. Generate problem directory
    # ------------------------------------------------------------------
    if work_dir is None:
        work_dir = output_dir / "workdir" / definition_key
    work_dir = work_dir.resolve()

    print("=" * 80)
    print("FlashInfer Kernel Optimizer")
    print("=" * 80)
    print(f"Definition:  {definition_key}")
    print(f"Description: {spec.description}")
    print(f"Strategy:    {strategy}")
    print(f"Max rounds:  {max_rounds}")
    print(f"Work dir:    {work_dir}")
    print(f"Output dir:  {output_dir}")
    print()

    adapter = FlashInferProblemAdapter(spec)
    problem_file = adapter.write_problem_dir(work_dir)
    print(f"Problem directory written to: {work_dir}")

    # ------------------------------------------------------------------
    # 2. Locate strategy config
    # ------------------------------------------------------------------
    config_path = _CONFIGS_DIR / f"{strategy}.yaml"
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        print(f"Available configs: {[p.stem for p in _CONFIGS_DIR.glob('*.yaml')]}")
        sys.exit(1)

    import tempfile
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Extract FlashInfer-specific keys before they leak into OptimizationWorker kwargs
    _FLASHINFER_KEYS = {"flashinfer_target_hardware"}
    target_hardware = cfg.get("flashinfer_target_hardware", "nvidia_h200")
    worker_cfg = {k: v for k, v in cfg.items() if k not in _FLASHINFER_KEYS}

    # Write filtered config to a temp yaml so config_injectable only sees worker keys
    _tmp_cfg = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    )
    _tmp_cfg_path = Path(_tmp_cfg.name)
    _tmp_cfg.write(yaml.dump(worker_cfg))
    _tmp_cfg.close()

    print_metrics(
        "Optimization Configuration",
        {
            "Config file":        str(config_path),  # original (flashinfer keys stripped for worker)
            "Strategy":           cfg.get("strategy", strategy),
            "Model":              cfg.get("openai_model", "(default)"),
            "Num workers":        cfg.get("num_workers", 4),
            "Max rounds":         max_rounds,
            "GPU":                cfg.get("gpu_name", "(default)"),
            "Target hardware":    target_hardware,
            "Benchmark warmup":   cfg.get("benchmark_warmup", "(default)"),
            "Benchmark repeat":   cfg.get("benchmark_repeat", "(default)"),
        },
    )

    # ------------------------------------------------------------------
    # 3. Run optimization
    # ------------------------------------------------------------------
    kernel_code = (work_dir / "input.py").read_text()
    test_code   = (work_dir / "test.py").read_text()
    log_dir     = work_dir / "opt_manager_logs"

    manager = OptimizationManager(
        config=str(_tmp_cfg_path),
        log_dir=log_dir / strategy,
        database_path=log_dir / strategy / "program_db.json",
    )

    result = manager.run_optimization(
        initial_kernel=kernel_code,
        problem_file=problem_file,
        test_code=test_code,
        max_rounds=max_rounds,
    )

    # ------------------------------------------------------------------
    # 4. Build solution.json
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    if result["success"]:
        optimized_kernel = result["kernel_code"]
        print()
        print("=" * 80)
        print("OPTIMIZATION SUCCESSFUL")
        print("=" * 80)
        print(f"Best time:    {result['best_time_ms']:.4f} ms")
        print(f"Total rounds: {result['total_rounds']}")

        # Save optimized kernel alongside the solution
        kernel_out = output_dir / f"{definition_key}_kernel.py"
        kernel_out.write_text(optimized_kernel)
        print(f"Kernel saved: {kernel_out}")
    else:
        print()
        print("=" * 80)
        print("OPTIMIZATION DID NOT IMPROVE — using initial kernel as baseline")
        print("=" * 80)
        optimized_kernel = kernel_code

    builder = SolutionBuilder(
        definition_key=definition_key,
        kernel_code=optimized_kernel,
        author=author,
        name=f"{definition_key} ({strategy})",
        target_hardware=target_hardware,
    )
    solution_path = builder.save(output_dir / definition_key / "solution.json")
    print(f"Solution JSON: {solution_path}")

    # Validate solution JSON is loadable
    import json
    sol = json.loads(solution_path.read_text())
    print()
    print_metrics(
        "Solution Summary",
        {
            "Name":                sol["name"],
            "Definition":          sol["definition"],
            "Author":              sol["author"],
            "Language":            sol["spec"]["language"],
            "Target hardware":     sol["spec"]["target_hardware"],
            "Entry point":         sol["spec"]["entry_point"],
            "Destination passing": str(sol["spec"]["destination_passing_style"]),
            "Source files":        str(len(sol["sources"])),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Optimize a FlashInfer contest kernel with KernelAgent and produce
            a flashinfer-bench solution.json.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available FlashInfer contest kernel definitions and exit",
    )
    parser.add_argument(
        "--definition",
        "-d",
        default="",
        help="FlashInfer definition key to optimize",
    )
    parser.add_argument(
        "--strategy",
        default="flashinfer_beam_search",
        help="Strategy config name under examples/configs/ (default: flashinfer_beam_search)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum optimization rounds (default: 10)",
    )
    _default_output = Path(__file__).resolve().parent.parent / "flashinfer_solutions"
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output,
        help=f"Directory to write solution.json files (default: {_default_output})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Override work directory for generated problem files",
    )
    parser.add_argument(
        "--author",
        default="kernelAgent",
        help="Author string in solution.json (default: kernelAgent)",
    )

    args = parser.parse_args()

    if args.list:
        _do_list()
        return

    if not args.definition:
        parser.error("--definition is required (or use --list to see options)")

    _optimize(
        definition_key=args.definition,
        strategy=args.strategy,
        max_rounds=args.max_rounds,
        output_dir=args.output_dir.resolve(),
        author=args.author,
        work_dir=args.work_dir,
    )


if __name__ == "__main__":
    main()
