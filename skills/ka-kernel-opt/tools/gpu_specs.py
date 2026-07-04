#!/usr/bin/env python3
"""
GPU specifications lookup for kernel optimization analysis.

Standalone: stdlib only (torch is used only for --detect, if available).
Vendored from kernel_perf_agent/kernel_opt/diagnose_prompt/gpu_specs_database.py
— keep in sync.

Usage:
    python gpu_specs.py --name "NVIDIA H100 NVL 94GB"
    python gpu_specs.py --detect          # auto-detect via torch.cuda
    python gpu_specs.py --list

Output:
    JSON specs dict to stdout (or a JSON list for --list).
"""

from __future__ import annotations

import argparse
import json
import sys

GPU_SPECS_DATABASE: dict[str, dict] = {
    "NVIDIA A100 SXM4 40GB": {
        "name": "NVIDIA A100 SXM4 40GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,
        "peak_bf16_tflops": 312.0,
        "peak_memory_bw_gbps": 1555,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 40,
        "memory_type": "HBM2e",
        "form_factor": "SXM4",
        "tdp_w": 400,
    },
    "NVIDIA A100 SXM4 80GB": {
        "name": "NVIDIA A100 SXM4 80GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,
        "peak_bf16_tflops": 312.0,
        "peak_memory_bw_gbps": 2039,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "form_factor": "SXM4",
        "tdp_w": 400,
    },
    "NVIDIA A100 PCIe 40GB": {
        "name": "NVIDIA A100 PCIe 40GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,
        "peak_bf16_tflops": 312.0,
        "peak_memory_bw_gbps": 1555,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 40,
        "memory_type": "HBM2e",
        "form_factor": "PCIe",
        "tdp_w": 250,
    },
    "NVIDIA A100 PCIe 80GB": {
        "name": "NVIDIA A100 PCIe 80GB",
        "architecture": "Ampere",
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,
        "peak_bf16_tflops": 312.0,
        "peak_memory_bw_gbps": 1935,
        "sm_count": 108,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 192,
        "l2_cache_mb": 40,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "form_factor": "PCIe",
        "tdp_w": 300,
    },
    "NVIDIA H100 SXM5 80GB": {
        "name": "NVIDIA H100 SXM5 80GB",
        "architecture": "Hopper",
        "peak_fp32_tflops": 67.0,
        "peak_fp16_tflops": 1979.0,
        "peak_bf16_tflops": 1979.0,
        "peak_memory_bw_gbps": 3350,
        "sm_count": 132,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 80,
        "memory_type": "HBM3",
        "form_factor": "SXM5",
        "tdp_w": 700,
    },
    "NVIDIA H100 PCIe 80GB": {
        "name": "NVIDIA H100 PCIe 80GB",
        "architecture": "Hopper",
        "peak_fp32_tflops": 51.0,
        "peak_fp16_tflops": 1513.0,
        "peak_bf16_tflops": 1513.0,
        "peak_memory_bw_gbps": 2000,
        "sm_count": 114,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "form_factor": "PCIe",
        "tdp_w": 350,
    },
    "NVIDIA H100 NVL 94GB": {
        "name": "NVIDIA H100 NVL 94GB",
        "architecture": "Hopper",
        "peak_fp32_tflops": 60.0,
        "peak_fp16_tflops": 1671.0,
        "peak_bf16_tflops": 1671.0,
        "peak_memory_bw_gbps": 3900,
        "sm_count": 132,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 94,
        "memory_type": "HBM3",
        "form_factor": "PCIe",
        "tdp_w": 400,
    },
    "NVIDIA H200 SXM 141GB": {
        "name": "NVIDIA H200 SXM 141GB",
        "architecture": "Hopper",
        "peak_fp32_tflops": 67.0,
        "peak_fp16_tflops": 989.5,
        "peak_bf16_tflops": 989.5,
        "peak_memory_bw_gbps": 4800,
        "sm_count": 132,
        "max_threads_per_sm": 2048,
        "l1_cache_kb": 256,
        "l2_cache_mb": 50,
        "memory_gb": 141,
        "memory_type": "HBM3e",
        "form_factor": "SXM",
        "tdp_w": 700,
    },
    "NVIDIA RTX 4090": {
        "name": "NVIDIA RTX 4090",
        "architecture": "Ada Lovelace",
        "peak_fp32_tflops": 82.58,
        "peak_fp16_tflops": 82.58,
        "peak_bf16_tflops": 82.58,
        "peak_memory_bw_gbps": 1008,
        "sm_count": 128,
        "max_threads_per_sm": 1536,
        "l1_cache_kb": 128,
        "l2_cache_mb": 72,
        "memory_gb": 24,
        "memory_type": "GDDR6X",
        "form_factor": "PCIe",
        "tdp_w": 450,
    },
    "NVIDIA RTX 5080": {
        "name": "NVIDIA RTX 5080",
        "architecture": "Blackwell",
        "peak_fp32_tflops": 56.28,
        "peak_fp16_tflops": 56.28,
        "peak_bf16_tflops": 56.28,
        "peak_memory_bw_gbps": 960,
        "sm_count": 84,
        "max_threads_per_sm": 1536,
        "l1_cache_kb": 128,
        "l2_cache_mb": 64,
        "memory_gb": 16,
        "memory_type": "GDDR7",
        "form_factor": "PCIe",
        "tdp_w": 360,
    },
}


def lookup(name: str) -> dict | None:
    """Look up specs by exact name, then case-insensitive substring match."""
    if name in GPU_SPECS_DATABASE:
        return dict(GPU_SPECS_DATABASE[name])
    lowered = name.lower()
    matches = [
        k for k in GPU_SPECS_DATABASE
        if lowered in k.lower() or k.lower() in lowered
    ]
    if len(matches) == 1:
        return dict(GPU_SPECS_DATABASE[matches[0]])
    # Fuzzy token match: all tokens of one name appear in the other
    tokens = set(lowered.replace("-", " ").split())
    scored = []
    for k in GPU_SPECS_DATABASE:
        k_tokens = set(k.lower().replace("-", " ").split())
        overlap = len(tokens & k_tokens)
        if overlap >= 2:
            scored.append((overlap, k))
    if scored:
        scored.sort(reverse=True)
        return dict(GPU_SPECS_DATABASE[scored[0][1]])
    return None


def detect() -> dict | None:
    """Detect the local GPU via torch and match against the database."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        return None
    specs = lookup(device_name)
    if specs is not None:
        specs["detected_device_name"] = device_name
    return specs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GPU specifications lookup")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--name", help="GPU name (exact or fuzzy match)")
    g.add_argument("--detect", action="store_true", help="Auto-detect via torch.cuda")
    g.add_argument("--list", action="store_true", help="List known GPUs")
    args = p.parse_args(argv)

    if args.list:
        print(json.dumps(sorted(GPU_SPECS_DATABASE.keys()), indent=2))
        return 0

    specs = detect() if args.detect else lookup(args.name)
    if specs is None:
        print(
            json.dumps(
                {
                    "error": "GPU not found in database",
                    "requested": None if args.detect else args.name,
                    "available": sorted(GPU_SPECS_DATABASE.keys()),
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(specs, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
