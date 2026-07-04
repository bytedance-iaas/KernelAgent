# KernelAgent Setup — Submodules and CodeGraph

One-time setup required before using the kernel pipeline skills
(`ka-kernel-parser`, `ka-kernel-gen`, `ka-kernel-opt`). Run these steps in
this exact order after cloning (Step 2 depends on Step 1's content being on
disk).

## Step 1: Initialize all submodules (recursively)

```bash
git submodule update --init --recursive
```

This fetches (large — allow time on first run):

| Submodule | Purpose |
|---|---|
| `reference/cuda/flash-attention` | Real production kernels — `ka-kernel-opt` rewrite patterns |
| `reference/cuda/flashinfer` | Real production kernels — `ka-kernel-opt` rewrite patterns |
| `reference/cuda/sgl-DeepGEMM` | Real production kernels — `ka-kernel-opt` rewrite patterns |
| `skills/KernelWiki` | Hopper/Blackwell optimization knowledge skill |
| `skills/ncu-report-skill` | Deep NCU profiling methodology skill |
| `skills/ptx-isa-markdown` | PTX ISA reference + `cuda_skill` |
| `examples/KernelBench` | KernelBench problem-format reference for `ka-kernel-parser` |

**Verify:**

```bash
git submodule status --recursive | grep '^-' && echo "MISSING SUBMODULES" || echo "OK"
ls reference/cuda/flashinfer | head   # must be non-empty
```

## Step 2: Initialize CodeGraph (AFTER Step 1)

```bash
codegraph init    # from the repo root
```

- Must run **after** submodules are populated so the index covers the
  third-party kernel sources — the skills use CodeGraph for structural
  search over those repos.
- The first index over all submodules takes several minutes.
- The index lives in `.codegraph/` (150+ MB, machine-local). It is
  gitignored — **never commit `codegraph.db`**; every machine builds its
  own.
- If `codegraph` is not installed, install it first (then re-run
  `codegraph init`).

**Verify:**

```bash
codegraph status                                  # healthy index
codegraph explore "swizzle in DeepGEMM"           # returns reference/cuda symbols
```

## Step 3: Keep the index fresh

- Normal edits are picked up automatically by the file watcher (~1s lag) —
  nothing to do.
- After pulling **submodule updates** (or adding submodules), rebuild:

```bash
git submodule update --init --recursive && codegraph index
```

## Troubleshooting

- `codegraph explore` finds nothing under `reference/cuda/` → the index was
  built before submodules were initialized; run `codegraph index`.
- Broken symlinks in `.claude/skills/` → Step 1 was skipped (the skill
  directories are submodules).
- Stale lock blocking indexing → `codegraph unlock`.
