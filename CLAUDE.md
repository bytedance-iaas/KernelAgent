# KernelAgent — Project Instructions

## Pre-Flight Check (before ka-kernel-parser / ka-kernel-gen / ka-kernel-opt)

These skills depend on git submodules (kernel repos, companion skills) and a
CodeGraph index that covers them. Before executing any of them, verify:

1. Submodules initialized: `git submodule status --recursive | grep '^-'`
   prints nothing.
2. CodeGraph indexed: `.codegraph/` exists and `codegraph status` is healthy.
   If missing, fall back to grep for searches and tell the user to run the
   setup.

If either check fails, follow `docs/SETUP.md` (submodules FIRST, then
`codegraph init` — the order matters so the index covers the submodules).

## Standing Rules

- Never commit `.codegraph/codegraph.db` (machine-local, 150+ MB, already
  gitignored).
- After pulling submodule updates, the CodeGraph index is stale — rebuild
  with `codegraph index`.
