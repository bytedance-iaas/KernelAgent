---
name: refine-kernel
description: "Iteratively test and fix a kernel until it passes verification. Reads error output and generates refined implementations. Supports triton, tilelang, and cutedsl backends."
allowed-tools:
  - Bash
---

# Skill: Iterative Kernel Refinement

You have been invoked to refine a failing kernel. Your complete, detailed instructions for this task are located in the project file:
`skillset/skills/04_refine_kernel.md`

**CRITICAL INSTRUCTION:**
Do not proceed until you have read the contents of `skillset/skills/04_refine_kernel.md`.
The user's input arguments (the path to the working directory) are: `$ARGUMENTS`

Follow the verification, PyTorch constraint checking, and refinement loops defined in that file exactly.
