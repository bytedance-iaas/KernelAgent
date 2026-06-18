---
name: direct-kernel
description: "Generate a kernel directly from a problem description, bypassing the Fuser pipeline. Best for simple/linear operations. Supports triton, tilelang, and cutedsl backends."
allowed-tools:
  - Bash
---

# Skill: Direct Triton Kernel Generation

You have been invoked to generate a kernel directly. Your complete, detailed instructions for this task are located in the project file:
`skillset/skills/06_direct_kernel.md`

**CRITICAL INSTRUCTION:**
Do not proceed until you have read the contents of `skillset/skills/06_direct_kernel.md`.
The user's input arguments (the problem path or description) are: `$ARGUMENTS`

Follow the generation, testing, and refinement rules defined in that file exactly.
