---
name: generate-kernel
description: "End-to-end kernel generation from a PyTorch problem file. Auto-routes between direct or Fuser pipeline. Supports triton, tilelang, and cutedsl backends."
allowed-tools:
  - Bash
---

# Skill: Main Kernel Generation Orchestrator

You have been invoked to generate a kernel. Your complete, detailed instructions for this task are located in the project file:
`skillset/skills/main_kernel_gen.md`

**CRITICAL INSTRUCTION:**
Do not proceed until you have read the contents of `skillset/skills/main_kernel_gen.md`.
The user's input arguments (e.g., the problem path) are: `$ARGUMENTS`

Follow the routing and execution steps defined in that file exactly. It will instruct you to read other files in `skillset/skills/` as you progress through the pipeline.
