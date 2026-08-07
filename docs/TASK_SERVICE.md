# KernelAgent Single-Node Task Service

`kernelagent_service` exposes the repository's Claude Code skills as an
asynchronous HTTP service. The first implementation intentionally uses an
in-process `asyncio.Queue`: one coroutine owns each configured GPU and pulls the
next task only after its previous task exits.

The submission contract mirrors KernelAgent's normal generation input: provide
a KernelBench-style PyTorch reference with concrete inputs, then select Triton
or CuTe DSL as the target. The worker generates, verifies, benchmarks, and
refines the GPU implementation before returning its artifacts.

This is a single-process service. Run exactly one Uvicorn worker. Multiple API
processes would create independent queues and could assign the same GPU twice.

## Prerequisites

Initialize the skill and reference submodules before building the worker image:

```bash
git submodule update --init --recursive
codegraph init
pip install -e .
```

Claude Code must be installed and available on `PATH`. The configured model
endpoint must implement the Anthropic Messages API and tool-use loop expected
by Claude Code. An authentication-free endpoint can ignore the dummy bearer
token sent by the service.

## Configuration

Required for a normal GPU deployment:

```bash
export KERNEL_AGENT_GPU_IDS=0,1
export KERNEL_AGENT_MODEL_BASE_URL=http://127.0.0.1:30000
export KERNEL_AGENT_MODEL=my-sglang-model
```

Useful optional settings:

| Variable | Default | Meaning |
|---|---:|---|
| `KERNEL_AGENT_MODEL_AUTH_TOKEN` | `dummy` | Bearer token sent to the model gateway |
| `KERNEL_AGENT_QUEUE_CAPACITY` | `100` | Maximum queued tasks |
| `KERNEL_AGENT_TASK_TIMEOUT_SECONDS` | `7200` | Default Claude Code wall-clock timeout |
| `KERNEL_AGENT_SHUTDOWN_GRACE_SECONDS` | `10` | Seconds between SIGTERM and SIGKILL |
| `KERNEL_AGENT_RUNS_DIR` | `.kernel_agent_service/runs` | Persistent task directory |
| `KERNEL_AGENT_SKILLS_DIR` | `<repo>/.claude/skills` | Project skills exposed to each task |
| `KERNEL_AGENT_MAX_INPUT_BYTES` | `10485760` | Combined UTF-8 request input limit |
| `KERNEL_AGENT_MAX_ARTIFACT_BYTES` | `52428800` | Per-file artifact limit |
| `KERNEL_AGENT_SERVICE_HOST` | `127.0.0.1` | Listen address |
| `KERNEL_AGENT_SERVICE_PORT` | `8080` | Listen port |

When `KERNEL_AGENT_GPU_IDS` is absent, the service checks
`CUDA_VISIBLE_DEVICES`, then queries GPU UUIDs through `nvidia-smi`. If no GPU
is found, health is `degraded` and task submission returns HTTP 503.

## Run

```bash
python -m kernelagent_service

# Equivalent after editable installation
kernel-agent-service
```

Do not add `--workers 2` (or more). The queue and GPU ownership are local to
the process.

## Submit a PyTorch-to-GPU-kernel task

```bash
curl -sS http://127.0.0.1:8080/v1/tasks \
  -H 'content-type: application/json' \
  -d '{
    "pytorch_code": "import torch\nfrom torch import nn\n\nclass Model(nn.Module):\n    def forward(self, x, y):\n        return x + y\n\ndef get_inputs():\n    return [torch.randn(1048576, device=\"cuda\"), torch.randn(1048576, device=\"cuda\")]\n\ndef get_init_inputs():\n    return []\n",
    "kernel_language": "triton",
    "max_rounds": 5
  }'
```

`pytorch_code` must be valid Python and contain:

- `class Model(nn.Module)` implementing the PyTorch reference in `forward()`
- `get_inputs()` returning concrete CUDA input tensors
- optionally `get_init_inputs()` returning constructor arguments for `Model`

Request fields:

| Field | Required | Default | Meaning |
|---|---|---|---|
| `pytorch_code` | yes | — | KernelBench-style PyTorch reference |
| `kernel_language` | no | `triton` | Target language: `triton` or `cutedsl` |
| `test_code` | no | — | Additional Python correctness test |
| `max_rounds` | no | `5` | Maximum generation/refinement rounds |
| `timeout_seconds` | no | service default | Per-task wall-clock timeout, 30–86400 seconds |
| `extra_instructions` | no | — | Extra optimization constraints or hints |

The service materializes the reference as `input/problem.py`, explicitly
invokes `/ka-kernel-gen input/problem.py`, forbids PyTorch fallback in the
generated implementation, and asks the skill to benchmark and refine rather
than stopping at the first correct candidate.

## Query, events, artifacts, and cancellation

```bash
curl -sS http://127.0.0.1:8080/v1/tasks
curl -sS 'http://127.0.0.1:8080/v1/tasks?status=queued&status=running&offset=0&limit=100'
curl -sS http://127.0.0.1:8080/v1/tasks/<task-id>
curl -sS 'http://127.0.0.1:8080/v1/tasks/<task-id>/events?after=0&limit=100'
curl -OJ http://127.0.0.1:8080/v1/tasks/<task-id>/artifacts/<artifact-id>
curl -X POST http://127.0.0.1:8080/v1/tasks/<task-id>/cancel
```

The task-list endpoint is ordered newest first. Repeat `status` to select more
than one state; omit it to include every state. `offset` defaults to `0` and
`limit` defaults to `100` (maximum `1000`).

Statuses are `queued`, `running`, `succeeded`, `failed`, `canceled`,
`timed_out`, and `lost`. Canceling a running task terminates the Claude Code
process group, including child compiler, Python, and profiler processes.

## Persistence and recovery

Queue membership is in memory, while task metadata and outputs are written to:

```text
<runs-dir>/<task-id>/
├── request.json
├── task.json
├── logs/
│   ├── events.jsonl
│   └── stderr.log
└── workspace/
    ├── input/
    ├── .fuse/
    ├── .optimize/
    └── generated artifacts
```

After a service restart, persisted `queued` tasks are re-enqueued. A task that
was `running` is marked `lost`; it is not retried automatically because its GPU
process and partially written artifacts cannot be assumed safe.

## Security boundary

Claude Code and generated kernels execute arbitrary shell/Python/GPU code. The
service creates separate task directories and restricts the Claude tool set,
but this is not a hostile-code sandbox. Before accepting untrusted public
requests, run each task in a non-root container or VM with a read-only base
repository, a task-only writable mount, explicit GPU assignment, resource
limits, restricted network egress, and no host credentials or Docker socket.
