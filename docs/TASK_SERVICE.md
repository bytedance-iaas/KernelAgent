# KernelAgent Single-Node Task Service

`kernelagent_service` exposes the repository's agent skills through Claude
Code, pi, or Codex as an asynchronous HTTP service. The first implementation
intentionally uses an in-process `asyncio.Queue`: one coroutine owns each
configured GPU and pulls the next task only after its previous task exits.

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

Install every coding-agent CLI you plan to select in task requests and make it
available on `PATH`:

- `claude` (the default) and `pi` use the Anthropic Messages API. Install pi
  with `npm install -g @earendil-works/pi-coding-agent` when needed. An
  authentication-free endpoint can ignore the dummy bearer token.
- `codex` uses the OpenAI Responses API (`/v1/responses`). Supporting only
  `/v1/messages` or `/v1/chat/completions` is not sufficient. The generated
  provider has `requires_openai_auth = false`, so no OpenAI API key or
  interactive login is required.

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
| `KERNEL_AGENT_CLAUDE_COMMAND` | `claude` | Claude Code executable |
| `KERNEL_AGENT_CODEX_COMMAND` | `codex` | Codex executable |
| `KERNEL_AGENT_CODEX_BASE_URL` | `<model-base-url>/v1` | Codex Responses API root |
| `KERNEL_AGENT_CODEX_SANDBOX` | `workspace-write` | Codex sandbox: `read-only`, `workspace-write`, or `danger-full-access` |
| `KERNEL_AGENT_MODEL_AUTH_TOKEN` | `dummy` | Bearer token sent to the model gateway |
| `KERNEL_AGENT_QUEUE_CAPACITY` | `100` | Maximum queued tasks |
| `KERNEL_AGENT_TASK_TIMEOUT_SECONDS` | `7200` | Default agent wall-clock timeout |
| `KERNEL_AGENT_SHUTDOWN_GRACE_SECONDS` | `10` | Seconds between SIGTERM and SIGKILL |
| `KERNEL_AGENT_RUNS_DIR` | `.kernel_agent_service/runs` | Persistent task directory |
| `KERNEL_AGENT_SKILLS_DIR` | `<repo>/.claude/skills` | Project skills exposed to each task |
| `KERNEL_AGENT_MAX_INPUT_BYTES` | `10485760` | Combined UTF-8 request input limit |
| `KERNEL_AGENT_MAX_ARTIFACT_BYTES` | `52428800` | Per-file artifact limit |
| `KERNEL_AGENT_SERVICE_HOST` | `127.0.0.1` | Listen address |
| `KERNEL_AGENT_SERVICE_PORT` | `8080` | Listen port |
| `KERNEL_AGENT_AUTH_ENABLED` | `1` | Require login for the UI and task API |
| `KERNEL_AGENT_USERS_FILE` | `.kernel_agent_service/users.json` | Local credential and role file |
| `KERNEL_AGENT_SESSION_TTL_SECONDS` | `86400` | Signed login-cookie lifetime |
| `KERNEL_AGENT_ADMIN_USERNAME` | `admin` | Provisioned administrator username |
| `KERNEL_AGENT_ADMIN_PASSWORD` | `kernelagent-admin` | Provisioned administrator password |
| `KERNEL_AGENT_PI_COMMAND` | `pi` | pi executable (only used when the agent is `pi`) |
| `KERNEL_AGENT_PI_CONTEXT_WINDOW` | `200000` | Context window (tokens) advertised to pi for the gateway model |
| `KERNEL_AGENT_PI_MAX_TOKENS` | `16384` | Max output tokens advertised to pi for the gateway model |

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

### Accounts and roles

Open `/v1/ui` in a browser. On a fresh installation, it redirects to the
login/signup page. Every self-service signup is assigned the `general` role
and succeeds when its username is unique and its password meets the minimum
length. General users can access the task UI and task API, while admins can
also access `/v1/console`.

On first startup the service creates a default admin with username `admin` and
password `kernelagent-admin`. The login page is shared, so the service
determines the role from the matching username and password; there is no admin
signup choice. Override the pair before a non-local deployment:

```bash
export KERNEL_AGENT_ADMIN_USERNAME=my-admin
export KERNEL_AGENT_ADMIN_PASSWORD='use-a-long-private-password'
```

The user file contains usernames, roles, salted PBKDF2 password hashes, and a
random session-signing key. It is created with owner-only permissions. Back it
up like other service state, and never commit it. To use the API from `curl`,
log in with a cookie jar first:

```bash
curl -c cookies.txt http://127.0.0.1:8080/v1/auth/login \
  -H 'content-type: application/json' \
  -d '{"username":"my-user","password":"my-password"}'
curl -b cookies.txt http://127.0.0.1:8080/v1/tasks
```

Authentication can be disabled for isolated development with
`KERNEL_AGENT_AUTH_ENABLED=0`.

### Choosing a runner per task

The service registers Claude Code, pi, and Codex at startup. Select the harness
runner with the request's `runner_backend` field; no service restart is needed.
The default is `claude` when the field is omitted.

pi loads the same skills as Claude Code directly from `KERNEL_AGENT_SKILLS_DIR`
(pi implements the Agent Skills standard used by `.claude/skills`), so no
skill changes are needed to switch agents. Because pi has no `--base-url`
flag for arbitrary OpenAI/Anthropic-compatible endpoints, the service
registers the configured model gateway as a custom `anthropic-messages`
provider in a per-task `models.json`, isolated under each task's workspace
(`PI_CODING_AGENT_DIR`) the same way Claude Code gets an isolated
`CLAUDE_CONFIG_DIR`. Since pi has no `--json-schema` equivalent either, the
service asks pi in-prompt to end its final message with a fenced JSON code
block and parses that out of the reply; a run that finishes without one is
reported as failed.

Codex uses the configured self-hosted SGLang Responses endpoint:

```bash
export KERNEL_AGENT_MODEL_BASE_URL=http://127.0.0.1:30000
export KERNEL_AGENT_MODEL=my-sglang-model
```

The service derives the provider root as
`http://127.0.0.1:30000/v1`; set `KERNEL_AGENT_CODEX_BASE_URL` when the exact
Responses-compatible API root differs. Each task receives an isolated
`CODEX_HOME`, and Codex discovers the shared project skills through
`.agents/skills`.

## Submit a PyTorch-to-GPU-kernel task

```bash
curl -sS http://127.0.0.1:8080/v1/tasks \
  -H 'content-type: application/json' \
  -d '{
    "pytorch_code": "import torch\nfrom torch import nn\n\nclass Model(nn.Module):\n    def forward(self, x, y):\n        return x + y\n\ndef get_inputs():\n    return [torch.randn(1048576, device=\"cuda\"), torch.randn(1048576, device=\"cuda\")]\n\ndef get_init_inputs():\n    return []\n",
    "runner_backend": "claude",
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
| `runner_backend` | no | `claude` | Harness runner: `claude`, `pi`, or `codex` |
| `kernel_language` | no | `triton` | Target language: `triton` or `cutedsl` |
| `test_code` | no | — | Additional Python correctness test |
| `max_rounds` | no | `5` | Maximum generation/refinement rounds |
| `timeout_seconds` | no | service default | Per-task wall-clock timeout, 30–86400 seconds |
| `extra_instructions` | no | — | Extra optimization constraints or hints |

The service materializes the reference as `input/problem.py` and invokes
`/ka-kernel-gen input/problem.py` for Claude Code,
`/skill:ka-kernel-gen input/problem.py` for pi, or
`$ka-kernel-gen input/problem.py` for Codex. It forbids PyTorch fallback in the
generated implementation and asks the skill to benchmark and refine rather
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
`timed_out`, and `lost`. Every task record includes `runner_backend`, recording
the agent selected at submission. Canceling a running task terminates the agent
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
    ├── .claude/skills -> <skills-dir>
    ├── .agents/skills -> <skills-dir>
    ├── .claude-runtime/
    ├── .codex-runtime/
    ├── .pi-runtime/
    ├── .fuse/
    ├── .optimize/
    └── generated artifacts
```

After a service restart, persisted `queued` tasks are re-enqueued. A task that
was `running` is marked `lost`; it is not retried automatically because its GPU
process and partially written artifacts cannot be assumed safe.

## Security boundary

The agent CLI and generated kernels execute arbitrary shell/Python/GPU code.
The service creates separate task directories and enables Codex's
`workspace-write` sandbox by default, but this is not a complete hostile-code
sandbox. Before accepting untrusted public requests, run each task in a
non-root container or VM with a read-only base repository, a task-only writable
mount, explicit GPU assignment, resource limits, restricted network egress,
and no host credentials or Docker socket. Use
`KERNEL_AGENT_CODEX_SANDBOX=danger-full-access` only inside such an external
isolation boundary.
