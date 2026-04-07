---
title: Incident RCA Env
sdk: docker
app_port: 7860
---

# IncidentRCAEnv

**OpenEnv-compliant environment for training AI agents to perform incident response and root cause analysis on production microservice systems.**

Built for the Meta × Hugging Face × PyTorch OpenEnv Hackathon 2026.

---

**Repository:** https://github.com/srujangowda07/openenv-incident-rca  
**Hugging Face Deployment:**

---

## Problem Statement

Production outages directly impact reliability, cost, and engineering productivity. Engineers must analyze logs, metrics, and dependencies under time pressure to identify root causes.

This environment models incident response using a controlled, deterministic microservice system. It enables training and evaluating agents that can diagnose failures through structured investigation rather than guesswork.

---

## Why This Task is Challenging

- **Partial observability** — agents do not see the full system state upfront
- **Multi-step reasoning** — diagnosis requires sequential investigation
- **Dependency tracing** — failures propagate across services
- **Noisy signals** — not all observations directly indicate the root cause
- **Efficiency trade-offs** — excessive actions reduce the final score

---

## Environment Design

### Observation Space

The agent receives a structured JSON observation at each step:

| Field | Type | Description |
|---|---|---|
| `step` | int | Current step number |
| `max_steps` | int | Step budget for the task |
| `task_id` | str | Task identifier |
| `task_description` | str | Incident description |
| `alerts` | list[dict] | Active alerts triggering investigation |
| `tool_result` | dict \| null | Output of the last action |
| `history` | list[dict] | Recent actions and outcomes |
| `available_actions` | list[str] | Valid actions |
| `done` | bool | Episode completion flag |

The environment is partially observable — agents must use actions to uncover relevant information. Observations are revealed incrementally based on what the agent investigates.

The environment does not expose the root cause directly — agents must infer it through interaction.

---

### Action Space

Five atomic actions represent real-world debugging tools:

| Action | Required Parameters | Description |
|---|---|---|
| `grep_logs` | `service`, `keyword` | Search logs for a service |
| `query_metrics` | `service`, `metric_name` | Retrieve a specific metric |
| `fetch_traces` | `request_id` | Retrieve a distributed trace |
| `query_dependencies` | `service` | Get upstream and downstream dependencies |
| `submit_diagnosis` | `root_cause_service`, `cause_type` | Submit final diagnosis (ends episode) |

All parameters are required and strictly validated. Invalid or missing parameters result in a `-0.10` penalty.

---

## Reward Function

Rewards are step-based and fully deterministic.

```
Action                           Reward    Condition
────────────────────────────────────────────────────────────────
submit_diagnosis (perfect)       +1.00     correct service and cause type
submit_diagnosis (partial)       +0.50     correct service, wrong cause type
submit_diagnosis (wrong)         -0.50     incorrect service

grep_logs        (cascade)       +0.05     service is in the failure cascade
query_metrics    (cascade)       +0.05     service is in the failure cascade
fetch_traces     (root cause)    +0.10     trace implicates root cause service
query_dependencies (cascade)     +0.05     service is in the failure cascade

repeated action                  -0.10     exact same call made twice
invalid action                   -0.10     missing required parameter or unknown service
step penalty                     -0.01     applied every step
```

This design ensures:
- agents must follow a correct reasoning path
- efficient behavior is rewarded
- reward hacking through repetition is prevented

---

## Tasks

### Easy — Single-service failures (≤3 services, 15 steps)

| Task | Scenario | Root Cause |
|---|---|---|
| `easy_001` | API gateway returning 502 errors | `postgres-primary` — connection pool exhausted |
| `easy_002` | Payment service pods crashing | `payment-service` — memory leak, unbounded cache |
| `easy_003` | Logging service not ingesting | `logging-service` — disk full, log rotation disabled |

---

### Medium — Dependency cascade (6–7 services, 25 steps)

| Task | Scenario | Root Cause |
|---|---|---|
| `medium_001` | Two services degraded simultaneously | `mysql-primary` — schema migration dropped index |

---

### Hard — Complex cascading failure (10 services, 40 steps, red herrings)

| Task | Scenario | Root Cause |
|---|---|---|
| `hard_001` | Five services failing simultaneously | `redis-cluster` — split-brain from network switch firmware upgrade |

---

## Grader

Evaluation is fully deterministic and reproducible. Ground truth is embedded in the scenario at generation time — no LLM grading.

| Dimension | Weight | Criteria |
|---|---|---|
| Root cause service | 0.50 | Exact match on service name |
| Cause type | 0.30 | Exact match (only awarded if service is also correct) |
| Tool evidence | 0.20 | Agent queried root cause service before diagnosing |
| Penalties | variable | `-0.10` per invalid action · `-0.20` for wrong diagnosis |

The grader evaluates both correctness and reasoning evidence — agents cannot achieve high scores through guessing.

**Pass threshold: 0.60 / 1.00**

---

## Score Distribution

| Agent Quality | Score Range |
|---|---|
| Bad agent (random / guessing) | 0.0 – 0.2 |
| Average agent (finds service, wrong cause) | 0.4 – 0.6 |
| Good agent (correct + evidence) | 0.7 – 0.9 |
| Perfect agent (correct + cause + efficient) | 0.95 – 1.0 |

---

## Design Principles

- **Deterministic** — same input produces the same output every run
- **Minimal** — five actions, no redundancy, clear semantics
- **Causal** — realistic cause–effect relationships across services
- **Measurable** — every action has a defined impact on score
- **Reproducible** — fixed seeds ensure consistent evaluation

---

## System Flow

1. Environment initializes a deterministic incident scenario
2. Agent observes alerts and limited context
3. Agent performs investigation actions (logs, metrics, traces, dependencies)
4. Environment reveals information incrementally based on each action
5. Agent submits final diagnosis
6. Grader evaluates correctness and reasoning evidence

This process mirrors real-world debugging workflows.

---

## Architecture

![Visual Architecture](assets/architecture.png)


## What Makes This Environment Unique

- Models real-world SRE debugging workflows
- Requires multi-step reasoning instead of single-shot answers
- Penalizes inefficient investigation strategies
- Fully deterministic and reproducible
- Designed for evaluating reasoning quality, not just correctness
- Supports plug-and-play LLM backends via OpenAI-compatible APIs
- Achieves consistent success across all tasks with strong LLMs (e.g., LLaMA 70B)

---

## Environment Variables

The following environment variables are required:

- `API_BASE_URL` — endpoint of the LLM provider
- `MODEL_NAME` — model identifier
- `HF_TOKEN` — API key for authentication
- `TASK_ID` — task identifier to run
- `SEED` — optional run seed

Valid `TASK_ID` values:
`easy_001`, `easy_002`, `easy_003`, `medium_001`, `hard_001`

Example `.env` configuration:
```env
API_BASE_URL=https://integrate.api.nvidia.com/v1
MODEL_NAME=meta/llama-3.3-70b-instruct
HF_TOKEN=your_nvidia_api_key

TASK_ID=easy_001
SEED=42
```

These variables allow the agent to use a pretrained LLM.

The environment supports any OpenAI-compatible API (e.g., NVIDIA NIM, OpenAI, HuggingFace Router).

---

## Inference (LLM Agent)

Run the environment with a language model agent:

```bash
python inference.py
```

- The agent uses a pretrained LLM
- It selects actions based on observations
- It does NOT perform training
- Requires API credentials

---

## Quickstart

```bash
git clone https://github.com/srujangowda07/openenv-incident-rca
cd incident-rca-env
pip install -r requirements.txt

# Smoke test — no API key needed
python baseline/run_baseline.py --task easy_001 --dry-run

# Run test suite
pytest tests/ -v

# Validate environment (must be 13/13)
python validate.py

# Run LLM agent (requires API credentials)
python inference.py
```

### Python SDK

```python
from environment.env import IncidentRCAEnv, ActionModel

env = IncidentRCAEnv(task_id="easy_001", seed=42)
obs = env.reset()

# Investigate
action = ActionModel(
    action_type="grep_logs",
    parameters={"service": "postgres-primary", "keyword": "connection"},
)
obs, reward, done, info = env.step(action)
print(f"Reward: {reward.total:+.3f} — {reward.reason}")

# Query a metric
action = ActionModel(
    action_type="query_metrics",
    parameters={"service": "postgres-primary", "metric_name": "active_connections"},
)
obs, reward, done, info = env.step(action)

# Submit diagnosis (ends episode)
action = ActionModel(
    action_type="submit_diagnosis",
    parameters={
        "root_cause_service": "postgres-primary",
        "cause_type": "connection pool exhausted",
    },
)
obs, reward, done, info = env.step(action)
print(f"Final reward: {reward.total:+.3f}")
print("Ground truth:", info.ground_truth_root_cause)
```

---

## Baseline Results

| Task | Score | Pass |
|---|---|---|
| `easy_001` | ~0.78–0.85 | ✓ |
| `easy_002` | ~0.74–0.80 | ✓ |
| `easy_003` | ~0.75–0.82 | ✓ |
| `medium_001` | ~0.60–0.68 | ✓ |
| `hard_001` | ~0.45–0.55 | ✗ |

Note: Baseline results use a smaller model and are intended to demonstrate task difficulty and evaluation sensitivity.

The lower performance on `hard_001` reflects the complexity of multi-service cascading failures and misleading signals, which require deeper reasoning and more efficient investigation strategies.

Stronger models (e.g., LLaMA 70B) consistently achieve successful diagnoses across all tasks.

These results validate that:
- simpler tasks are solvable with shallow reasoning
- harder tasks require structured multi-step investigation
- the environment effectively differentiates agent quality

*Run `python baseline/run_baseline.py --all` to reproduce.*

---

## Project Structure

```
incident-rca-env/
├── environment/
│   ├── env.py                  # Core OpenEnv API (reset / step / state)
│   ├── scenario_generator.py  # Deterministic synthetic incident generator
│   ├── reward_shaper.py       # All reward logic — single source of truth
│   ├── state_manager.py       # Mutable episode state
│   └── server.py              # FastAPI HTTP server for HF Spaces
├── graders/
│   └── grader.py              # Deterministic 0.0–1.0 episode grader
├── tasks/
│   └── task_definitions.py    # Task metadata and descriptions
├── baseline/
│   └── run_baseline.py        # LLM baseline inference script
├── tests/
│   └── test_env.py            # Full pytest test suite (30 tests)
├── openenv.yaml               # OpenEnv spec compliance
├── validate.py                # Pre-submission validation (13/13)
├── Dockerfile
└── README.md
```

---

## Why This Environment Matters

This project focuses on evaluating reasoning quality, not just final answers. It provides a structured, reproducible benchmark for training agents that can navigate multi-step diagnostic workflows — the same workflows on-call engineers perform under pressure during production incidents.

A well-trained agent that reduces mean time to resolution, even marginally, translates directly to fewer outages, lower costs, and less engineer burnout.

This environment prioritizes reliable evaluation over complexity, making it a strong benchmark for reasoning-driven agents.
