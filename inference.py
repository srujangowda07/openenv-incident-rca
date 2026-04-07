from __future__ import annotations
import json
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from environment.env import IncidentRCAEnv, ActionModel
from graders.grader import IncidentRCAGrader
from tasks.task_definitions import get_task

API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta/llama-3.3-70b-instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
TASK_ID      = os.getenv("TASK_ID", "easy_001")
SEED         = int(os.getenv("SEED", "42"))
ENV_NAME     = "incident-rca-env"

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer performing incident response.
Diagnose the root cause service and failure type, then submit your diagnosis.

Allowed cause_type values:
- "connection pool exhausted"
- "memory leak OOMKilled"
- "disk full no log rotation"
- "missing index slow query full table scan"
- "cluster split-brain 2 nodes network partition"

Available actions — respond ONLY with valid JSON:

grep_logs:
{"action_type": "grep_logs", "parameters": {"service": "<name>", "keyword": "<term>"}}

query_metrics:
{"action_type": "query_metrics", "parameters": {"service": "<name>", "metric_name": "<metric>"}}

fetch_traces:
{"action_type": "fetch_traces", "parameters": {"request_id": "<id>"}}

query_dependencies:
{"action_type": "query_dependencies", "parameters": {"service": "<name>"}}

submit_diagnosis:
{"action_type": "submit_diagnosis", "parameters": {"root_cause_service": "<name>", "cause_type": "<description>"}}

Rules:
- Output ONLY valid JSON. No explanation, no markdown, no extra text.
- All parameters are required. Missing parameters incur a penalty.
- metric_name is required for query_metrics.
- Do NOT repeat the same action with the same parameters.
- Do NOT explore the same service multiple times unless new evidence is required.
- Avoid unnecessary actions — each step must provide new information.
- Focus only on the most likely failing service.
- Call submit_diagnosis as soon as strong evidence is found — do NOT delay.

Strategy:
1. Start from alerts and identify the most likely failing service.
2. Use query_dependencies to trace the failure chain BEFORE diagnosing.
3. Always follow dependency chain to the downstream service before concluding.
4. Use grep_logs on the suspected root service to confirm the issue.
5. Use query_metrics only if logs are unclear.
6. Use fetch_traces only if dependency relationship is unclear.
7. Only call submit_diagnosis AFTER confirming root cause from logs or metrics.
8. Avoid repeated or unnecessary actions, but DO NOT skip investigation steps.

Respond ONLY with JSON. No explanation. Keep output minimal.

Examples:
{"action_type": "query_dependencies", "parameters": {"service": "api-gateway"}}

{"action_type": "grep_logs", "parameters": {"service": "user-service", "keyword": "timeout"}}
"""


class ParseError(ValueError):
    pass

def validate_env() -> None:
    if API_BASE_URL in ("", "base_url"):
        raise ValueError("Invalid API_BASE_URL")
    if MODEL_NAME in ("", "model_name"):
        raise ValueError("Invalid MODEL_NAME")
    if not HF_TOKEN or HF_TOKEN == "api_key":
        raise ValueError("Invalid HF_TOKEN")


def _format_action_str(action: ActionModel) -> str:
    parts = ", ".join(f"{k}={v}" for k, v in action.parameters.items())
    return f"{action.action_type}({parts})"


def _build_prompt(obs: dict, step: int) -> str:
    history_lines = ""
    for h in obs.get("history", [])[-3:]:
        history_lines += (
            f"\n  {h.get('action')}({h.get('parameters', {})}) "
            f"→ reward={h.get('reward', 0):+.2f}"
        )

    return f"""STEP {step} of {obs.get('max_steps', 25)}

TASK:
{obs.get('task_description', '')}

ALERTS:
{json.dumps(obs.get('alerts', []), indent=2)}

LAST TOOL RESULT:
{json.dumps(obs.get('tool_result'), indent=2)}

HISTORY:{history_lines if history_lines else ' none'}

Respond with JSON only."""


def _call_llm(messages: list[dict]) -> str:
    validate_env()
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed — run: pip install openai")

    client   = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=30.0)
    last_err = None

    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=120,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(1.5)

    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def _parse_action(raw: str) -> ActionModel:
    """
    Parse the LLM response into an ActionModel.
    Raises ParseError if the response is invalid.
    """
    valid_actions = {
        "grep_logs",
        "query_metrics",
        "fetch_traces",
        "query_dependencies",
        "submit_diagnosis",
    }

    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "")

    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        raise ParseError(
            "Response contains no JSON object. "
            "Reply with ONLY a JSON object, no explanation."
        )

    json_str = raw[start : end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ParseError(
            f"Invalid JSON: {exc}.  "
            "Fix the JSON syntax and reply with the corrected object only."
        ) from exc

    action_type = data.get("action_type")
    if not action_type:
        raise ParseError(
            'Missing "action_type" field.  '
            'Example: {"action_type": "query_dependencies", "parameters": {"service": "api-gateway"}}'
        )

    if action_type not in valid_actions:
        raise ParseError(
            f'Unknown action_type "{action_type}".  '
            f"Valid values: {sorted(valid_actions)}"
        )

    params = data.get("parameters", {})

    if action_type == "submit_diagnosis":
        if not params.get("root_cause_service") or not params.get("cause_type"):
            raise ParseError(
                "submit_diagnosis requires both 'root_cause_service' and 'cause_type' parameters."
            )

    if action_type == "query_metrics" and not params.get("metric_name"):
        raise ParseError(
            "query_metrics requires 'service' and 'metric_name' parameters."
        )

    return ActionModel(action_type=action_type, parameters=params)


def main() -> None:
    task = get_task(TASK_ID)
    print(f"[START] task={TASK_ID} env={ENV_NAME} model={MODEL_NAME}")

    env    = IncidentRCAEnv(task_id=TASK_ID, seed=SEED)
    obs    = env.reset()
    obs_dict = obs.model_dump()

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards:  list[float] = []
    actions_taken: list[tuple] = []

    step  = 0
    info  = None
    done  = False
    result = None

    try:
        for step in range(1, task["max_steps"] + 1):
            if obs_dict.get("done"):
                break

            messages.append({"role": "user", "content": _build_prompt(obs_dict, step)})

            action_str = "unknown()"
            reward_val = 0.0
            error_str  = "null"

           
            llm_failed   = False
            raw_response = None
            try:
                raw_response = _call_llm(messages)
                messages.append({"role": "assistant", "content": raw_response})
            except Exception as e:
                llm_failed = True
                error_str  = f"llm_error: {str(e)[:80]}"
                raw_response = None

            if llm_failed:
                print(f"[ERROR] LLM failed repeatedly: {error_str}")
                break

            try:
                action = _parse_action(raw_response)
            except ParseError as pe:
                hint = str(pe)[:200]
                messages.append({
                    "role": "system",
                    "content": (
                        f"Your previous response could not be parsed: {hint}  "
                        "Reply with a single valid JSON object and nothing else."
                    ),
                })
                error_str = f"parse_error (hint injected)"
                rewards.append(0.0)
                print(
                    f"[STEP] step={step} action=parse_failed "
                    f"reward=0.00 done=false error={error_str}"
                )
                continue

            try:
                action_str = _format_action_str(action)
                obs, reward, done, info = env.step(action)
                obs_dict   = obs.model_dump()
                reward_val = getattr(reward, "total", 0.0)
                actions_taken.append((action.model_dump(), reward.model_dump()))
            except Exception as e:
                error_str = str(e).replace("\n", " ")[:100]
                done = False

            rewards.append(reward_val)
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward_val:.2f} done={'true' if done else 'false'} "
                f"error={error_str}"
            )

            if done:
                break

    finally:
        score = 0.0
        if info is not None:
            try:
                episode = {
                    "task_id":     TASK_ID,
                    "scenario":    {"root_cause": info.model_dump().get("ground_truth_root_cause")},
                    "actions_taken": actions_taken,
                    "final_state": env.state(),
                    "info":        info.model_dump(),
                }
                result = IncidentRCAGrader().grade(episode)
                score  = result.score
            except Exception as e:
                score = 0.0

        success     = score >= 0.60 if (info is not None and result is not None) else False
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step} rewards={rewards_str}"
        )


if __name__ == "__main__":
    main()
