import argparse
import json
import os
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.env import IncidentRCAEnv, ActionModel
from graders.grader import IncidentRCAGrader
from tasks.task_definitions import TASKS, get_task


SYSTEM_PROMPT = """You are an expert Site Reliability Engineer performing incident response.
You are given an incident environment with logs, metrics, traces, and service dependency maps.

Your goal is to diagnose the root cause service and failure type, then submit your diagnosis.

Available actions — respond ONLY with valid JSON matching one of these schemas:

grep_logs:
  {"action_type": "grep_logs", "parameters": {"service": "<service-name>", "keyword": "<search-term>"}}

query_metrics:
  {"action_type": "query_metrics", "parameters": {"service": "<service-name>", "metric_name": "<metric-name>"}}

fetch_traces:
  {"action_type": "fetch_traces", "parameters": {"request_id": "<trace-id>"}}

query_dependencies:
  {"action_type": "query_dependencies", "parameters": {"service": "<service-name>"}}

submit_diagnosis:
  {"action_type": "submit_diagnosis", "parameters": {"root_cause_service": "<service-name>", "cause_type": "<failure-description>"}}

IMPORTANT RULES:
- All parameters shown above are REQUIRED. Omitting any parameter causes an invalid action penalty.
- metric_name is REQUIRED for query_metrics (examples: cpu_usage, memory_usage_mb, error_rate, latency_p99)
- submit_diagnosis ENDS the episode — only call it when you are confident.
- Do NOT repeat the exact same tool call with the same parameters (penalty applies).

INVESTIGATION STRATEGY:
1. Read alerts to identify which services are affected.
2. Use query_dependencies to understand the service dependency chain.
3. Use grep_logs on suspicious services to find error messages.
4. Use query_metrics to confirm timing and severity of degradation.
5. Use fetch_traces to see the full request path (trace IDs appear in alerts/logs).
6. Look for the DEEPEST service in the chain that is failing — that is the root cause.
7. Call submit_diagnosis when you are confident (>80%) of the root cause.

Respond with ONLY the JSON action object. No explanation. No markdown. Just the JSON."""


def build_user_prompt(obs: dict, step: int) -> str:
    history_lines = ""
    for h in obs.get("history", []):
        history_lines += f"\n  Step {h.get('action')}: reward={h.get('reward', 0):+.3f}"

    return f"""STEP {step} of {obs.get('max_steps', 25)} — INCIDENT ENVIRONMENT STATE

=== TASK ===
{obs.get('task_description', '')}

=== ACTIVE ALERTS ===
{json.dumps(obs.get('alerts', []), indent=2)}

=== LAST TOOL RESULT ===
{json.dumps(obs.get('tool_result'), indent=2)}

=== ACTION HISTORY (last 5) ==={history_lines if history_lines else " (none yet)"}

=== AVAILABLE ACTIONS ===
{', '.join(obs.get('available_actions', []))}

What is your next action? Respond with JSON only."""


def call_llm(messages: list[dict], model: str = "gpt-4o") -> str:
    """Call OpenAI API. Returns raw response text."""
    try:
        import openai
    except ImportError:
        raise ImportError("Run: pip install openai")

    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set HF_TOKEN or OPENAI_API_KEY")

    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def parse_action(raw: str) -> ActionModel:
    try:
        data = json.loads(raw)
        return ActionModel(
            action_type=data.get("action_type", "submit_diagnosis"),
            parameters=data.get("parameters", {}),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return ActionModel(
            action_type="submit_diagnosis",
            parameters={"root_cause_service": "unknown", "cause_type": "parse_error"},
        )


def run_episode(task_id: str, model: str = "gpt-4o", seed: int = 42,
                verbose: bool = True) -> dict:
    task = get_task(task_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task['name']}  ({task_id})")
        print(f"  Model: {model}  |  Seed: {seed}")
        print(f"{'='*60}")

    env = IncidentRCAEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    obs_dict = obs.model_dump()

    actions_taken = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    info = None

    for step in range(1, task["max_steps"] + 1):
        if obs_dict.get("done"):
            break

        user_msg = build_user_prompt(obs_dict, step)
        messages.append({"role": "user", "content": user_msg})

        if verbose:
            print(f"\n  Step {step}: ", end="", flush=True)

        try:
            raw_response = call_llm(messages, model=model)
            messages.append({"role": "assistant", "content": raw_response})
        except Exception as e:
            print(f"LLM error: {e}")
            break

        action = parse_action(raw_response)

        if verbose:
            print(f"{action.action_type}({json.dumps(action.parameters)})")

        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()
        actions_taken.append((action.model_dump(), reward.model_dump()))

        if verbose:
            print(f"    Reward: {reward.total:+.3f} — {reward.reason}")

        if done:
            break

        time.sleep(0.3)

    if info is None:
        _, _, _, info = env.step(ActionModel(
            action_type="submit_diagnosis",
            parameters={"root_cause_service": "unknown", "cause_type": "no_steps"},
        ))

    final_state = env.state()

    if verbose:
        rca = env._scenario["root_cause"]
        print(f"\n  Ground truth: {rca['service']} — {rca['cause_type']}")
        print(f"  Agent diagnosed: {final_state.get('diagnosed_service', 'None')}")

    return {
        "task_id": task_id,
        "model": model,
        "seed": seed,
        "scenario": env._scenario,
        "actions_taken": actions_taken,
        "final_state": final_state,
        "info": info.model_dump(),
        "max_steps": task["max_steps"],
    }


def grade_episode(episode: dict, verbose: bool = True) -> dict:
    grader = IncidentRCAGrader()
    result = grader.grade(episode)

    if verbose:
        print(f"\n  GRADE: {result.score:.3f} / 1.000  ({'PASS' if result.passed else 'FAIL'})")
        print("  Breakdown:")
        for dim, score in result.breakdown.items():
            if isinstance(score, float) and score != 0.0:
                bar_len = max(0, int(abs(score) * 20))
                bar = "█" * bar_len + "░" * (20 - bar_len)
                sign = "-" if score < 0 else " "
                print(f"    {dim:<25} {sign}{bar}  {score:+.3f}")
            else:
                print(f"    {dim:<25} {'░' * 20}  {score:+.3f}")
        print(f"  Feedback: {result.feedback}")

    return {
        "task_id": episode["task_id"],
        "score": result.score,
        "passed": result.passed,
        "breakdown": result.breakdown,
        "feedback": result.feedback,
    }


def run_all_tasks(model: str = "gpt-4o", seed: int = 42) -> dict:
    results = {}
    for task_id in TASKS:
        episode = run_episode(task_id, model=model, seed=seed)
        grade = grade_episode(episode)
        results[task_id] = grade

    print("\n" + "=" * 60)
    print("  BASELINE REPORT")
    print("=" * 60)
    total_score = sum(r["score"] for r in results.values())
    avg = total_score / len(results)
    print(f"  Model: {model}")
    print(f"  Average Score: {avg:.3f}")
    print()
    for task_id, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {task_id:<15} {r['score']:.3f}  [{status}]")
    print("=" * 60)

    report_path = Path("baseline_scores.json")
    with open(report_path, "w") as f:
        json.dump({"model": model, "avg_score": avg, "tasks": results}, f, indent=2)
    print(f"\n  Report saved to {report_path}")

    return results


def _run_dry(task_id: str):
    print(f"\n[DRY RUN] Task: {task_id}")
    env = IncidentRCAEnv(task_id=task_id, seed=42)
    obs = env.reset()

    smart_actions = [
        ActionModel(
            action_type="query_dependencies",
            parameters={"service": "api-gateway"},
        ),
        ActionModel(
            action_type="grep_logs",
            parameters={"service": "postgres-primary", "keyword": "connection"},
        ),
        ActionModel(
            action_type="query_metrics",
            parameters={"service": "postgres-primary", "metric_name": "active_connections"},
        ),
        ActionModel(
            action_type="fetch_traces",
            parameters={"request_id": "req-7f3a"},
        ),
        ActionModel(
            action_type="submit_diagnosis",
            parameters={
                "root_cause_service": "postgres-primary",
                "cause_type": "connection pool exhausted",
            },
        ),
    ]

    actions_taken = []
    info = None

    for action in smart_actions:
        if obs.done:
            break
        obs, reward, done, info = env.step(action)
        actions_taken.append((action.model_dump(), reward.model_dump()))
        print(f"  {action.action_type}({json.dumps(action.parameters)})")
        print(f"    reward={reward.total:+.3f} — {reward.reason}")

    if info is None:
        print("  [WARN] No steps taken — episode may have ended immediately.")
        return

    episode = {
        "task_id": task_id,
        "model": "dry-run-scripted",
        "seed": 42,
        "scenario": env._scenario,
        "actions_taken": actions_taken,
        "final_state": env.state(),
        "info": info.model_dump(),
        "max_steps": get_task(task_id)["max_steps"],
    }
    grade_episode(episode)


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline LLM agent on IncidentRCAEnv"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Task ID (e.g. easy_001, medium_001, hard_001)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tasks and produce full baseline report",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without calling LLM — uses scripted valid actions for smoke testing",
    )

    args = parser.parse_args()

    if args.dry_run:
        task = args.task or "easy_001"
        print(f"[DRY RUN] Using scripted actions — no LLM calls (task: {task})")
        _run_dry(task)
        return

    if args.all:
        run_all_tasks(model=args.model, seed=args.seed)
    elif args.task:
        episode = run_episode(args.task, model=args.model, seed=args.seed)
        grade_episode(episode)
    else:
        print("Specify --task <id> or --all  (or --dry-run for a smoke test)")
        parser.print_help()


if __name__ == "__main__":
    main()
