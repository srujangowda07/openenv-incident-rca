"""
validate.py — simulates 'openenv validate' checks.
Run before submitting: python validate.py

Checks:
  1.  openenv.yaml exists
  2.  Dockerfile exists
  3.  README.md exists
  4.  requirements.txt exists
  5.  openenv.yaml has required fields and valid structure
  6.  3 difficulty levels present in task definitions
  7.  reset() returns ObservationModel with required fields
  8.  step() returns (obs, reward, done, info) tuple
  9.  Reward is float in valid range [-1.0, 1.0]
  10. state() returns dict with required keys
  11. All 5 tasks loadable by environment
  12. Deterministic with same seed
  13. Grader returns score in [0.0, 1.0]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append(True)
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        results.append(False)


print("\n=== openenv validate ===\n")

# ── File presence ──────────────────────────────────────────────

check("openenv.yaml exists",   lambda: open("openenv.yaml").close())
check("Dockerfile exists",     lambda: open("Dockerfile").close())
check("README.md exists",      lambda: open("README.md").close())
check("requirements.txt exists", lambda: open("requirements.txt").close())


# ── openenv.yaml structure ────────────────────────────────────

def check_yaml():
    import yaml
    with open("openenv.yaml") as f:
        cfg = yaml.safe_load(f)
    required_fields = ["name", "version", "tasks", "observation_space",
                       "action_space", "reward_space", "grader"]
    for field in required_fields:
        assert field in cfg, f"Missing top-level field: '{field}'"
    assert len(cfg["tasks"]) >= 3, "Need at least 3 tasks"
    # Verify action_space enum contains only the 5 valid actions
    valid_actions = {
        "grep_logs", "query_metrics", "fetch_traces",
        "query_dependencies", "submit_diagnosis",
    }
    declared = set(cfg["action_space"]["fields"]["action_type"]["enum"])
    assert declared == valid_actions, (
        f"action_space mismatch. Got: {declared}, expected: {valid_actions}"
    )

check("openenv.yaml structure", check_yaml)


# ── Task definitions ──────────────────────────────────────────

def check_tasks():
    from tasks.task_definitions import TASKS
    assert "easy_001" in TASKS, "easy_001 not in TASKS"
    assert "medium_001" in TASKS, "medium_001 not in TASKS"
    assert "hard_001" in TASKS, "hard_001 not in TASKS"
    diffs = {t["difficulty"] for t in TASKS.values()}
    assert "easy" in diffs, "No easy tasks"
    assert "medium" in diffs, "No medium tasks"
    assert "hard" in diffs, "No hard tasks"

check("3 difficulty levels", check_tasks)


# ── Environment API ───────────────────────────────────────────

try:
    from environment.env import IncidentRCAEnv, ActionModel

    def check_reset():
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        obs = env.reset()
        assert obs.step == 0, f"Expected step=0, got {obs.step}"
        assert len(obs.alerts) > 0, "No alerts in observation"
        assert len(obs.available_actions) > 0, "No available actions"
        assert obs.task_id == "easy_001", "task_id mismatch"
        assert obs.done is False, "Episode should not be done after reset"

    check("reset() returns ObservationModel", check_reset)

    def check_step():
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        # Use a fully valid action with all required parameters
        action = ActionModel(
            action_type="grep_logs",
            parameters={"service": "api-gateway", "keyword": "error"},
        )
        obs, reward, done, info = env.step(action)
        assert isinstance(reward.total, float), "reward.total must be float"
        assert isinstance(done, bool), "done must be bool"
        assert hasattr(info, "steps_taken"), "info missing steps_taken"
        assert obs.step == 1, f"Expected step=1 after one action, got {obs.step}"

    check("step() returns (obs, reward, done, info)", check_step)

    def check_reward_range():
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        # query_metrics requires BOTH service AND metric_name
        action = ActionModel(
            action_type="query_metrics",
            parameters={"service": "postgres-primary", "metric_name": "active_connections"},
        )
        for _ in range(5):
            _, r, done, _ = env.step(action)
            assert -1.0 <= r.total <= 1.0, f"Reward out of range: {r.total}"
            if done:
                break

    check("reward in [-1.0, 1.0]", check_reward_range)

    def check_state():
        env = IncidentRCAEnv(task_id="medium_001", seed=42)
        env.reset()
        s = env.state()
        assert isinstance(s, dict), "state() must return dict"
        assert "step" in s, "state missing 'step'"
        assert "done" in s, "state missing 'done'"

    check("state() returns dict", check_state)

    def check_all_tasks():
        for tid in ["easy_001", "easy_002", "easy_003", "medium_001", "hard_001"]:
            env = IncidentRCAEnv(task_id=tid, seed=42)
            obs = env.reset()
            assert obs.task_id == tid, f"task_id mismatch for {tid}"

    check("all 5 tasks loadable", check_all_tasks)

    def check_determinism():
        env1 = IncidentRCAEnv(task_id="easy_001", seed=42)
        env2 = IncidentRCAEnv(task_id="easy_001", seed=42)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.task_description == obs2.task_description, (
            "Same seed must produce same task_description"
        )
        assert len(obs1.alerts) == len(obs2.alerts), (
            "Same seed must produce same number of alerts"
        )

    check("deterministic with same seed", check_determinism)

except ImportError as e:
    print(f"  {FAIL}  Could not import environment: {e} (run: pip install pydantic)")


# ── Grader ────────────────────────────────────────────────────

def check_grader_range():
    from environment.scenario_generator import ScenarioGenerator
    from graders.grader import IncidentRCAGrader

    gen = ScenarioGenerator(seed=42)
    scenario = gen.generate("easy_001")
    grader = IncidentRCAGrader()

    # Construct a valid episode matching the current grader.grade() contract:
    #   episode["final_state"] must have "diagnosed_service" and "diagnosed_cause"
    #   episode["info"] must have "invalid_actions"
    root_cause = scenario["root_cause"]
    episode = {
        "task_id": "easy_001",
        "scenario": scenario,
        "actions_taken": [
            (
                {"action_type": "grep_logs",
                 "parameters": {"service": root_cause["service"], "keyword": "error"}},
                {},
            ),
            (
                {"action_type": "submit_diagnosis",
                 "parameters": {
                     "root_cause_service": root_cause["service"],
                     "cause_type": root_cause["cause_type"],
                 }},
                {},
            ),
        ],
        "final_state": {
            "step": 2,
            "done": True,
            "diagnosed_service": root_cause["service"],
            "diagnosed_cause": root_cause["cause_type"],
            "action_history": [
                {
                    "action": "grep_logs",
                    "parameters": {"service": root_cause["service"], "keyword": "error"},
                    "result": {"logs": []},
                    "reward": 0.05,
                }
            ],
        },
        "info": {
            "tools_used": ["grep_logs", "submit_diagnosis"],
            "invalid_actions": 0,
        },
        "max_steps": 15,
    }

    result = grader.grade(episode)
    assert 0.0 <= result.score <= 1.0, f"Score out of range: {result.score}"
    assert result.score >= 0.60, (
        f"Perfect episode should pass threshold (0.60). Got: {result.score}"
    )
    assert result.passed, "Perfect episode should be marked as passed"

check("grader score in [0.0, 1.0]", check_grader_range)


# ── Summary ───────────────────────────────────────────────────

passed = sum(results)
total = len(results)
print(f"\n{'='*32}")
print(f"  {passed}/{total} checks passed")
if passed == total:
    print("  Environment is valid. Ready to submit.")
else:
    print("  Fix failures before submitting.")
print()
