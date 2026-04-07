"""
Test suite for IncidentRCAEnv.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from environment.env import IncidentRCAEnv, ActionModel
from graders.grader import IncidentRCAGrader
from tasks.task_definitions import TASKS, get_task


# Smoke Tests 

class TestEnvAPI:
    """Test OpenEnv API compliance: reset(), step(), state()."""

    def test_reset_returns_observation(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        obs = env.reset()
        assert obs.step == 0
        assert obs.task_id == "easy_001"
        assert len(obs.alerts) > 0
        assert len(obs.available_actions) > 0
        assert obs.done is False

    def test_step_returns_tuple(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="grep_logs",
            parameters={"keyword": "error", "service": "postgres-primary"},
        )
        obs, reward, done, info = env.step(action)
        assert 0 <= obs.step <= 15
        assert isinstance(reward.total, float)
        assert isinstance(done, bool)

    def test_state_returns_dict(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        state = env.state()
        assert isinstance(state, dict)
        assert "step" in state
        assert "done" in state

    def test_step_before_reset_raises(self):
        env = IncidentRCAEnv(task_id="easy_001")
        action = ActionModel(
            action_type="grep_logs",
            parameters={"service": "api-gateway", "keyword": "error"},
        )
        with pytest.raises(AssertionError):
            env.step(action)

    def test_max_steps_terminates_episode(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        # query_metrics without metric_name → invalid each time, but should not crash
        action = ActionModel(
            action_type="grep_logs",
            parameters={"service": "unknown-service", "keyword": "x"},
        )
        done = False
        for _ in range(20):  # easy_001 max_steps = 15
            if done:
                break
            _, _, done, _ = env.step(action)
        assert done


# Action Tests 

class TestActions:

    def test_grep_logs_returns_logs(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="grep_logs",
            parameters={"keyword": "error", "service": "postgres-primary"},
        )
        obs, reward, _, _ = env.step(action)
        assert obs.tool_result is not None
        assert "logs" in obs.tool_result
        # Querying a cascade service gives positive reward
        assert reward.total >= 0.0

    def test_correct_diagnosis_gives_high_reward(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        
        # Execute 2 steps to avoid early submission penalty
        env.step(ActionModel(
            action_type="grep_logs",
            parameters={"keyword": "error", "service": "postgres-primary"}
        ))
        env.step(ActionModel(
            action_type="query_metrics",
            parameters={"service": "postgres-primary", "metric_name": "active_connections"}
        ))
        
        action = ActionModel(
            action_type="submit_diagnosis",
            parameters={
                "root_cause_service": "postgres-primary",
                "cause_type": "connection pool exhausted",
            },
        )
        _, reward, _, _ = env.step(action)
        # Perfect diagnosis = +1.00 minus step penalty (-0.01) = +0.99
        assert reward.total > 0.4

    def test_wrong_diagnosis_gives_penalty(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="submit_diagnosis",
            parameters={
                "root_cause_service": "api-gateway",       # victim, not root cause
                "cause_type": "configuration error",
            },
        )
        _, reward, _, _ = env.step(action)
        # Wrong diagnosis = -0.50 minus step penalty = -0.51
        assert reward.total <= 0.0

    def test_submission_ends_episode(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="submit_diagnosis",
            parameters={"root_cause_service": "api-gateway", "cause_type": "unknown"},
        )
        _, _, done, _ = env.step(action)
        assert done is True

    def test_query_dependencies_returns_graph(self):
        env = IncidentRCAEnv(task_id="medium_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="query_dependencies",
            parameters={"service": "api-gateway"},
        )
        obs, _, _, _ = env.step(action)
        assert obs.tool_result is not None
        assert "upstream" in obs.tool_result
        assert "downstream" in obs.tool_result

    def test_query_metrics_requires_metric_name(self):
        """query_metrics without metric_name must return an invalid-action penalty."""
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="query_metrics",
            parameters={"service": "postgres-primary"},   # metric_name missing
        )
        _, reward, _, _ = env.step(action)
        assert reward.total < 0.0

    def test_query_metrics_with_metric_name_succeeds(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="query_metrics",
            parameters={
                "service": "postgres-primary",
                "metric_name": "active_connections",
            },
        )
        obs, reward, _, _ = env.step(action)
        assert obs.tool_result is not None
        assert "metrics" in obs.tool_result
        assert reward.total >= 0.0

    def test_repeated_action_gives_penalty(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="grep_logs",
            parameters={"service": "postgres-primary", "keyword": "error"},
        )
        _, reward1, _, _ = env.step(action)
        _, reward2, _, _ = env.step(action)  # exact duplicate
        assert reward1.total > reward2.total

    def test_invalid_service_gives_penalty(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="grep_logs",
            parameters={"service": "nonexistent-service", "keyword": "error"},
        )
        _, reward, _, _ = env.step(action)
        assert reward.total < 0.0

    def test_fetch_traces_with_valid_id(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="fetch_traces",
            parameters={"request_id": "req-7f3a"},
        )
        obs, reward, _, _ = env.step(action)
        assert obs.tool_result is not None
        assert "trace" in obs.tool_result
        assert reward.total > 0.0

    def test_fetch_traces_with_invalid_id_gives_penalty(self):
        env = IncidentRCAEnv(task_id="easy_001", seed=42)
        env.reset()
        action = ActionModel(
            action_type="fetch_traces",
            parameters={"request_id": "req-nonexistent"},
        )
        _, reward, _, _ = env.step(action)
        assert reward.total < 0.0


# Grader Tests

class TestGrader:

    def _make_episode(self, task_id: str, diagnosed_svc: str, diagnosed_cause: str,
                      invalid_actions: int = 0, queried_root_cause: bool = True) -> dict:
        """Build a minimal valid episode dict for the grader."""
        env = IncidentRCAEnv(task_id=task_id, seed=42)
        env.reset()
        root_cause_svc = env._scenario["root_cause"]["service"]

        action_history = []
        if queried_root_cause:
            action_history.append({
                "action": "grep_logs",
                "parameters": {"service": root_cause_svc, "keyword": "error"},
                "result": {"logs": []},
                "reward": 0.05,
            })

        return {
            "task_id": task_id,
            "scenario": env._scenario,
            "actions_taken": [
                (
                    {"action_type": "grep_logs",
                     "parameters": {"service": root_cause_svc, "keyword": "error"}},
                    {},
                ),
                (
                    {"action_type": "submit_diagnosis",
                     "parameters": {
                         "root_cause_service": diagnosed_svc,
                         "cause_type": diagnosed_cause,
                     }},
                    {},
                ),
            ],
            "final_state": {
                "step": 2,
                "done": True,
                "diagnosed_service": diagnosed_svc,
                "diagnosed_cause": diagnosed_cause,
                "action_history": action_history,
            },
            "info": {
                "tools_used": ["grep_logs", "submit_diagnosis"],
                "invalid_actions": invalid_actions,
            },
            "max_steps": 15,
        }

    def test_perfect_score_correct_answer(self):
        grader = IncidentRCAGrader()
        episode = self._make_episode(
            "easy_001",
            diagnosed_svc="postgres-primary",
            diagnosed_cause="connection pool exhausted",
        )
        result = grader.grade(episode)
        assert result.score >= 0.60
        assert result.passed
        # Perfect: 0.50 + 0.30 + 0.20 = 1.00
        assert result.score == 1.00

    def test_wrong_service_gives_low_score(self):
        grader = IncidentRCAGrader()
        episode = self._make_episode(
            "easy_001",
            diagnosed_svc="api-gateway",          
            diagnosed_cause="unknown",
            queried_root_cause=False,
        )
        result = grader.grade(episode)
        assert result.score < 0.50
        assert not result.passed

    def test_correct_service_wrong_cause(self):
        grader = IncidentRCAGrader()
        episode = self._make_episode(
            "easy_001",
            diagnosed_svc="postgres-primary",
            diagnosed_cause="wrong cause type",     
        )
        result = grader.grade(episode)
        assert 0.60 <= result.score < 1.00

    def test_invalid_actions_penalizes_score(self):
        grader = IncidentRCAGrader()
        ep1 = self._make_episode("easy_001", "postgres-primary",
                                 "connection pool exhausted", invalid_actions=0)
        ep2 = self._make_episode("easy_001", "postgres-primary",
                                 "connection pool exhausted", invalid_actions=2)
        r1 = grader.grade(ep1)
        r2 = grader.grade(ep2)
        assert r1.score > r2.score

    def test_score_range_clamped(self):
        grader = IncidentRCAGrader()
        episode = self._make_episode("easy_001", "postgres-primary",
                                     "connection pool exhausted", invalid_actions=10)
        result = grader.grade(episode)
        assert 0.0 <= result.score <= 1.0

    def test_grade_result_has_feedback(self):
        grader = IncidentRCAGrader()
        episode = self._make_episode("easy_001", "wrong-service", "wrong-fix",
                                     queried_root_cause=False)
        result = grader.grade(episode)
        assert isinstance(result.feedback, str)
        assert len(result.feedback) > 0

    def test_no_evidence_loses_evidence_score(self):
        grader = IncidentRCAGrader()
        ep_with = self._make_episode("easy_001", "postgres-primary",
                                     "connection pool exhausted", queried_root_cause=True)
        ep_without = self._make_episode("easy_001", "postgres-primary",
                                        "connection pool exhausted", queried_root_cause=False)
        r_with = grader.grade(ep_with)
        r_without = grader.grade(ep_without)
        assert r_with.score > r_without.score
        assert r_without.breakdown["tool_evidence"] == 0.0


# Task Definition Tests

class TestTasks:

    def test_all_tasks_exist(self):
        for task_id in ["easy_001", "easy_002", "easy_003", "medium_001", "hard_001"]:
            task = get_task(task_id)
            assert task["id"] == task_id

    def test_task_has_required_fields(self):
        for task_id, task in TASKS.items():
            assert "difficulty" in task, f"{task_id} missing 'difficulty'"
            assert "max_steps" in task, f"{task_id} missing 'max_steps'"
            assert "description" in task, f"{task_id} missing 'description'"
            assert task["difficulty"] in ("easy", "medium", "hard"), (
                f"{task_id} has invalid difficulty: {task['difficulty']}"
            )

    def test_invalid_task_raises(self):
        with pytest.raises(KeyError):
            get_task("fake_999")

    def test_env_accepts_all_task_ids(self):
        for task_id in TASKS:
            env = IncidentRCAEnv(task_id=task_id, seed=42)
            obs = env.reset()
            assert obs.task_id == task_id


# Determinism Tests

class TestDeterminism:

    def test_same_seed_same_scenario(self):
        env1 = IncidentRCAEnv(task_id="easy_001", seed=42)
        env2 = IncidentRCAEnv(task_id="easy_001", seed=42)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.task_description == obs2.task_description
        assert len(obs1.alerts) == len(obs2.alerts)

    def test_same_seed_same_rewards(self):
        """Same seed + same actions must produce identical rewards."""
        action = ActionModel(
            action_type="query_metrics",
            parameters={"service": "postgres-primary", "metric_name": "active_connections"},
        )
        env1 = IncidentRCAEnv(task_id="easy_001", seed=42)
        env2 = IncidentRCAEnv(task_id="easy_001", seed=42)
        env1.reset()
        env2.reset()
        _, r1, _, _ = env1.step(action)
        _, r2, _, _ = env2.step(action)
        assert r1.total == r2.total

    def test_different_seeds_no_crash(self):
        env1 = IncidentRCAEnv(task_id="hard_001", seed=1)
        env2 = IncidentRCAEnv(task_id="hard_001", seed=999)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.task_id == obs2.task_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
