from __future__ import annotations
import random

from pydantic import BaseModel, Field

from .scenario_generator import ScenarioGenerator
from .state_manager import StateManager
from .reward_shaper import RewardShaper


class ObservationModel(BaseModel):
    step: int = 0
    max_steps: int = 25
    task_id: str = ""
    task_description: str = ""
    alerts: list[dict] = Field(default_factory=list)
    tool_result: dict | None = None
    history: list[dict] = Field(default_factory=list)
    available_actions: list[str] = Field(default_factory=list)
    done: bool = False


class ActionModel(BaseModel):
    action_type: str
    parameters: dict = Field(default_factory=dict)


class RewardModel(BaseModel):
    total: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)
    reason: str = ""


class InfoModel(BaseModel):
    ground_truth_root_cause: dict | None = None
    steps_taken: int = 0
    tools_used: list[str] = Field(default_factory=list)
    invalid_actions: int = 0
    cumulative_reward: float = 0.0


AVAILABLE_ACTIONS = [
    "grep_logs",
    "query_metrics",
    "fetch_traces",
    "query_dependencies",
    "submit_diagnosis",
]

MAX_STEPS = {"easy": 15, "medium": 25, "hard": 40}


class IncidentRCAEnv:
    def __init__(self, task_id: str = "easy_001", seed: int | None = None):
        self.task_id = task_id
        self.seed = seed if seed is not None else random.randint(0, 99_999)
        difficulty = task_id.split("_")[0]
        self.max_steps = MAX_STEPS.get(difficulty, 25)
        self._generator = ScenarioGenerator(seed=self.seed)
        self._sm = StateManager()
        self._scenario: dict = {}
        self._reward_shaper: RewardShaper | None = None
        self._ready = False

    def reset(self) -> ObservationModel:
        self._scenario = self._generator.generate(self.task_id)
        self._sm.reset()
        self._reward_shaper = RewardShaper(self._scenario)
        self._ready = True
        return self._build_obs()

    def step(self, action: ActionModel) -> tuple[ObservationModel, RewardModel, bool, InfoModel]:
        assert self._ready, "Call reset() before step()"
        assert not self._sm.state.done, "Episode done — call reset()"

        self._sm.increment_step()
        is_duplicate = self._sm.record_tool(action.action_type, action.parameters)
        self._sm.set_tool_result(None)

        total, breakdown, reason = self._dispatch(action, is_duplicate)
        total, breakdown = self._reward_shaper.applying_step_penalty(total, breakdown)
        self._sm.add_reward(total)

        self._sm.record_action({
            "action": action.action_type,
            "parameters": action.parameters,
            "result": self._sm.state.tool_result,
            "reward": round(total, 4),
        })

        done = self._sm.should_terminate(self.max_steps)
        self._sm.set_done(done)

        return (
            self._build_obs(),
            RewardModel(total=round(total, 4), breakdown=breakdown, reason=reason),
            done,
            self._build_info(done),
        )

    def state(self) -> dict:
        return self._sm.snapshot()

    def _dispatch(self, action: ActionModel, is_duplicate: bool) -> tuple[float, dict, str]:
        at, p, rs = action.action_type, action.parameters, self._reward_shaper

        if is_duplicate and at != "submit_diagnosis":
            return rs.reward_repeated_action(at)

        if at == "grep_logs":
            svc = p.get("service", "")
            kw = p.get("keyword", "")
            if not svc:
                self._sm.record_invalid_action()
                return rs.reward_invalid_action(at, "missing 'service'")
            if not any(s["name"] == svc for s in self._scenario.get("services", [])):
                self._sm.record_invalid_action()
                self._sm.set_tool_result({"error": f"service '{svc}' not found"})
                return rs.reward_invalid_action(at, "service not found")
            logs = [
                l for l in self._scenario.get("logs", [])
                if kw.lower() in l["message"].lower()
                and svc.lower() in l.get("service", "").lower()
            ]
            self._sm.set_tool_result({"logs": logs[:20]})
            self._sm.state.visited_services.add(svc)
            return rs.reward_grep_logs(svc)

        if at == "query_metrics":
            svc = p.get("service", "")
            metric = p.get("metric_name", "")
            if not svc or not metric:
                self._sm.record_invalid_action()
                return rs.reward_invalid_action(at, "missing 'service' or 'metric_name'")
            if not any(s["name"] == svc for s in self._scenario.get("services", [])):
                self._sm.record_invalid_action()
                self._sm.set_tool_result({"error": f"service '{svc}' not found"})
                return rs.reward_invalid_action(at, "service not found")
            metrics = [
                m for m in self._scenario.get("metrics", [])
                if svc.lower() in m["service"].lower()
                and metric.lower() in m["metric"].lower()
            ]
            self._sm.set_tool_result({"metrics": metrics})
            self._sm.state.visited_services.add(svc)
            return rs.reward_query_metrics(svc)

        if at == "fetch_traces":
            rid = p.get("request_id", "")
            if not rid:
                self._sm.record_invalid_action()
                return rs.reward_invalid_action(at, "missing 'request_id'")
            trace = self._scenario.get("traces", {}).get(rid)
            if not trace:
                self._sm.record_invalid_action()
                self._sm.set_tool_result({"error": f"no trace for '{rid}'"})
                return rs.reward_invalid_action(at, "trace id not found")
            self._sm.set_tool_result({"trace": trace})
            rca_svc = self._scenario["root_cause"]["service"].lower()
            trace_services = [span.get("service", "") for span in trace]
            implicates = any(rca_svc in span.get("service", "").lower() or rca_svc in span.get("error", "").lower() for span in trace)
            for ts in trace_services:
                self._sm.state.visited_services.add(ts)
            return rs.reward_fetch_traces(implicates, trace_services)

        if at == "query_dependencies":
            svc = p.get("service", "")
            if not svc:
                self._sm.record_invalid_action()
                return rs.reward_invalid_action(at, "missing 'service'")
            graph = self._scenario.get("dependency_graph", {})
            upstream = [k for k, v in graph.items() if svc in v]
            downstream = graph.get(svc, [])
            if not upstream and not downstream and svc not in graph:
                self._sm.record_invalid_action()
                self._sm.set_tool_result({"error": f"service '{svc}' not in graph"})
                return rs.reward_invalid_action(at, "service not found in graph")
            self._sm.set_tool_result({"service": svc, "upstream": upstream, "downstream": downstream})
            self._sm.state.visited_services.add(svc)
            if svc not in self._sm.state.dependency_path:
                self._sm.state.dependency_path.append(svc)
            return rs.reward_query_dependencies(svc)

        if at == "submit_diagnosis":
            svc = p.get("root_cause_service", "")
            cause = p.get("cause_type", "")
            if not svc or not cause:
                self._sm.record_invalid_action()
                return rs.reward_invalid_action(at, "missing 'root_cause_service' or 'cause_type'")
            self._sm.record_diagnosis(svc, cause)
            self._sm.set_tool_result({"diagnosis_submitted": True})
            return rs.reward_diagnosis(svc, cause, self._sm.state.step)

        self._sm.record_invalid_action()
        return rs.reward_invalid_action(at, f"unknown action: {at}")

    def _build_obs(self) -> ObservationModel:
        s, sc = self._sm.state, self._scenario
        return ObservationModel(
            step=s.step,
            max_steps=self.max_steps,
            task_id=self.task_id,
            task_description=sc.get("description", ""),
            alerts=sc.get("alerts", []),
            tool_result=s.tool_result,
            history=s.action_history[-5:],
            available_actions=AVAILABLE_ACTIONS,
            done=s.done,
        )

    def _build_info(self, done: bool) -> InfoModel:
        s = self._sm.state
        return InfoModel(
            ground_truth_root_cause=self._scenario["root_cause"] if done else None,
            steps_taken=s.step,
            tools_used=list(set(s.tools_used)),
            invalid_actions=s.invalid_actions,
            cumulative_reward=s.cumulative_reward,
        )
