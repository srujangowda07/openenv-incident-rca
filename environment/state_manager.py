from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
import json


@dataclass
class EnvState:
    step: int = 0
    done: bool = False
    diagnosed_service: str | None = None
    diagnosed_cause: str | None = None
    tools_used: list[str] = field(default_factory=list)
    unique_tool_calls: set[str] = field(default_factory=set)
    invalid_actions: int = 0
    cumulative_reward: float = 0.0
    tool_result: dict | None = None
    action_history: list[dict] = field(default_factory=list)
    visited_services: set[str] = field(default_factory=set)
    dependency_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "done": self.done,
            "diagnosed_service": self.diagnosed_service,
            "diagnosed_cause": self.diagnosed_cause,
            "tools_used": list(set(self.tools_used)),
            "invalid_actions": self.invalid_actions,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "tool_result": self.tool_result,
            "action_history": self.action_history,
            "visited_services": list(self.visited_services),
            "dependency_path": self.dependency_path,
        }


class StateManager:
    def __init__(self):
        self._state = EnvState()

    def reset(self) -> EnvState:
        self._state = EnvState()
        return self._state

    @property
    def state(self) -> EnvState:
        return self._state

    def increment_step(self):
        self._state.step += 1

    def record_tool(self, tool_name: str, params: dict) -> bool:
        self._state.tools_used.append(tool_name)
        svc = params.get("service", params.get("root_cause_service", params.get("request_id", "")))
        sig = f"{tool_name}:{svc}"
        is_duplicate = sig in self._state.unique_tool_calls
        if not is_duplicate:
            self._state.unique_tool_calls.add(sig)
        return is_duplicate

    def set_tool_result(self, result: dict | None):
        self._state.tool_result = result

    def record_action(self, action_dict: dict):
        self._state.action_history.append(deepcopy(action_dict))

    def add_reward(self, amount: float):
        self._state.cumulative_reward = round(self._state.cumulative_reward + amount, 4)

    def record_diagnosis(self, service: str, cause: str):
        self._state.diagnosed_service = service
        self._state.diagnosed_cause = cause
        self._state.done = True

    def record_invalid_action(self):
        self._state.invalid_actions += 1

    def set_done(self, done: bool):
        self._state.done = done

    def should_terminate(self, max_steps: int) -> bool:
        return self._state.step >= max_steps or self._state.diagnosed_service is not None

    def snapshot(self) -> dict:
        return self._state.to_dict()
