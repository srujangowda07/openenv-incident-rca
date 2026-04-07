from __future__ import annotations
from .canonical import normalize_cause_type, normalize_service



REWARD_TABLE: dict[str, float] = {
    "submit_diagnosis_perfect":   +1.00,   # exact service + cause match
    "submit_diagnosis_partial":   +0.50,   # correct service, wrong/unrecognised cause
    "submit_diagnosis_wrong":     -0.50,   # wrong service
    "submit_diagnosis_early":     -0.20,   # diagnosis before gathering any evidence
    "reaching_root":              +0.30,   # first time agent queries the root cause service
    "correct_dependency_step":    +0.10,   # queried a service in the cascade (not root)
    "useful_exploration":         +0.05,   # trace implicates root cause but service already seen
    "wrong_direction":            -0.05,   # queried a service outside the cascade
    "repeated_action":            -0.10,   # exact same tool call repeated
    "invalid_action":             -0.10,   # missing required param or unknown service
    "step_penalty":               -0.01,   # applied every step as efficiency incentive
}


class RewardShaper:
    def __init__(self, scenario: dict):
        self.rca_service = normalize_service(scenario["root_cause"]["service"])
        self.rca_cause = scenario["root_cause"]["cause_type"]
        self.cascade = [normalize_service(s) for s in scenario["root_cause"].get("cascade", [])]
        self.dependency_graph = scenario.get("dependency_graph", {})

        self.rewarded_services: set[str] = set()

    def _evaluate_service_progress(
        self, service: str, action: str
    ) -> tuple[float, dict, str]:
        service = normalize_service(service)

        if service in self.rewarded_services:
            return 0.0, {action: 0.0}, f"already rewarded for {service}"

        self.rewarded_services.add(service)

        if service == self.rca_service:
            v = REWARD_TABLE["reaching_root"]
            return v, {action: v}, f"reached root cause service: {service}"

        if service in self.cascade:
            v = REWARD_TABLE["correct_dependency_step"]
            return v, {action: v}, f"correct dependency step toward root: {service}"

        v = REWARD_TABLE["wrong_direction"]
        return v, {action: v}, f"wrong direction — service not in cascade: {service}"

    def reward_grep_logs(self, service: str) -> tuple[float, dict, str]:
        return self._evaluate_service_progress(service, "grep_logs")

    def reward_query_metrics(self, service: str) -> tuple[float, dict, str]:
        return self._evaluate_service_progress(service, "query_metrics")

    def reward_fetch_traces(
        self, implicates: bool, trace_services: list[str]
    ) -> tuple[float, dict, str]:
        
        if implicates:
            rca_norm = self.rca_service
            for svc in trace_services:
                if normalize_service(svc) == rca_norm and rca_norm not in self.rewarded_services:
                    self.rewarded_services.add(rca_norm)
                    v = REWARD_TABLE["reaching_root"]
                    return v, {"fetch_traces": v}, f"trace reached root cause service: {rca_norm}"

            # Root cause service implicated but already rewarded — still useful.
            v = REWARD_TABLE["useful_exploration"]
            return v, {"fetch_traces": v}, "trace implicates root cause (already visited)"

        v = REWARD_TABLE["wrong_direction"]
        return v, {"fetch_traces": v}, "trace does not implicate root cause"

    def reward_query_dependencies(self, service: str) -> tuple[float, dict, str]:
        return self._evaluate_service_progress(service, "query_dependencies")

    def reward_invalid_action(
        self, action_type: str, reason: str = ""
    ) -> tuple[float, dict, str]:
        val = REWARD_TABLE["invalid_action"]
        return val, {"invalid_action": val}, f"invalid action {action_type}: {reason}"

    def reward_repeated_action(self, action_type: str) -> tuple[float, dict, str]:
        val = REWARD_TABLE["repeated_action"]
        return val, {"repeated_action": val}, f"repeated action: {action_type}"

    def reward_diagnosis(
        self, guessed_service: str, guessed_cause: str, steps: int
    ) -> tuple[float, dict, str]:
        """
        Terminal reward for submit_diagnosis.
        Requires at least 2 steps of evidence gathering before diagnosing.
        """
        if steps < 2:
            val = REWARD_TABLE["submit_diagnosis_early"]
            return val, {"diagnosis": val}, "diagnosis submitted too early (step < 2)"

        guessed_service_norm = normalize_service(guessed_service)
        normalized_guess_cause = normalize_cause_type(guessed_cause)

        if guessed_service_norm == self.rca_service:
            if normalized_guess_cause == self.rca_cause:
                val = REWARD_TABLE["submit_diagnosis_perfect"]
                return val, {"diagnosis": val}, "correct service and canonical cause"

            val = REWARD_TABLE["submit_diagnosis_partial"]
            return (
                val,
                {"diagnosis": val},
                f"correct service, cause mismatch "
                f"(got '{normalized_guess_cause}', expected '{self.rca_cause}')",
            )

        val = REWARD_TABLE["submit_diagnosis_wrong"]
        return (
            val,
            {"diagnosis": val},
            f"wrong service (expected '{self.rca_service}', got '{guessed_service_norm}')",
        )

    def applying_step_penalty(
        self, total: float, breakdown: dict
    ) -> tuple[float, dict]:
        """Applied every step as an efficiency incentive (-0.01 per step)."""
        total += REWARD_TABLE["step_penalty"]
        breakdown["step_penalty"] = REWARD_TABLE["step_penalty"]
        return round(total, 4), breakdown
