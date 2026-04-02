from __future__ import annotations
from dataclasses import dataclass
from environment.canonical import normalize_cause_type, normalize_service


@dataclass
class GradeResult:
    score: float
    breakdown: dict
    passed: bool
    feedback: str


class IncidentRCAGrader:
    """
    Deterministic grader.  Score range: 0.0 – 1.0.
    Dimensions and weights match openenv.yaml exactly:

        root_cause_service  0.50
        cause_type          0.30  (only if service is correct)
        tool_evidence       0.20  (queried root cause service before diagnosing)
        penalties           variable (invalid actions, wrong diagnosis)

    Pass threshold: 0.60
    """

    PASS_THRESHOLD = 0.60

    # Dimension weights — keep in sync with openenv.yaml grader section.
    W_SERVICE  = 0.50
    W_CAUSE    = 0.30
    W_EVIDENCE = 0.20
    W_PENALTY_PER_INVALID   = 0.10
    W_PENALTY_WRONG_SERVICE = 0.20

    def grade(self, episode: dict) -> GradeResult:
        breakdown: dict[str, float] = {}
        breakdown["root_cause_service"] = self._score_service(episode)
        breakdown["cause_type"]         = self._score_cause_type(episode)
        breakdown["tool_evidence"]      = self._score_evidence(episode)
        breakdown["penalties"]          = self._score_penalties(episode)

        total = max(0.0, min(1.0, round(sum(breakdown.values()), 4)))

        return GradeResult(
            score=total,
            breakdown=breakdown,
            passed=total >= self.PASS_THRESHOLD,
            feedback=self._generate_feedback(breakdown, episode),
        )

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _score_service(self, episode: dict) -> float:
        ground_truth = normalize_service(episode["scenario"]["root_cause"]["service"])
        diagnosed    = normalize_service(episode["final_state"].get("diagnosed_service") or "")
        return self.W_SERVICE if diagnosed == ground_truth else 0.0

    def _score_cause_type(self, episode: dict) -> float:
        """
        BUG FIX: Old code called normalize_cause_type() on the agent's guess
        but compared the result against the raw ground-truth string.  This meant
        that if ground_truth was "connection pool exhausted" and the normaliser
        returned "connection pool exhausted", the comparison still failed if
        there was trailing whitespace or capitalisation in the ground-truth field.

        Fix: normalise BOTH sides before comparing.
        Cause type score is only awarded when the service is also correct
        (matches openenv.yaml spec: "only awarded if service is also correct").
        """
        ground_truth_svc   = normalize_service(episode["scenario"]["root_cause"]["service"])
        diagnosed_svc      = normalize_service(episode["final_state"].get("diagnosed_service") or "")

        if diagnosed_svc != ground_truth_svc:
            return 0.0

        # Normalise both sides so "Connection Pool Exhausted" == "connection pool exhausted".
        ground_truth_cause = normalize_cause_type(
            episode["scenario"]["root_cause"]["cause_type"]
        )
        diagnosed_cause    = normalize_cause_type(
            episode["final_state"].get("diagnosed_cause") or ""
        )
        return self.W_CAUSE if diagnosed_cause == ground_truth_cause else 0.0

    def _score_evidence(self, episode: dict) -> float:
        """
        Returns W_EVIDENCE (0.20) if the agent queried the root cause service
        with at least one tool call before submitting a diagnosis.
        Accepts both direct service queries and traces that mention the service.
        """
        ground_truth_svc = normalize_service(episode["scenario"]["root_cause"]["service"])

        for entry in episode["final_state"].get("action_history", []):
            if entry.get("action") == "submit_diagnosis":
                continue

            # Direct service parameter match.
            params = entry.get("parameters", {})
            if normalize_service(params.get("service") or "") == ground_truth_svc:
                return self.W_EVIDENCE

            # Trace result that mentions the root cause service.
            if entry.get("action") == "fetch_traces":
                result = entry.get("result") or {}
                if isinstance(result, dict):
                    for span in result.get("trace", []):
                        svc_match   = ground_truth_svc in normalize_service(span.get("service", ""))
                        error_match = ground_truth_svc in span.get("error", "").lower()
                        if svc_match or error_match:
                            return self.W_EVIDENCE

        return 0.0

    def _score_penalties(self, episode: dict) -> float:
        """
        Deductions:
          - invalid actions: -0.10 each (W_PENALTY_PER_INVALID)
          - wrong diagnosis: -0.20 flat (W_PENALTY_WRONG_SERVICE)
        Returns a negative float (or 0.0 if no penalties apply).
        """
        penalty = 0.0

        invalid = episode["info"].get("invalid_actions", 0)
        penalty -= invalid * self.W_PENALTY_PER_INVALID

        ground_truth_svc = normalize_service(episode["scenario"]["root_cause"]["service"])
        diagnosed        = normalize_service(episode["final_state"].get("diagnosed_service") or "")
        if diagnosed and diagnosed != ground_truth_svc:
            penalty -= self.W_PENALTY_WRONG_SERVICE

        return round(penalty, 4)

    # ------------------------------------------------------------------
    # Feedback generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_feedback(breakdown: dict, episode: dict) -> str:
        """
        BUG FIX: Old code used hardcoded 0.5 and 0.3 thresholds to check
        whether service/cause dimensions were awarded, but those magic numbers
        were copies of the weight constants.  Using the class constants directly
        (or checking > 0) is explicit and won't silently drift if weights change.
        """
        rca   = episode["scenario"]["root_cause"]
        lines = []

        if breakdown.get("root_cause_service", 0.0) == 0.0:
            lines.append(f"wrong root cause service — correct: '{rca['service']}'")

        if breakdown.get("cause_type", 0.0) == 0.0:
            if breakdown.get("root_cause_service", 0.0) > 0.0:
                # Service was correct but cause was wrong.
                diagnosed_cause = episode["final_state"].get("diagnosed_cause") or "(none)"
                lines.append(
                    f"cause type mismatch — got: '{diagnosed_cause}', "
                    f"correct: '{rca['cause_type']}'"
                )
            # If service was also wrong, the service line already covers this.

        if breakdown.get("tool_evidence", 0.0) == 0.0:
            lines.append(
                f"no tool evidence for root cause service '{rca['service']}' before diagnosis"
            )

        invalid = episode["info"].get("invalid_actions", 0)
        if invalid > 0:
            lines.append(f"{invalid} invalid action(s) (-{invalid * 0.10:.2f})")

        return " | ".join(lines) if lines else "correct"
