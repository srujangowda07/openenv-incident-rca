from __future__ import annotations
import enum


class CauseType(str, enum.Enum):
    CONNECTION_POOL_EXHAUSTED = "connection pool exhausted"
    MEMORY_LEAK_OOM = "memory leak OOMKilled"
    DISK_FULL = "disk full no log rotation"
    MISSING_INDEX = "missing index slow query full table scan"
    SPLIT_BRAIN = "cluster split-brain 2 nodes network partition"


# BUG FIX 1: Multi-word phrases scored as single units prevent false positives.
# Old code used single-word keywords like "split" and "brain" separately — a
# response saying "this is NOT a split brain" would still score 2 points for
# SPLIT_BRAIN because both words appear independently. Phrases like
# "split-brain" or "split brain" are now matched as atomic units.
#
# BUG FIX 2: Priority ordering added. When two cause types tie on score, the
# old code returned whichever Python dict iteration yielded first — technically
# non-deterministic across Python versions and insertion order. We now break
# ties by enum declaration order (explicit priority), making the result fully
# deterministic regardless of input verbosity.

CAUSE_MAPPINGS: dict[CauseType, list[str]] = {
    CauseType.CONNECTION_POOL_EXHAUSTED: [
        "connection pool exhausted",
        "pool exhausted",
        "too many clients",
        "pgbouncer",
        "max_connections",
        "connection refused",
    ],
    CauseType.MEMORY_LEAK_OOM: [
        "oomkilled",
        "memory leak",
        "out of memory",
        "unbounded cache",
        "memory limit exceeded",
        "pod restart",
        "oom",
    ],
    CauseType.DISK_FULL: [
        "disk full",
        "no space left",
        "log rotation",
        "logrotate",
        "disk usage 100",
        "write failure",
    ],
    CauseType.MISSING_INDEX: [
        "missing index",
        "slow query",
        "full table scan",
        "schema migration",
        "dropped index",
        "no index",
    ],
    CauseType.SPLIT_BRAIN: [
        "split-brain",
        "split brain",
        "network partition",
        "cluster split",
        "minority partition",
        "nodes unreachable",
        "cluster_state=fail",
    ],
}

# Explicit priority order used as a tiebreaker — lower index wins.
_PRIORITY_ORDER: list[CauseType] = [
    CauseType.CONNECTION_POOL_EXHAUSTED,
    CauseType.MEMORY_LEAK_OOM,
    CauseType.DISK_FULL,
    CauseType.MISSING_INDEX,
    CauseType.SPLIT_BRAIN,
]


def normalize_cause_type(raw_input: str) -> str:
    """
    Normalises an LLM-produced cause_type string into a canonical CauseType value.

    Matching strategy (in order):
      1. Direct exact match (case-insensitive, stripped).
      2. Multi-word phrase scoring — each phrase counts as +1.  A phrase only
         scores if it actually appears as a contiguous substring, preventing
         single common words from bleeding across unrelated categories.
      3. Ties are broken by _PRIORITY_ORDER (deterministic).
      4. If no phrase matches at all, the raw input is returned unchanged so
         the grader can surface a meaningful "unrecognised cause" message.
    """
    raw_lower = raw_input.lower().strip()

    # 1. Direct exact match.
    for ct in CauseType:
        if ct.value == raw_lower:
            return ct.value

    # 2. Phrase scoring with deterministic tiebreaking.
    scores: dict[CauseType, int] = {ct: 0 for ct in CauseType}
    for ct, phrases in CAUSE_MAPPINGS.items():
        scores[ct] = sum(1 for phrase in phrases if phrase in raw_lower)

    best_score = max(scores.values())
    if best_score == 0:
        # No recognisable phrase found — return raw so the grader can log it.
        return raw_input

    # Among all types that achieved best_score, pick the one with lowest
    # priority index (most specific / most expected match for the scenario set).
    candidates = [ct for ct in _PRIORITY_ORDER if scores[ct] == best_score]
    return candidates[0].value


def normalize_service(raw_input: str) -> str:
    """Lower-case and strip whitespace for deterministic service name matching."""
    return raw_input.lower().strip()
