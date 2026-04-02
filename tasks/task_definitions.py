
TASKS = {

    "easy_001": {
        "id": "easy_001",
        "name": "Database Connection Pool Exhaustion",
        "difficulty": "easy",
        "max_steps": 15,
        "description": """
TASK: easy_001 — DB Connection Pool Exhaustion

SITUATION:
    api-gateway is returning HTTP 502 errors to 40% of users.
    The incident started at 14:32 UTC, 3 minutes ago.
    3 services are in scope: api-gateway → user-service → postgres-primary.

YOUR GOAL:
    1. Identify which service is the root cause.
    2. Identify what type of failure it is.
    3. Call submit_diagnosis with your findings.

SUCCESS CRITERIA:
    - Correctly identify root cause service (required)
    - Correctly identify cause type (required)
    - Gather evidence using grep_logs or query_metrics

DIFFICULTY NOTES:
    - Only 3 services — the dependency chain is short
    - Root cause is clearly signalled in logs and metrics
    - No red herrings
    - Complete efficiently for full strict grading score
        """.strip(),
        "hint": "Start with grep_logs on 'error' or 'connection', then query_metrics on postgres-primary.",
        "expected_root_cause": "postgres-primary — connection pool exhausted",
        "example_fix": "restart pgbouncer connection pooler",
    },

    "easy_002": {
        "id": "easy_002",
        "name": "Payment Service OOM Kill",
        "difficulty": "easy",
        "max_steps": 15,
        "description": """
TASK: easy_002 — Payment Service OOMKilled

SITUATION:
    payment-service pods are crashing with OOMKilled.
    Checkout flows failing for all users since 09:16 UTC.

YOUR GOAL:
    1. Identify the crashing service and failure mode.
    2. Find what triggered the memory growth.
    3. Call submit_diagnosis with the service and cause_type.

SUCCESS CRITERIA:
    - Diagnose payment-service as root cause
    - Identify memory leak / OOMKilled cause type
    - Find evidence in metrics/logs

DIFFICULTY NOTES:
    - 2 services in scope. Memory metrics tell the full story.
    - Logs show the deploy timestamp — connect deploy to incident start.
        """.strip(),
        "hint": "Check logs for recent deploys. Then query_metrics for memory_usage on payment-service.",
        "expected_root_cause": "payment-service — OOMKilled memory leak from unbounded cache",
        "example_fix": "rollback deploy v2.4.1 that introduced unbounded cache",
    },

    "easy_003": {
        "id": "easy_003",
        "name": "Disk Full — Logging Service Down",
        "difficulty": "easy",
        "max_steps": 15,
        "description": """
TASK: easy_003 — Disk Full on Logging Service

SITUATION:
    logging-service has stopped ingesting logs. Monitoring dashboards showing stale data.

YOUR GOAL:
    1. Identify why logging-service stopped.
    2. Find what caused disk to fill.
    3. Call submit_diagnosis.

SUCCESS CRITERIA:
    - Diagnose logging-service as root cause
    - Identify disk full cause type

DIFFICULTY NOTES:
    - Simplest scenario. 1 service. Metrics show linear disk growth.
    - Logs confirm cron job was disabled.
        """.strip(),
        "hint": "query_metrics on disk_usage for logging-service. Then grep_logs for 'rotation' or 'cron'.",
        "expected_root_cause": "logging-service — disk full, log rotation disabled",
        "example_fix": "re-enable log rotation cron job and delete old compressed logs",
    },

    "medium_001": {
        "id": "medium_001",
        "name": "Slow Query Cascade — Missing DB Index",
        "difficulty": "medium",
        "max_steps": 25,
        "description": """
TASK: medium_001 — Slow Query Cascade (Missing Index)

SITUATION:
    E-commerce platform partially degraded.
    - product-search: p99 latency 8s (was 200ms)
    - order-service: timeout rate 23%
    - Both started degrading at 11:03 UTC
    6 services in scope: api-gateway → [product-search, order-service] → inventory-service → mysql-primary

YOUR GOAL:
    1. Trace the failure through the dependency chain.
    2. Identify the single root cause causing both symptoms.
    3. Verify with tool queries (logs, traces).
    4. Call submit_diagnosis.

SUCCESS CRITERIA:
    - Identify mysql-primary as root cause (not inventory-service)
    - Identify slow query / missing index as cause type
    - Use fetch_traces to see the query in the trace

DIFFICULTY NOTES:
    - Two services are degraded but share one root cause — agent must see through the symptom layer
    - Fetching a trace will show the slow SQL query with "full table scan" in the error
    - The schema migration log entry is the key trigger event
        """.strip(),
        "hint": "Both degraded services call inventory-service. Trace inventory-service's calls downstream.",
        "expected_root_cause": "mysql-primary — missing index on inventory.sku (dropped by migration)",
        "example_fix": "CREATE INDEX idx_inventory_sku ON inventory(sku)",
    },

    "hard_001": {
        "id": "hard_001",
        "name": "Redis Cluster Split-Brain — Multi-Service Cascade",
        "difficulty": "hard",
        "max_steps": 40,
        "description": """
TASK: hard_001 — Redis Cluster Split-Brain (P0 Incident)

SITUATION:
    P0 incident. Multiple teams reporting different symptoms simultaneously.
    03:02 UTC. 10 services in scope. High noise environment.

    Symptoms:
    - api-gateway: 45% 5xx rate
    - auth-service: JWT validation failing
    - session-service: 100% lookup failure
    - user-service: cache miss 100%
    - notification-service: email queue backlog 50k msgs
    - recommendation-engine: A/B test running (deployed at 03:00)  ← POSSIBLE RED HERRING

    Multiple teams are blaming different services. Your job is to find the SINGLE root cause.

YOUR GOAL:
    1. Cut through the noise — identify which alerts are symptoms vs root cause
    2. Find the single service failure driving ALL the symptoms
    3. Call submit_diagnosis with the service and split-brain cause

SUCCESS CRITERIA:
    - Identify redis-cluster as root cause (not auth-service, not api-gateway)
    - Identify split-brain / network partition as cause type
    - Do NOT diagnose recommendation-engine (red herring)

DIFFICULTY NOTES:
    - recommendation-engine is a red herring — new deploy but NOT causing issues
    - auth-service is a VICTIM not the cause — its Redis dependency is the key
    - The network-infra log about a switch firmware upgrade is the causal trigger
    - Use query_dependencies systematically
        """.strip(),
        "hint": "Multiple services failing simultaneously → look for a shared dependency. Use query_dependencies.",
        "expected_root_cause": "redis-cluster — split-brain from network switch firmware upgrade isolating 2 nodes",
        "example_fix": "CLUSTER RESET SOFT on minority redis nodes redis-4 and redis-5",
    },
}


def get_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise KeyError(f"Task '{task_id}' not found. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks(difficulty: str | None = None) -> list[dict]:
    tasks = list(TASKS.values())
    if difficulty:
        tasks = [t for t in tasks if t["difficulty"] == difficulty]
    return tasks

