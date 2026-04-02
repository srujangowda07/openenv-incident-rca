from __future__ import annotations
import random
from datetime import datetime, timedelta


class ScenarioGenerator:
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, task_id: str) -> dict:
        difficulty = task_id.split("_")[0]
        variant = int(task_id.split("_")[1]) if "_" in task_id else 1
        return {
            "easy":   self._easy,
            "medium": self._medium,
            "hard":   self._hard,
        }.get(difficulty, self._easy)(variant)


    def _easy(self, variant: int) -> dict:
        return [
            self._easy_db_pool,
            self._easy_oom,
            self._easy_disk,
        ][(variant - 1) % 3]()

    def _easy_db_pool(self) -> dict:
        base = datetime(2026, 3, 15, 14, 30, 0)
        return {
            "description": (
                "INCIDENT ALERT: api-gateway is returning HTTP 502 errors to 40% of users. "
                "Started 14:32 UTC. 3 services in scope. Diagnose and fix."
            ),
            "alerts": [
                {"id": "ALT-001", "service": "api-gateway", "severity": "critical",
                 "message": "HTTP 502 error rate > 35%", "time": "14:32:05"},
                {"id": "ALT-002", "service": "user-service", "severity": "warning",
                 "message": "Response latency p99 > 5000ms", "time": "14:32:10"},
            ],
            "services": [
                {"name": "api-gateway",      "status": "degraded",  "calls": ["user-service"]},
                {"name": "user-service",     "status": "degraded",  "calls": ["postgres-primary"]},
                {"name": "postgres-primary", "status": "unhealthy", "calls": []},
            ],
            "dependency_graph": {
                "api-gateway": ["user-service"],
                "user-service": ["postgres-primary"],
                "postgres-primary": [],
            },
            "logs": [
                self._log(base, -8, "user-service",     "INFO",     "Starting request GET /api/user/profile"),
                self._log(base, -5, "postgres-primary", "WARN",     "Connection pool at 95/100"),
                self._log(base, -3, "postgres-primary", "ERROR",    "Connection pool exhausted: max_connections=100 reached"),
                self._log(base, -3, "user-service",     "ERROR",    "Failed to acquire DB connection after 3000ms"),
                self._log(base, -2, "api-gateway",      "ERROR",    "Upstream user-service returned 503"),
                self._log(base, -1, "postgres-primary", "ERROR",    "FATAL: too many clients: connection refused"),
                self._log(base,  0, "api-gateway",      "CRITICAL", "Circuit breaker OPEN for user-service"),
                self._log(base,  1, "postgres-primary", "ERROR",    "pg_stat_activity: 100 active, 47 idle-in-txn"),
                self._log(base,  2, "postgres-primary", "INFO",     "Waiting for connection: queue=480 timeout_ms=9000"),
            ],
            "metrics": [
                {"service": "postgres-primary", "metric": "active_connections",
                 "values": [45, 48, 97, 100, 100, 100], "unit": "count",
                 "timestamps": self._ts(base, 6)},
                {"service": "postgres-primary", "metric": "connection_wait_time_ms",
                 "values": [2, 3, 450, 4200, 8900, 9100], "unit": "ms",
                 "timestamps": self._ts(base, 6)},
                {"service": "api-gateway", "metric": "http_502_rate",
                 "values": [0, 0, 5.0, 32.0, 40.0, 40.0], "unit": "percent",
                 "timestamps": self._ts(base, 6)},
            ],
            "traces": {
                "req-7f3a": [
                    {"service": "api-gateway",      "duration_ms": 9100, "status": "error"},
                    {"service": "user-service",     "duration_ms": 9050, "status": "error"},
                    {"service": "postgres-primary", "duration_ms": 9000, "status": "timeout",
                     "error": "connection pool exhausted: max_connections=100 reached"},
                ]
            },
            "runbooks": {
                "postgres_connection_pool": (
                    "1. Check pg_stat_activity for blocking queries. "
                    "2. Increase max_connections in postgresql.conf. "
                    "3. Restart PgBouncer connection pooler. "
                    "4. Kill long-running idle connections."
                ),
            },
            "root_cause": {
                "service": "postgres-primary",
                "cause_type": "connection pool exhausted",
                "trigger": "deploy at 14:28 increased connection count per pod by 3x",
                "valid_fixes": [
                    "increase max_connections in postgresql.conf",
                    "restart pgbouncer connection pooler",
                    "rollback recent deploy that increased connection count",
                    "kill idle connections in pg_stat_activity",
                ],
                "cascade": ["postgres-primary", "user-service", "api-gateway"],
            },
        }

    def _easy_oom(self) -> dict:
        base = datetime(2026, 3, 16, 9, 15, 0)
        return {
            "description": (
                "INCIDENT ALERT: payment-service pods OOMKilled. "
                "Checkout failing for all users since 09:16 UTC."
            ),
            "alerts": [
                {"id": "ALT-001", "service": "payment-service", "severity": "critical",
                 "message": "OOMKilled: 3 pod restarts in 5 minutes", "time": "09:16:00"},
                {"id": "ALT-002", "service": "checkout-api", "severity": "critical",
                 "message": "Upstream payment-service unreachable", "time": "09:16:15"},
            ],
            "services": [
                {"name": "checkout-api",   "status": "degraded",  "calls": ["payment-service"]},
                {"name": "payment-service","status": "crashing",  "calls": ["redis-cache"]},
                {"name": "redis-cache",    "status": "healthy",   "calls": []},
            ],
            "dependency_graph": {
                "checkout-api": ["payment-service"],
                "payment-service": ["redis-cache"],
                "redis-cache": [],
            },
            "logs": [
                self._log(base, -10, "payment-service", "INFO",     "New deploy: v2.4.1 rolled out"),
                self._log(base,  -8, "payment-service", "INFO",     "In-memory cache initialized — no TTL configured"),
                self._log(base,  -5, "payment-service", "WARN",     "Memory usage 650MB / 1024MB limit"),
                self._log(base,  -1, "payment-service", "CRITICAL", "OOMKilled: memory limit 1024Mi exceeded"),
                self._log(base,   0, "checkout-api",    "ERROR",    "payment-service pod restarting: connection refused"),
                self._log(base,   3, "payment-service", "CRITICAL", "OOMKilled again: cache grew back immediately"),
            ],
            "metrics": [
                {"service": "payment-service", "metric": "memory_usage_mb",
                 "values": [210, 310, 450, 680, 850, 1024, 0, 210], "unit": "MB",
                 "timestamps": self._ts(base, 8)},
                {"service": "payment-service", "metric": "pod_restarts",
                 "values": [0, 0, 0, 0, 1, 2, 3, 3], "unit": "count",
                 "timestamps": self._ts(base, 8)},
            ],
            "traces": {
                "req-9b2c": [
                    {"service": "checkout-api",    "duration_ms": 500, "status": "error"},
                    {"service": "payment-service", "duration_ms": 0,   "status": "connection_refused",
                     "error": "pod restarting: OOMKilled"},
                ]
            },
            "runbooks": {
                "oom_killed": (
                    "1. Check recent deploys for memory limit changes. "
                    "2. Increase pod memory limits. "
                    "3. Add TTL to in-memory caches."
                ),
            },
            "root_cause": {
                "service": "payment-service",
                "cause_type": "memory leak OOMKilled",
                "trigger": "deploy at 09:05 introduced unbounded in-memory cache with no TTL",
                "valid_fixes": [
                    "rollback deploy v2.4.1",
                    "increase memory limits in kubernetes deployment",
                    "add TTL to in-memory cache",
                    "fix memory leak in cache implementation",
                ],
                "cascade": ["payment-service", "checkout-api"],
            },
        }

    def _easy_disk(self) -> dict:
        base = datetime(2026, 3, 17, 22, 0, 0)
        return {
            "description": (
                "INCIDENT ALERT: logging-service stopped ingesting. "
                "Monitoring dashboards showing stale data."
            ),
            "alerts": [
                {"id": "ALT-001", "service": "logging-service", "severity": "critical",
                 "message": "Disk usage 100% on /var/log", "time": "22:00:10"},
            ],
            "services": [
                {"name": "logging-service", "status": "degraded", "calls": []},
            ],
            "dependency_graph": {"logging-service": []},
            "logs": [
                self._log(base, -180, "logging-service", "INFO",     "Log rotation cron disabled for maintenance"),
                self._log(base,  -60, "logging-service", "WARN",     "Disk usage 89% on /var/log"),
                self._log(base,   -5, "logging-service", "ERROR",    "Disk usage 99% — write failures"),
                self._log(base,    0, "logging-service", "CRITICAL", "Disk full 100% — log writes failing"),
                self._log(base,    0, "logging-service", "ERROR",    "No space left on device"),
            ],
            "metrics": [
                {"service": "logging-service", "metric": "disk_usage_percent",
                 "values": [60, 72, 81, 89, 95, 99, 100, 100], "unit": "percent",
                 "timestamps": self._ts(base, 8)},
            ],
            "traces": {},
            "runbooks": {
                "disk_full": (
                    "1. Run df -h. 2. du -sh /var/log/*. "
                    "3. Delete old logs. 4. Add logrotate policy."
                ),
            },
            "root_cause": {
                "service": "logging-service",
                "cause_type": "disk full no log rotation",
                "trigger": "log rotation cron job disabled 3 days ago during maintenance",
                "valid_fixes": [
                    "delete old compressed log files",
                    "add log rotation policy with logrotate",
                    "mount additional disk volume",
                    "re-enable log rotation cron job",
                ],
                "cascade": ["logging-service"],
            },
        }


    def _medium(self, variant: int) -> dict:
        if variant == 2:
            try:
                from data.scenarios.extra_scenarios import medium_cpu_throttling
                return medium_cpu_throttling()
            except ImportError:
                pass
        return self._medium_slow_query()

    def _medium_slow_query(self) -> dict:
        base = datetime(2026, 3, 18, 11, 0, 0)
        return {
            "description": (
                "INCIDENT ALERT: E-commerce platform degraded. "
                "product-search p99 latency 8s. order-service timeout 23%. "
                "Started 11:03 UTC. 7 services in scope."
            ),
            "alerts": [
                {"id": "ALT-001", "service": "product-search",  "severity": "warning",
                 "message": "p99 latency > 8s", "time": "11:03:00"},
                {"id": "ALT-002", "service": "order-service",   "severity": "critical",
                 "message": "Timeout rate 23%", "time": "11:03:45"},
                {"id": "ALT-003", "service": "inventory-service","severity": "warning",
                 "message": "DB query p95 > 12s", "time": "11:02:30"},
            ],
            "services": [
                {"name": "api-gateway",       "status": "healthy",  "calls": ["product-search","order-service"]},
                {"name": "product-search",    "status": "degraded", "calls": ["elasticsearch","inventory-service"]},
                {"name": "order-service",     "status": "degraded", "calls": ["inventory-service","payment-service"]},
                {"name": "inventory-service", "status": "degraded", "calls": ["mysql-primary"]},
                {"name": "payment-service",   "status": "healthy",  "calls": []},
                {"name": "mysql-primary",     "status": "degraded", "calls": []},
                {"name": "elasticsearch",     "status": "healthy",  "calls": []},
            ],
            "dependency_graph": {
                "api-gateway":       ["product-search","order-service"],
                "product-search":    ["elasticsearch","inventory-service"],
                "order-service":     ["inventory-service","payment-service"],
                "inventory-service": ["mysql-primary"],
                "payment-service":   [],
                "mysql-primary":     [],
                "elasticsearch":     [],
            },
            "logs": [
                self._log(base, -8, "mysql-primary",     "INFO",  "Schema migration 20260318_drop_sku_index executed"),
                self._log(base, -5, "mysql-primary",     "WARN",  "Slow query: 4500ms SELECT * FROM inventory WHERE sku IN (...)"),
                self._log(base, -4, "inventory-service", "WARN",  "DB query timeout after 5000ms"),
                self._log(base, -3, "mysql-primary",     "WARN",  "Slow query: 11000ms — full table scan on inventory (2.3M rows)"),
                self._log(base, -3, "order-service",     "ERROR", "inventory-service timeout: 12000ms exceeded"),
                self._log(base, -2, "product-search",    "ERROR", "inventory-service timeout on stock check"),
                self._log(base,  0, "order-service",     "ERROR", "Request timeout: /api/orders returning 504"),
            ],
            "metrics": [
                {"service": "mysql-primary",     "metric": "slow_query_count",
                 "values": [0, 0, 2, 18, 45, 78], "unit": "count/min",
                 "timestamps": self._ts(base, 6)},
                {"service": "mysql-primary",     "metric": "query_duration_p95_ms",
                 "values": [12, 14, 800, 4500, 11000, 12500], "unit": "ms",
                 "timestamps": self._ts(base, 6)},
                {"service": "inventory-service", "metric": "response_time_p99_ms",
                 "values": [45, 50, 900, 5000, 13000, 13000], "unit": "ms",
                 "timestamps": self._ts(base, 6)},
                {"service": "order-service",     "metric": "timeout_rate_percent",
                 "values": [0, 0, 2, 8, 15, 23], "unit": "percent",
                 "timestamps": self._ts(base, 6)},
            ],
            "traces": {
                "req-order-4f2a": [
                    {"service": "api-gateway",       "duration_ms": 12800, "status": "ok"},
                    {"service": "order-service",     "duration_ms": 12750, "status": "timeout"},
                    {"service": "inventory-service", "duration_ms": 12700, "status": "slow"},
                    {"service": "mysql-primary",     "duration_ms": 12650, "status": "slow",
                     "query": "SELECT * FROM inventory WHERE sku IN (...) -- missing index on sku column"},
                ],
                "req-search-9c1b": [
                    {"service": "api-gateway",       "duration_ms": 8300, "status": "ok"},
                    {"service": "product-search",    "duration_ms": 8250, "status": "slow"},
                    {"service": "inventory-service", "duration_ms": 8200, "status": "slow"},
                    {"service": "mysql-primary",     "duration_ms": 8150, "status": "slow",
                     "query": "SELECT stock_count FROM inventory WHERE sku=? -- full table scan"},
                ],
            },
            "runbooks": {
                "mysql_slow_query": (
                    "1. EXPLAIN slow queries. 2. Check missing indexes. "
                    "3. Add index or rewrite query. 4. Check table locks."
                ),
            },
            "root_cause": {
                "service": "mysql-primary",
                "cause_type": "missing index slow query full table scan",
                "trigger": "schema migration at 10:55 dropped index on inventory.sku column",
                "valid_fixes": [
                    "re-add index on inventory.sku column",
                    "create index on sku column in inventory table",
                    "rollback schema migration",
                    "add covering index on inventory sku and stock_count",
                ],
                "cascade": ["mysql-primary","inventory-service","order-service","product-search"],
            },
        }


    def _hard(self, variant: int) -> dict:
        if variant == 2:
            try:
                from data.scenarios.extra_scenarios import hard_dns_failure
                return hard_dns_failure()
            except ImportError:
                pass
        return self._hard_redis_split_brain()

    def _hard_redis_split_brain(self) -> dict:
        base = datetime(2026, 3, 19, 3, 0, 0)
        return {
            "description": (
                "CRITICAL P0: Multiple services failing simultaneously. "
                "Started 03:02 UTC. 10 services in scope. High noise. "
                "Find the SINGLE root cause driving all symptoms. Budget: 40 steps."
            ),
            "alerts": [
                {"id": "ALT-001", "service": "api-gateway",    "severity": "critical",
                 "message": "5xx rate 45%", "time": "03:02:00"},
                {"id": "ALT-002", "service": "auth-service",   "severity": "critical",
                 "message": "JWT validation failing — Redis unreachable", "time": "03:02:10"},
                {"id": "ALT-003", "service": "session-service","severity": "critical",
                 "message": "100% session lookup failure", "time": "03:02:05"},
                {"id": "ALT-004", "service": "redis-cluster",  "severity": "critical",
                 "message": "Cluster split-brain: 2 of 6 nodes unreachable", "time": "03:01:45"},
                {"id": "ALT-005", "service": "user-service",   "severity": "warning",
                 "message": "Cache miss 100% — falling back to DB", "time": "03:02:15"},
                {"id": "ALT-006", "service": "recommendation-engine", "severity": "info",
                 "message": "New ML model deployed at 03:00 — A/B test running",
                 "time": "03:00:00"},  # RED HERRING
            ],
            "services": [
                {"name": "api-gateway",            "status": "degraded",    "calls": ["auth-service","user-service","order-service"]},
                {"name": "auth-service",           "status": "critical",    "calls": ["redis-cluster","postgres-auth"]},
                {"name": "user-service",           "status": "degraded",    "calls": ["redis-cluster","postgres-users"]},
                {"name": "session-service",        "status": "critical",    "calls": ["redis-cluster"]},
                {"name": "order-service",          "status": "degraded",    "calls": ["inventory-service","payment-service"]},
                {"name": "notification-service",   "status": "degraded",    "calls": ["redis-cluster","smtp-gateway"]},
                {"name": "redis-cluster",          "status": "split-brain", "calls": []},
                {"name": "recommendation-engine",  "status": "healthy",     "calls": ["redis-cluster"]},
                {"name": "postgres-auth",          "status": "healthy",     "calls": []},
                {"name": "postgres-users",         "status": "healthy",     "calls": []},
            ],
            "dependency_graph": {
                "api-gateway":           ["auth-service","user-service","order-service"],
                "auth-service":          ["redis-cluster","postgres-auth"],
                "user-service":          ["redis-cluster","postgres-users"],
                "session-service":       ["redis-cluster"],
                "notification-service":  ["redis-cluster","smtp-gateway"],
                "recommendation-engine": ["redis-cluster"],
                "redis-cluster":         [],
                "order-service":         ["inventory-service","payment-service"],
                "postgres-auth":         [],
                "postgres-users":        [],
            },
            "logs": [
                self._log(base, -3, "network-infra",       "INFO",     "Switch sw-rack-3 firmware upgrade initiated"),
                self._log(base, -1, "redis-cluster",       "ERROR",    "Node redis-4 lost connectivity: timeout"),
                self._log(base, -1, "redis-cluster",       "ERROR",    "Node redis-5 lost connectivity: timeout"),
                self._log(base,  0, "redis-cluster",       "CRITICAL", "CLUSTER INFO: cluster_state=fail — split-brain"),
                self._log(base,  0, "auth-service",        "CRITICAL", "Redis split-brain — JWT blacklist unavailable"),
                self._log(base,  0, "session-service",     "CRITICAL", "Redis MOVED loop — cluster topology inconsistent"),
                self._log(base,  0, "user-service",        "ERROR",    "Redis failed — falling back to postgres"),
                self._log(base,  1, "api-gateway",         "CRITICAL", "auth-service returning 503: upstream Redis down"),
                self._log(base,  1, "recommendation-engine","INFO",    "A/B test running normally — no issues"),  # Red herring
                self._log(base,  2, "redis-cluster",       "ERROR",    "Minority partition refusing writes: redis-4, redis-5"),
            ],
            "metrics": [
                {"service": "redis-cluster", "metric": "reachable_nodes",
                 "values": [6, 6, 6, 4, 4, 4], "unit": "count",
                 "timestamps": self._ts(base, 6)},
                {"service": "auth-service",  "metric": "error_rate_percent",
                 "values": [0, 0, 0, 78, 85, 87], "unit": "percent",
                 "timestamps": self._ts(base, 6)},
                {"service": "session-service","metric": "lookup_success_rate",
                 "values": [100, 100, 100, 0, 0, 0], "unit": "percent",
                 "timestamps": self._ts(base, 6)},
                {"service": "recommendation-engine","metric": "p99_latency_ms",
                 "values": [45, 43, 47, 52, 50, 48], "unit": "ms",
                 "timestamps": self._ts(base, 6)},  # Red herring — healthy
            ],
            "traces": {
                "req-auth-1a2b": [
                    {"service": "api-gateway",  "duration_ms": 5100, "status": "error"},
                    {"service": "auth-service", "duration_ms": 5050, "status": "error",
                     "error": "Redis CLUSTER INFO: split-brain 2 nodes failed"},
                ],
                "req-sess-3c4d": [
                    {"service": "session-service", "duration_ms": 3000, "status": "error",
                     "error": "redis MOVED redirect loop — cluster topology inconsistent"},
                ],
            },
            "runbooks": {
                "redis_cluster_split_brain": (
                    "1. Run CLUSTER INFO on all nodes. "
                    "2. Identify minority partition. "
                    "3. CLUSTER RESET SOFT on minority nodes. "
                    "4. Re-join to majority. "
                    "5. Verify cluster_state:ok."
                ),
            },
            "root_cause": {
                "service": "redis-cluster",
                "cause_type": "cluster split-brain 2 nodes network partition",
                "trigger": "network switch firmware upgrade at 03:01 isolated redis-4 and redis-5",
                "valid_fixes": [
                    "run cluster reset soft on minority redis nodes",
                    "restore network connectivity to isolated redis nodes",
                    "rejoin minority redis nodes to majority cluster",
                    "failover redis cluster and reset topology",
                ],
                "cascade": [
                    "redis-cluster","auth-service","session-service",
                    "user-service","notification-service","api-gateway"
                ],
            },
        }


    @staticmethod
    def _log(base: datetime, offset_min: int, service: str,
             level: str, message: str) -> dict:
        t = base + timedelta(minutes=offset_min)
        return {"timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "service": service, "level": level, "message": message}

    @staticmethod
    def _ts(base: datetime, count: int) -> list[str]:
        return [(base + timedelta(minutes=i - count)).strftime("%H:%M")
                for i in range(count)]
