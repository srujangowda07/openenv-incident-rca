from datetime import datetime, timedelta


def _log(base: datetime, offset_min: int, service: str,
         level: str, message: str) -> dict:
    t = base + timedelta(minutes=offset_min)
    return {
        "timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "service": service,
        "level": level,
        "message": message,
    }


def _gen_timestamps(base: datetime, count: int) -> list[str]:
    return [
        (base + timedelta(minutes=i - count)).strftime("%H:%M")
        for i in range(count)
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIUM_002: CPU Throttling Cascade
#  Root cause: cpu_limit set too low in kubernetes deployment
# ─────────────────────────────────────────────────────────────────────────────

def medium_cpu_throttling() -> dict:
    base = datetime(2026, 3, 20, 16, 0, 0)
    return {
        "description": (
            "INCIDENT ALERT: checkout-service latency p99 spiked from 300ms to 14s. "
            "Started at 16:02 UTC after a Kubernetes deployment. "
            "5 services in scope. Find what the deploy changed that caused this."
        ),
        "alerts": [
            {"id": "ALT-001", "service": "checkout-service", "severity": "critical",
             "message": "p99 latency 14s — SLA breach", "time": "16:02:10"},
            {"id": "ALT-002", "service": "pricing-engine", "severity": "warning",
             "message": "CPU throttling: 92% of CPU time throttled", "time": "16:01:55"},
            {"id": "ALT-003", "service": "cart-service", "severity": "warning",
             "message": "Upstream pricing-engine responding slowly", "time": "16:02:30"},
        ],
        "services": [
            {"name": "api-gateway", "status": "healthy",
             "calls": ["checkout-service"]},
            {"name": "checkout-service", "status": "degraded",
             "calls": ["cart-service", "pricing-engine"]},
            {"name": "cart-service", "status": "degraded",
             "calls": ["pricing-engine"]},
            {"name": "pricing-engine", "status": "throttled",
             "calls": ["product-catalog"]},
            {"name": "product-catalog", "status": "healthy",
             "calls": []},
        ],
        "dependency_graph": {
            "api-gateway": ["checkout-service"],
            "checkout-service": ["cart-service", "pricing-engine"],
            "cart-service": ["pricing-engine"],
            "pricing-engine": ["product-catalog"],
            "product-catalog": [],
        },
        "logs": [
            _log(base, -5, "pricing-engine", "INFO",
                 "New deploy: kubernetes cpu_limit changed from 2000m to 100m"),
            _log(base, -3, "pricing-engine", "WARN",
                 "CPU throttled: container limited to 100m cores — 92% throttle ratio"),
            _log(base, -2, "pricing-engine", "WARN",
                 "Request processing time 8000ms — CPU starvation"),
            _log(base, -1, "cart-service", "ERROR",
                 "pricing-engine timeout after 10000ms"),
            _log(base, 0, "checkout-service", "ERROR",
                 "Upstream pricing-engine and cart-service both timing out"),
            _log(base, 1, "api-gateway", "WARN",
                 "checkout-service p99 latency 14000ms"),
        ],
        "metrics": [
            {"service": "pricing-engine", "metric": "cpu_throttle_ratio",
             "values": [0.01, 0.02, 0.88, 0.91, 0.92, 0.92], "unit": "ratio",
             "timestamps": _gen_timestamps(base, 6)},
            {"service": "pricing-engine", "metric": "cpu_limit_millicores",
             "values": [2000, 2000, 100, 100, 100, 100], "unit": "millicores",
             "timestamps": _gen_timestamps(base, 6)},
            {"service": "checkout-service", "metric": "p99_latency_ms",
             "values": [290, 310, 2100, 9000, 13800, 14000], "unit": "ms",
             "timestamps": _gen_timestamps(base, 6)},
        ],
        "traces": {
            "req-chk-3a1b": [
                {"service": "api-gateway", "duration_ms": 14050, "status": "ok"},
                {"service": "checkout-service", "duration_ms": 14000, "status": "slow"},
                {"service": "pricing-engine", "duration_ms": 13900, "status": "slow",
                 "error": "CPU throttled — container limited to 100m millicores"},
            ]
        },
        "runbooks": {
            "kubernetes_cpu_throttling": (
                "1. Run kubectl describe pod <pod-name> — check cpu limits. "
                "2. Check HPA/VPA configuration. "
                "3. Increase cpu_limit in deployment manifest. "
                "4. Rollback deployment if accidental limit reduction."
            ),
        },
        "root_cause": {
            "service": "pricing-engine",
            "cause_type": "kubernetes cpu limit misconfigured throttling",
            "trigger": "deploy at 15:55 accidentally set cpu_limit from 2000m to 100m",
            "valid_fixes": [
                "rollback kubernetes deployment to previous version",
                "increase cpu_limit to 2000m in deployment manifest",
                "apply kubectl set resources deployment pricing-engine --limits cpu=2000m",
                "patch kubernetes deployment cpu limit to restore 2000m",
            ],
            "cascade": ["pricing-engine", "cart-service", "checkout-service"],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HARD_002: DNS Resolution Failure — Multi-Service Impact
#  Red herring: a noisy but unrelated certificate expiry alert
# ─────────────────────────────────────────────────────────────────────────────

def hard_dns_failure() -> dict:
    base = datetime(2026, 3, 21, 7, 0, 0)
    return {
        "description": (
            "CRITICAL P0: 8 services reporting connection failures to external dependencies. "
            "Started 07:03 UTC. Alerts from multiple teams simultaneously. "
            "One team reports a certificate expiry — may or may not be related. "
            "10 services in scope. Find the single root cause."
        ),
        "alerts": [
            {"id": "ALT-001", "service": "payment-gateway", "severity": "critical",
             "message": "stripe.com connection failed: Name or service not known", "time": "07:03:00"},
            {"id": "ALT-002", "service": "email-service", "severity": "critical",
             "message": "smtp.mailgun.org resolution failed", "time": "07:03:05"},
            {"id": "ALT-003", "service": "auth-service", "severity": "critical",
             "message": "oauth2.googleapis.com DNS lookup failed", "time": "07:03:10"},
            {"id": "ALT-004", "service": "cdn-service", "severity": "warning",
             "message": "TLS certificate expires in 2 days — renewal pending",  # RED HERRING
             "time": "07:00:00"},
            {"id": "ALT-005", "service": "monitoring", "severity": "critical",
             "message": "datadog.com unreachable — metrics pipeline down", "time": "07:03:15"},
            {"id": "ALT-006", "service": "feature-flags", "severity": "warning",
             "message": "launchdarkly.com connection timeout", "time": "07:03:20"},
            {"id": "ALT-007", "service": "core-dns", "severity": "critical",
             "message": "Upstream DNS resolver 8.8.8.8 unreachable from pod network",
             "time": "07:02:50"},
        ],
        "services": [
            {"name": "payment-gateway", "status": "critical", "calls": ["stripe-api"]},
            {"name": "email-service", "status": "critical", "calls": ["mailgun-api"]},
            {"name": "auth-service", "status": "critical", "calls": ["google-oauth"]},
            {"name": "monitoring", "status": "degraded", "calls": ["datadog-api"]},
            {"name": "feature-flags", "status": "degraded", "calls": ["launchdarkly-api"]},
            {"name": "cdn-service", "status": "warning", "calls": []},
            {"name": "core-dns", "status": "critical", "calls": ["upstream-resolver"]},
            {"name": "upstream-resolver", "status": "unreachable", "calls": []},
            {"name": "api-gateway", "status": "degraded",
             "calls": ["payment-gateway", "auth-service"]},
        ],
        "dependency_graph": {
            "api-gateway": ["payment-gateway", "auth-service"],
            "payment-gateway": ["stripe-api"],
            "email-service": ["mailgun-api"],
            "auth-service": ["google-oauth"],
            "monitoring": ["datadog-api"],
            "feature-flags": ["launchdarkly-api"],
            "cdn-service": [],
            "core-dns": ["upstream-resolver"],
            "upstream-resolver": [],
        },
        "logs": [
            _log(base, -5, "network-ops", "INFO",
                 "Firewall rule update: egress ACL rule 47 modified — UDP port 53 blocked on prod-vpc"),
            _log(base, -3, "core-dns", "ERROR",
                 "Upstream resolver 8.8.8.8:53 unreachable — UDP 53 blocked by firewall ACL"),
            _log(base, -2, "core-dns", "CRITICAL",
                 "All external DNS resolution failing — serving NXDOMAIN for all external hostnames"),
            _log(base, 0, "payment-gateway", "ERROR",
                 "socket.gaierror: [Errno -2] Name or service not known: stripe.com"),
            _log(base, 0, "auth-service", "ERROR",
                 "dns.resolver.NXDOMAIN: oauth2.googleapis.com"),
            _log(base, 0, "email-service", "ERROR",
                 "smtplib.SMTPConnectError: Could not resolve smtp.mailgun.org"),
            _log(base, 0, "cdn-service", "WARN",
                 "TLS certificate expires 2026-03-23 — acme renewal job pending"),  # Red herring
            _log(base, 1, "monitoring", "ERROR",
                 "Failed to flush metrics: datadog.com NXDOMAIN"),
            _log(base, 1, "feature-flags", "ERROR",
                 "launchdarkly SDK init failed: connection timeout on DNS lookup"),
        ],
        "metrics": [
            {"service": "core-dns", "metric": "dns_resolution_success_rate",
             "values": [99.9, 99.8, 100.0, 0.0, 0.0, 0.0], "unit": "percent",
             "timestamps": _gen_timestamps(base, 6)},
            {"service": "payment-gateway", "metric": "external_call_success_rate",
             "values": [99.5, 99.3, 99.2, 0.0, 0.0, 0.0], "unit": "percent",
             "timestamps": _gen_timestamps(base, 6)},
            {"service": "cdn-service", "metric": "cert_days_remaining",
             "values": [4, 4, 4, 2, 2, 2], "unit": "days",
             "timestamps": _gen_timestamps(base, 6)},  # Red herring metric
        ],
        "traces": {
            "req-pay-7g8h": [
                {"service": "api-gateway", "duration_ms": 3000, "status": "error"},
                {"service": "payment-gateway", "duration_ms": 2990, "status": "error",
                 "error": "DNS resolution failed: stripe.com NXDOMAIN — core-dns upstream blocked"},
            ],
            "req-auth-2e3f": [
                {"service": "api-gateway", "duration_ms": 3000, "status": "error"},
                {"service": "auth-service", "duration_ms": 2985, "status": "error",
                 "error": "DNS resolution failed: oauth2.googleapis.com — upstream resolver unreachable"},
            ],
        },
        "runbooks": {
            "dns_resolution_failure": (
                "1. Run kubectl exec -it <pod> -- nslookup google.com. "
                "2. Check CoreDNS pod logs: kubectl logs -n kube-system -l k8s-app=kube-dns. "
                "3. Verify upstream resolver connectivity: dig @8.8.8.8 google.com. "
                "4. Check network ACLs for UDP port 53 egress rules."
            ),
            "firewall_acl": (
                "1. Review recent ACL rule changes in cloud console. "
                "2. Check if UDP port 53 egress is permitted. "
                "3. Revert accidental rule changes."
            ),
        },
        "root_cause": {
            "service": "core-dns",
            "cause_type": "DNS resolution failure firewall UDP 53 blocked",
            "trigger": "firewall ACL rule update at 06:55 blocked UDP port 53 egress from prod-vpc",
            "valid_fixes": [
                "revert firewall ACL rule that blocked UDP port 53",
                "add egress rule allowing UDP port 53 to upstream DNS resolvers",
                "restore firewall rule to permit DNS traffic UDP 53",
                "rollback network ACL change from 06:55 deployment",
            ],
            "cascade": [
                "core-dns", "payment-gateway", "auth-service",
                "email-service", "monitoring", "feature-flags", "api-gateway"
            ],
        },
    }
