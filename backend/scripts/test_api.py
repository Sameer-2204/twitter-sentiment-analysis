#!/usr/bin/env python3
"""
test_api.py — Comprehensive API test suite for the Twitter Sentiment
Analysis backend.

Usage:
    python scripts/test_api.py [API_URL]

Default API_URL: http://localhost:8000
"""

import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)

API_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
TIMEOUT = 120  # seconds


# ── Result tracking ──────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    detail: str = ""
    status_code: int = 0


@dataclass
class TestSuite:
    results: List[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def passed(self): return sum(1 for r in self.results if r.passed)

    @property
    def failed(self): return sum(1 for r in self.results if not r.passed)

    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"{'TEST RESULTS':^70}")
        print("=" * 70)
        print(f"{'Test':<40} {'Status':<8} {'Time':>8} {'Code':>6}")
        print("-" * 70)
        for r in self.results:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            print(f"{r.name:<40} {status:<8} {r.duration:>7.3f}s {r.status_code:>6}")
            if not r.passed and r.detail:
                print(f"  → {r.detail[:80]}")
        print("-" * 70)

        durations = [r.duration for r in self.results if r.passed]
        avg = sum(durations) / len(durations) if durations else 0

        print(f"\nTotal: {len(self.results)} | "
              f"Passed: {self.passed} | "
              f"Failed: {self.failed} | "
              f"Avg response: {avg:.3f}s")
        print("=" * 70)


suite = TestSuite()
client = httpx.Client(base_url=API_URL, timeout=TIMEOUT)


# ── Helpers ──────────────────────────────────────────────────

def run_test(name: str, method: str, path: str, *,
             json_body: dict = None, expected_status: int = 200,
             check_field: str = None) -> Optional[dict]:
    """Execute a test and record the result."""
    start = time.time()
    try:
        if method == "GET":
            resp = client.get(path)
        elif method == "POST":
            resp = client.post(path, json=json_body)
        else:
            raise ValueError(f"Unknown method: {method}")

        duration = time.time() - start
        passed = resp.status_code == expected_status

        detail = ""
        data = None
        try:
            data = resp.json()
        except Exception:
            pass

        if passed and check_field and data:
            if check_field not in data:
                passed = False
                detail = f"Missing field: {check_field}"

        if not passed:
            detail = detail or f"Expected {expected_status}, got {resp.status_code}"
            if data and isinstance(data, dict):
                detail += f" | {data.get('detail', '')}"

        suite.add(TestResult(
            name=name, passed=passed, duration=duration,
            detail=detail, status_code=resp.status_code,
        ))
        return data

    except Exception as exc:
        duration = time.time() - start
        suite.add(TestResult(
            name=name, passed=False, duration=duration,
            detail=str(exc)[:120], status_code=0,
        ))
        return None


# ── Tests ────────────────────────────────────────────────────

print(f"\n🧪 Running API tests against: {API_URL}\n")

# 1. Health check
print("1. Health & Info Endpoints")
health = run_test("GET /api/health", "GET", "/api/health", check_field="status")
run_test("GET /api/info", "GET", "/api/info", check_field="version")
run_test("GET / (root)", "GET", "/", check_field="message")

# 2. Dashboard stats
print("2. Dashboard Endpoints")
run_test("GET /api/dashboard/stats", "GET", "/api/dashboard/stats")
run_test("GET /api/dashboard/trends", "GET", "/api/dashboard/trends")

# 3. EDA endpoints
print("3. EDA Endpoints")
eda_paths = [
    "/api/eda/sentiment-distribution",
    "/api/eda/text-length",
    "/api/eda/wordcloud",
    "/api/eda/top-words",
    "/api/eda/timeline",
]
for path in eda_paths:
    name = path.split("/")[-1]
    run_test(f"GET /api/eda/{name}", "GET", path)

# 4. Model comparison
print("4. Model Endpoints")
run_test("GET /api/models/comparison", "GET", "/api/models/comparison")
run_test("GET /api/predict/models", "GET", "/api/predict/models", check_field="models")

# 5. Single prediction — each model
print("5. Single Predictions")
test_text = "I love this product, it is absolutely amazing and wonderful!"
for model in ["logistic_regression", "lstm", "bilstm", "cnn", "distilbert"]:
    run_test(
        f"POST /predict ({model})", "POST", "/api/predict/",
        json_body={"text": test_text, "model_name": model},
        check_field="label",
    )

# 6. All-models prediction
print("6. All-Models Prediction")
run_test(
    "POST /predict/all", "POST", "/api/predict/all",
    json_body={"text": "The weather is nice today."},
    check_field="consensus",
)

# 7. Error handling
print("7. Error Handling")
run_test(
    "Empty text → 422", "POST", "/api/predict/",
    json_body={"text": "", "model_name": "logistic_regression"},
    expected_status=422,
)
run_test(
    "Invalid model → 400", "POST", "/api/predict/",
    json_body={"text": "test", "model_name": "fake_model"},
    expected_status=400,
)

# 8. Response time benchmarks — quick burst
print("8. Response Time Benchmark (5 rapid requests)")
times = []
for i in range(5):
    start = time.time()
    try:
        resp = client.post("/api/predict/", json={
            "text": f"Benchmark test number {i+1} for latency measurement.",
            "model_name": "logistic_regression",
        })
        elapsed = time.time() - start
        if resp.status_code == 200:
            times.append(elapsed)
    except Exception:
        pass

if times:
    avg_t = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    suite.add(TestResult(
        name=f"Benchmark (avg={avg_t:.3f}s, min={min_t:.3f}s, max={max_t:.3f}s)",
        passed=avg_t < 5.0,
        duration=avg_t,
        status_code=200,
    ))
else:
    suite.add(TestResult(
        name="Benchmark", passed=False, duration=0,
        detail="All benchmark requests failed",
    ))

# ── Summary ──────────────────────────────────────────────────

client.close()
suite.print_summary()

sys.exit(0 if suite.failed == 0 else 1)
