#!/usr/bin/env python3
"""
scripts/test_api.py — Smoke-test all API endpoints locally.

Requires the server to be running at http://localhost:8000.
Uses ``httpx`` (preferred) or ``requests`` as HTTP client.

Usage:
    # Start server first:
    uvicorn app.main:app --reload

    # Then in another terminal:
    python scripts/test_api.py
"""

from __future__ import annotations

import json
import sys
import time

# Try httpx first, fall back to requests
try:
    import httpx as http_client

    def _get(url, **kw):
        r = http_client.get(url, timeout=30, **kw)
        return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text

    def _post(url, **kw):
        r = http_client.post(url, timeout=60, **kw)
        return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
except ImportError:
    try:
        import requests as http_client

        def _get(url, **kw):
            r = http_client.get(url, timeout=30, **kw)
            try:
                return r.status_code, r.json()
            except Exception:
                return r.status_code, r.text

        def _post(url, **kw):
            r = http_client.post(url, timeout=60, **kw)
            try:
                return r.status_code, r.json()
            except Exception:
                return r.status_code, r.text
    except ImportError:
        print("❌ Please install httpx or requests: pip install httpx")
        sys.exit(1)


BASE_URL = "http://localhost:8000"

# Test results tracking
_passed = 0
_failed = 0
_errors = []


def _test(name: str, method: str, path: str, expected_status: int = 200, **kwargs):
    """Run a single test and print result."""
    global _passed, _failed

    url = f"{BASE_URL}{path}"
    start = time.time()

    try:
        if method == "GET":
            status, data = _get(url)
        elif method == "POST":
            status, data = _post(url, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - start

        if status == expected_status:
            _passed += 1
            print(f"  ✅ {name} → {status} ({elapsed:.2f}s)")
            return data
        else:
            _failed += 1
            detail = data.get("detail", "") if isinstance(data, dict) else str(data)[:100]
            err_msg = f"{name} → expected {expected_status}, got {status}: {detail}"
            _errors.append(err_msg)
            print(f"  ❌ {err_msg}")
            return None
    except Exception as exc:
        _failed += 1
        err_msg = f"{name} → ERROR: {exc}"
        _errors.append(err_msg)
        print(f"  ❌ {err_msg}")
        return None


def main():
    print("╔══════════════════════════════════════════╗")
    print("║  API Endpoint Tests                      ║")
    print("║  Twitter Sentiment Analysis API           ║")
    print("╚══════════════════════════════════════════╝")
    print(f"\n  Base URL: {BASE_URL}\n")

    # ── Health ──────────────────────────────────────────
    print("─── Health ───")
    data = _test("Health check", "GET", "/api/health")
    if data:
        print(f"       models_loaded={data.get('models_loaded')}, "
              f"data_loaded={data.get('data_loaded')}, "
              f"available={data.get('available_models')}")

    # ── CORS test ───────────────────────────────────────
    print("\n─── CORS ───")
    _test("CORS test", "GET", "/api/cors-test")

    # ── Root ────────────────────────────────────────────
    print("\n─── Root ───")
    _test("Root endpoint", "GET", "/")

    # ── Dashboard ───────────────────────────────────────
    print("\n─── Dashboard ───")
    _test("Dashboard stats", "GET", "/api/dashboard/stats")
    _test("Recent tweets", "GET", "/api/dashboard/recent-tweets")
    _test("Sentiment trend", "GET", "/api/dashboard/sentiment-trend")

    # ── EDA ─────────────────────────────────────────────
    print("\n─── EDA ───")
    _test("Class distribution", "GET", "/api/eda/class-distribution")
    _test("Word frequency (all)", "GET", "/api/eda/word-frequency")
    _test("Word frequency (positive)", "GET", "/api/eda/word-frequency?sentiment=positive")
    _test("Bigrams", "GET", "/api/eda/bigrams")
    _test("Trigrams", "GET", "/api/eda/trigrams")
    _test("Tweet lengths", "GET", "/api/eda/tweet-lengths")
    _test("Wordcloud data", "GET", "/api/eda/wordcloud-data")
    _test("Hashtags", "GET", "/api/eda/hashtags")
    _test("Mentions", "GET", "/api/eda/mentions")

    # ── Models ──────────────────────────────────────────
    print("\n─── Models ───")
    _test("Model comparison", "GET", "/api/models/comparison")
    _test("Confusion matrix (LR)", "GET", "/api/models/confusion-matrix/logistic_regression")
    _test("Training history (LSTM)", "GET", "/api/models/training-history/lstm")
    _test("Training history (LR → 400)", "GET",
          "/api/models/training-history/logistic_regression", expected_status=400)
    _test("Available models", "GET", "/api/models/available")

    # ── Prediction ──────────────────────────────────────
    print("\n─── Prediction ───")

    data = _test(
        "Single prediction (positive)",
        "POST",
        "/api/predict/",
        json={"text": "I absolutely love this! Best day ever!", "model_name": "logistic_regression"},
    )
    if data and isinstance(data, dict):
        label = data.get("label", "?")
        conf = data.get("confidence", "?")
        print(f"       label={label}, confidence={conf}%")

    data = _test(
        "Single prediction (negative)",
        "POST",
        "/api/predict/",
        json={"text": "This is terrible and I hate it.", "model_name": "logistic_regression"},
    )
    if data and isinstance(data, dict):
        label = data.get("label", "?")
        conf = data.get("confidence", "?")
        print(f"       label={label}, confidence={conf}%")

    data = _test(
        "All models prediction",
        "POST",
        "/api/predict/all",
        json={"text": "I hate this product so much!"},
    )
    if data and isinstance(data, dict):
        consensus = data.get("consensus", "?")
        agreement = data.get("agreement_count", "?")
        n_results = len(data.get("results", []))
        print(f"       consensus={consensus}, agreement={agreement}/{n_results}")

    _test("Sample CSV download", "GET", "/api/predict/sample-csv")

    _test(
        "Invalid model → 400",
        "POST",
        "/api/predict/",
        expected_status=400,
        json={"text": "test", "model_name": "nonexistent"},
    )

    # ── Summary ─────────────────────────────────────────
    total = _passed + _failed
    print(f"\n{'═' * 46}")
    print(f"  Results: {_passed}/{total} passed, {_failed} failed")
    print(f"{'═' * 46}")

    if _errors:
        print("\n  Failures:")
        for err in _errors:
            print(f"    • {err}")

    print()
    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
