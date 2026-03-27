#!/usr/bin/env python3
"""
Validate a backend rollout for the Twitter Sentiment API.

Checks:
1. Health and docs endpoints
2. CORS behavior for frontend origin
3. Single prediction for each model
4. All-model prediction includes all five models
5. Stability rounds (repeat all-model prediction)

Usage:
    python scripts/validate_rollout.py --base-url https://your-service.example.com
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, List, Tuple

DEFAULT_FRONTEND_ORIGIN = "https://twitter-sentiment-analysis-mocha.vercel.app"
MODEL_NAMES = [
    "logistic_regression",
    "lstm",
    "bilstm",
    "cnn",
    "distilbert",
]
SAMPLE_TEXT = "The new update is useful and performance looks better than before."


def _load_http_client():
    try:
        import httpx as http_client
        return "httpx", http_client
    except ImportError:
        try:
            import requests as http_client
            return "requests", http_client
        except ImportError:
            print("ERROR: Install httpx or requests first (pip install httpx).")
            sys.exit(1)


CLIENT_KIND, HTTP = _load_http_client()


def _json_or_text(response) -> Any:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            return response.json()
        except Exception:
            return response.text
    return response.text


def _get(url: str, timeout: int = 60, headers: Dict[str, str] | None = None):
    if CLIENT_KIND == "httpx":
        return HTTP.get(url, timeout=timeout, headers=headers)
    return HTTP.get(url, timeout=timeout, headers=headers)


def _post(
    url: str,
    payload: Dict[str, Any],
    timeout: int = 120,
    headers: Dict[str, str] | None = None,
):
    if CLIENT_KIND == "httpx":
        return HTTP.post(url, json=payload, timeout=timeout, headers=headers)
    return HTTP.post(url, json=payload, timeout=timeout, headers=headers)


def _ok(
    checks: List[Tuple[str, bool, str]],
    name: str,
    condition: bool,
    detail: str = "",
) -> None:
    checks.append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    if detail:
        print(f"[{status}] {name}: {detail}")
    else:
        print(f"[{status}] {name}")


def _detail(payload: Any, max_chars: int = 220) -> str:
    if isinstance(payload, dict):
        if "detail" in payload:
            text = str(payload["detail"])
        else:
            text = str(payload)
    else:
        text = str(payload)
    return text[:max_chars]


def _base(url: str) -> str:
    return url.rstrip("/")


def run(args) -> int:
    checks: List[Tuple[str, bool, str]] = []
    base_url = _base(args.base_url)

    print(f"Validating backend: {base_url}")
    print(f"Frontend origin: {args.frontend_origin}")
    print(f"Client: {CLIENT_KIND}\n")

    # 1) Health
    health_url = f"{base_url}/api/health"
    health_response = None
    health_payload = None
    health_error = None
    for attempt in range(1, args.health_retries + 1):
        try:
            health_response = _get(health_url, timeout=args.health_timeout)
            health_payload = _json_or_text(health_response)
            if health_response.status_code == 200:
                break
            health_error = f"status={health_response.status_code}, body={_detail(health_payload)}"
        except Exception as exc:
            health_error = str(exc)
        if attempt < args.health_retries:
            time.sleep(args.health_retry_sleep)

    if health_response is None:
        _ok(checks, "Health endpoint reachable", False, str(health_error))
    else:
        is_json = isinstance(health_payload, dict)
        _ok(checks, "Health endpoint reachable", health_response.status_code == 200, f"status={health_response.status_code}")
        _ok(checks, "Health payload is JSON", is_json, f"type={type(health_payload).__name__}")
        if is_json:
            _ok(checks, "Health includes data_loaded", "data_loaded" in health_payload, f"value={health_payload.get('data_loaded')}")
            _ok(checks, "Health includes models_loaded", "models_loaded" in health_payload, f"value={health_payload.get('models_loaded')}")
            _ok(checks, "Health includes available_models", "available_models" in health_payload, f"value={health_payload.get('available_models')}")
            available = health_payload.get("available_models")
            has_any_model = isinstance(available, list) and len(available) >= 1
            _ok(checks, "Health has at least one loaded model", has_any_model, f"value={available}")

            lightweight = health_payload.get("lightweight_mode")
            lazy_loading = health_payload.get("lazy_loading")
            if args.mode == "render-lite":
                _ok(checks, "Health lightweight mode true", lightweight is True, f"value={lightweight}")
                _ok(checks, "Health lazy loading true", lazy_loading is True, f"value={lazy_loading}")
            elif args.mode == "hf-full":
                _ok(checks, "Health lightweight mode false", lightweight is False, f"value={lightweight}")
                _ok(checks, "Health lazy loading false", lazy_loading is False, f"value={lazy_loading}")
                _ok(
                    checks,
                    "HF full mode preloaded all models",
                    isinstance(available, list) and len(available) >= 5,
                    f"value={available}",
                )

    # 2) Docs
    docs_url = f"{base_url}/docs"
    try:
        response = _get(docs_url, timeout=60)
        body = response.text if hasattr(response, "text") else str(_json_or_text(response))
        _ok(checks, "Docs endpoint reachable", response.status_code == 200, f"status={response.status_code}")
        _ok(checks, "Docs page looks like Swagger UI", "swagger" in body.lower(), "contains 'swagger'")
    except Exception as exc:
        _ok(checks, "Docs endpoint reachable", False, str(exc))

    # 3) CORS
    cors_url = f"{base_url}/api/cors-test"
    try:
        response = _get(cors_url, timeout=60, headers={"Origin": args.frontend_origin})
        allow_origin = response.headers.get("access-control-allow-origin", "")
        _ok(checks, "CORS test endpoint reachable", response.status_code == 200, f"status={response.status_code}")
        _ok(
            checks,
            "CORS allows frontend origin",
            allow_origin == args.frontend_origin,
            f"header={allow_origin or '(missing)'}",
        )
    except Exception as exc:
        _ok(checks, "CORS allows frontend origin", False, str(exc))

    # 4) Single-model checks
    predict_url = f"{base_url}/api/predict/"
    for model_name in MODEL_NAMES:
        try:
            started = time.time()
            response = _post(
                predict_url,
                {"text": SAMPLE_TEXT, "model_name": model_name},
                timeout=args.single_model_timeout,
            )
            elapsed = time.time() - started
            payload = _json_or_text(response)
            ok_status = response.status_code == 200 and isinstance(payload, dict)
            detail = _detail(payload)
            _ok(
                checks,
                f"Single prediction ({model_name})",
                ok_status,
                f"status={response.status_code}, time={elapsed:.2f}s, body={detail}",
            )
            if ok_status:
                _ok(
                    checks,
                    f"Single prediction model_used ({model_name})",
                    payload.get("model_used") == model_name,
                    f"value={payload.get('model_used')}",
                )
        except Exception as exc:
            _ok(checks, f"Single prediction ({model_name})", False, str(exc))

    # 5) All-model endpoint
    all_models_url = f"{base_url}/api/predict/all"
    try:
        started = time.time()
        response = _post(
            all_models_url,
            {"text": SAMPLE_TEXT},
            timeout=args.all_models_timeout,
        )
        elapsed = time.time() - started
        payload = _json_or_text(response)
        ok_status = response.status_code == 200 and isinstance(payload, dict)
        _ok(
            checks,
            "All-model prediction endpoint",
            ok_status,
            f"status={response.status_code}, time={elapsed:.2f}s, body={_detail(payload)}",
        )
        if ok_status:
            results = payload.get("results", [])
            model_used = {item.get("model_used") for item in results if isinstance(item, dict)}
            _ok(checks, "All-model returns 5 results", len(results) >= 5, f"count={len(results)}")
            _ok(checks, "All-model contains all model names", set(MODEL_NAMES).issubset(model_used), f"returned={sorted(model_used)}")
    except Exception as exc:
        _ok(checks, "All-model prediction endpoint", False, str(exc))

    # 6) Stability rounds
    for i in range(args.stability_rounds):
        try:
            started = time.time()
            response = _post(
                all_models_url,
                {"text": f"{SAMPLE_TEXT} [round {i + 1}]"},
                timeout=args.all_models_timeout,
            )
            elapsed = time.time() - started
            payload = _json_or_text(response)
            ok_round = response.status_code == 200 and isinstance(payload, dict) and len(payload.get("results", [])) >= 5
            _ok(
                checks,
                f"Stability round {i + 1}/{args.stability_rounds}",
                ok_round,
                f"status={response.status_code}, time={elapsed:.2f}s, body={_detail(payload)}",
            )
        except Exception as exc:
            _ok(checks, f"Stability round {i + 1}/{args.stability_rounds}", False, str(exc))

        if i < args.stability_rounds - 1:
            time.sleep(args.sleep_seconds)

    # 7) Optional idle wake-up check
    if args.idle_seconds > 0:
        print(f"\nSleeping {args.idle_seconds}s to simulate idle cooldown...")
        time.sleep(args.idle_seconds)
        try:
            started = time.time()
            response = _post(
                all_models_url,
                {"text": f"{SAMPLE_TEXT} [idle-wakeup]"},
                timeout=args.all_models_timeout,
            )
            elapsed = time.time() - started
            payload = _json_or_text(response)
            ok_idle = response.status_code == 200 and isinstance(payload, dict) and len(payload.get("results", [])) >= 5
            _ok(
                checks,
                "Idle wake-up all-model check",
                ok_idle,
                f"status={response.status_code}, time={elapsed:.2f}s, body={_detail(payload)}",
            )
        except Exception as exc:
            _ok(checks, "Idle wake-up all-model check", False, str(exc))

    passed = sum(1 for _, ok, _ in checks if ok)
    failed = len(checks) - passed
    print("\n" + "=" * 64)
    print(f"Validation complete: {passed} passed, {failed} failed")
    print("=" * 64)
    return 0 if failed == 0 else 1


def parse_args():
    parser = argparse.ArgumentParser(description="Validate backend rollout for all 5 models.")
    parser.add_argument(
        "--base-url",
        required=True,
        help="Backend base URL, e.g. https://my-api.onrender.com",
    )
    parser.add_argument(
        "--frontend-origin",
        default=DEFAULT_FRONTEND_ORIGIN,
        help="Frontend origin for CORS check.",
    )
    parser.add_argument(
        "--mode",
        choices=["render-lite", "hf-full", "custom"],
        default="render-lite",
        help=(
            "Validation profile: "
            "render-lite checks lightweight/lazy=true, "
            "hf-full checks lightweight/lazy=false and 5 preloaded models, "
            "custom skips mode assertions."
        ),
    )
    parser.add_argument(
        "--stability-rounds",
        type=int,
        default=3,
        help="Number of repeated all-model checks.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=8,
        help="Delay between stability rounds.",
    )
    parser.add_argument(
        "--single-model-timeout",
        type=int,
        default=240,
        help="Timeout for each single-model prediction request (seconds).",
    )
    parser.add_argument(
        "--all-models-timeout",
        type=int,
        default=420,
        help="Timeout for each /predict/all request (seconds).",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=90,
        help="Timeout for each health request (seconds).",
    )
    parser.add_argument(
        "--health-retries",
        type=int,
        default=3,
        help="Retry attempts for health check (helps with free-tier cold starts).",
    )
    parser.add_argument(
        "--health-retry-sleep",
        type=int,
        default=15,
        help="Seconds to sleep between health retries.",
    )
    parser.add_argument(
        "--idle-seconds",
        type=int,
        default=0,
        help="Optional long sleep before a final wake-up check (e.g. 960 for 16 minutes).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
