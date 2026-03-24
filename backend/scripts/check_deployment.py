#!/usr/bin/env python3
"""
scripts/check_deployment.py — Pre-deployment readiness checker.

Verifies that all required files, packages, and resources are available
before deploying to Railway.

Usage:
    python scripts/check_deployment.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# Resolve project root (backend/)
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

# Add backend to sys.path so we can import app.config
sys.path.insert(0, str(BACKEND_DIR))


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def check_files():
    """Check for required model and data files."""
    print("\n═══ File Checks ═══\n")

    checks = {
        # Data files
        "data/train_data.csv": BACKEND_DIR / "data" / "train_data.csv",
        "data/valid_data.csv": BACKEND_DIR / "data" / "valid_data.csv",
        # Model files
        "models/logistic_regression.pkl": BACKEND_DIR / "models" / "logistic_regression.pkl",
        "models/tfidf_vectorizer.pkl": BACKEND_DIR / "models" / "tfidf_vectorizer.pkl",
        "models/tokenizer.pkl": BACKEND_DIR / "models" / "tokenizer.pkl",
        "models/lstm_model.h5": BACKEND_DIR / "models" / "lstm_model.h5",
        "models/bilstm_model.h5": BACKEND_DIR / "models" / "bilstm_model.h5",
        "models/cnn_model.h5": BACKEND_DIR / "models" / "cnn_model.h5",
        "models/distilbert_model/": BACKEND_DIR / "models" / "distilbert_model",
        "models/distilbert_tokenizer/": BACKEND_DIR / "models" / "distilbert_tokenizer",
        # Report files
        "reports/model_comparison.json": BACKEND_DIR / "reports" / "model_comparison.json",
    }

    found = 0
    missing = 0
    total_size = 0

    for label, path in checks.items():
        if path.exists():
            size = _dir_size(path)
            total_size += size
            print(f"  ✅ {label} ({_human_size(size)})")
            found += 1
        else:
            print(f"  ❌ {label} — NOT FOUND")
            missing += 1

    print(f"\n  Found: {found}/{found + missing} | Total size: {_human_size(total_size)}")
    return missing == 0


def check_packages():
    """Check that required Python packages are importable."""
    print("\n═══ Package Checks ═══\n")

    packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "sklearn",
        "nltk": "nltk",
        "tensorflow": "tensorflow",
        "torch": "torch",
        "transformers": "transformers",
    }

    installed = 0
    missing_pkgs = []

    for name, module in packages.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "?")
            print(f"  ✅ {name} ({version})")
            installed += 1
        except ImportError:
            print(f"  ❌ {name} — NOT INSTALLED")
            missing_pkgs.append(name)

    print(f"\n  Installed: {installed}/{installed + len(missing_pkgs)}")
    return len(missing_pkgs) == 0


def check_ram():
    """Estimate available RAM."""
    print("\n═══ Memory Check ═══\n")
    try:
        import psutil
        mem = psutil.virtual_memory()
        total = _human_size(mem.total)
        available = _human_size(mem.available)
        used_pct = mem.percent
        print(f"  Total RAM:     {total}")
        print(f"  Available RAM: {available}")
        print(f"  Used:          {used_pct}%")

        avail_mb = mem.available / (1024 * 1024)
        if avail_mb < 512:
            print("  ⚠️  Available RAM < 512 MB — may not fit all models")
            return False
        elif avail_mb < 2048:
            print("  ⚠️  Available RAM < 2 GB — recommend LIGHTWEIGHT_MODE=true")
            return True
        else:
            print("  ✅ Sufficient RAM for all models")
            return True
    except ImportError:
        print("  ⚠️  psutil not installed — cannot check RAM")
        print("     Install with: pip install psutil")
        return True


def check_env():
    """Check environment variable configuration."""
    print("\n═══ Environment Checks ═══\n")

    env_path = BACKEND_DIR / ".env"
    if env_path.exists():
        print("  ✅ .env file found")
    else:
        print("  ⚠️  .env file not found (using defaults)")

    lightweight = os.getenv("LIGHTWEIGHT_MODE", "true")
    print(f"  LIGHTWEIGHT_MODE = {lightweight}")

    lazy = os.getenv("LAZY_LOADING", "true")
    print(f"  LAZY_LOADING = {lazy}")

    origins = os.getenv("ALLOWED_ORIGINS", "(not set)")
    print(f"  ALLOWED_ORIGINS = {origins}")


def main():
    print("╔══════════════════════════════════════════╗")
    print("║  Deployment Readiness Check              ║")
    print("║  Twitter Sentiment Analysis API           ║")
    print("╚══════════════════════════════════════════╝")

    files_ok = check_files()
    packages_ok = check_packages()
    ram_ok = check_ram()
    check_env()

    # Summary
    print("\n═══ Summary ═══\n")

    if files_ok and packages_ok and ram_ok:
        print("  ✅ All checks passed! Ready to deploy.")
    else:
        if not files_ok:
            print("  ⚠️  Some files are missing. Models will fail to load.")
        if not packages_ok:
            print("  ⚠️  Some packages missing. Run: pip install -r requirements.txt")
        if not ram_ok:
            print("  ⚠️  Low RAM. Use LIGHTWEIGHT_MODE=true in .env")

    # Recommendation
    print("\n═══ Recommendation ═══\n")
    try:
        import psutil
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        if avail_mb < 1024:
            print("  → Use LIGHTWEIGHT_MODE=true (only Logistic Regression)")
            print("  → Heavy models will lazy-load on demand")
        else:
            print("  → Set LIGHTWEIGHT_MODE=false to load all models at startup")
    except ImportError:
        print("  → For Railway free tier: LIGHTWEIGHT_MODE=true")
        print("  → For Railway Pro or local: LIGHTWEIGHT_MODE=false")

    print()


if __name__ == "__main__":
    main()
