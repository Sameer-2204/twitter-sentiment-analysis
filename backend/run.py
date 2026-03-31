"""
run.py — Start the FastAPI backend locally with hot-reload.

Usage:
    cd backend
    python run.py

Loads .env automatically, resolves paths, and starts uvicorn
with reload enabled for development.
"""

import os
import sys
from pathlib import Path


def main() -> None:
    # ── Resolve paths ─────────────────────────────────────────
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent

    os.chdir(backend_dir)

    # ── Load .env if it exists ────────────────────────────────
    env_file = backend_dir / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
            print(f"  Loaded .env from {env_file}")
        except ImportError:
            print("  python-dotenv not installed, skipping .env loading")
            print("  Install with: pip install python-dotenv")
    else:
        env_example = backend_dir / ".env.example"
        if env_example.exists():
            print(f"  No .env found. Copy the example:")
            print(f"    copy .env.example .env")
            print()

    # ── Startup banner ────────────────────────────────────────
    print()
    print("=" * 52)
    print("  🐦 Twitter Sentiment Analyzer — Local Dev")
    print("=" * 52)
    print()
    print(f"  Project Root: {project_root}")
    print(f"  Backend Dir:  {backend_dir}")
    print(f"  Python:       {sys.version.split()[0]}")
    print()
    print("  Starting uvicorn with hot-reload...")
    print("  Press Ctrl+C to stop.")
    print()
    print("-" * 52)
    print()

    # ── Start uvicorn ─────────────────────────────────────────
    try:
        import uvicorn

        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", "8000"))

        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=[str(backend_dir / "app")],
            log_level="debug" if os.getenv("DEBUG", "true").lower() == "true" else "info",
        )
    except KeyboardInterrupt:
        print()
        print("=" * 52)
        print("  Server stopped. Goodbye!")
        print("=" * 52)
    except ImportError:
        print("  ERROR: uvicorn not installed.")
        print("  Run: pip install uvicorn[standard]")
        sys.exit(1)


if __name__ == "__main__":
    main()
