"""Container entrypoint: wait for MLflow, seed model, start FastAPI."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wait_for_mlflow(uri: str, timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(uri, timeout=3):
                print(f"MLflow is reachable at {uri}")
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise RuntimeError(f"MLflow not reachable at {uri} after {timeout_s}s")


def main() -> int:
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    print(f"Waiting for MLflow at {mlflow_uri} ...")
    wait_for_mlflow(mlflow_uri)

    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    print("Ensuring a Production model is available...")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "seed_model.py")])

    print("Starting FastAPI on port 8080...")
    os.execvp(
        "uvicorn",
        ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
