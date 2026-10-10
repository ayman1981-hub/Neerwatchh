"""NEERWATCH entry point: starts the Streamlit dashboard + terminal simulation.

Usage
-----
    python run.py               # live random simulation loop + dashboard
    python run.py --replay      # scripted Ward 9 outbreak from demo_reports.csv
    python run.py --replay --delay 2.5 --port 8501 --no-dashboard

Flow (per spec):
  1. initialise the SQLite database (schema + wards + leak logs),
  2. launch app.py as a Streamlit subprocess,
  3. read demo_reports.csv row by row with a 2-3s delay per row,
  4. print the exact pipeline stage lines for each report,
  5. persist everything to SQLite so the 3s dashboard refresh picks it up.
"""
from __future__ import annotations

import argparse
import csv
import random
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import db, pipeline, simulate  # noqa: E402


def find_free_port(start: int = 8501, tries: int = 20) -> int:
    for port in range(start, start + tries):
        # 1) skip ports where something already accepts connections on loopback
        #    (a bind test alone misses listeners that set SO_REUSEADDR)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                continue
        # 2) confirm we can actually bind it (catches bound-but-idle sockets).
        #    Deliberately no SO_REUSEADDR: on Windows it would let this probe
        #    "succeed" on a port another app is already listening on.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port found")


def start_dashboard(port: int) -> subprocess.Popen | None:
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"

    # Health-wait in the background so a slow Streamlit boot never blocks the
    # replay; the dashboard simply catches up on its 3s refresh.
    def _wait() -> None:
        for _ in range(300):
            if proc.poll() is not None:
                print("[runner] streamlit exited early; dashboard not started")
                return
            try:
                with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as resp:
                    if resp.status == 200:
                        print(f"[runner] dashboard ready → {url}")
                        return
            except Exception:
                time.sleep(1)
        print(f"[runner] dashboard still starting; open {url} in a browser")

    threading.Thread(target=_wait, daemon=True).start()
    return proc


def print_stages(stages: dict) -> None:
    """Print the exact pipeline stage lines requested by the spec."""
    ex = stages.get("extract")
    if ex is None:
        return
    if stages.get("note"):
        print(f"[extract] ward=? symptom={ex['symptom']} ({stages['note']})")
        return
    print(f"[extract] ward={ex['ward']} symptom={ex['symptom']}")

    cluster_stage = stages.get("cluster")
    if cluster_stage and cluster_stage.get("size", 0) > 0:
        print(f"[cluster] ward {cluster_stage['ward']} cluster size={cluster_stage['size']}")

    anomaly_stage = stages.get("anomaly")
    if anomaly_stage:
        print(f"[anomaly] score={anomaly_stage['score']:.2f}")

    verdict = stages.get("alert") or {}
    if verdict.get("fired"):
        print(f"[ALERT] Ward {ex['ward']} crossed threshold. Triggering dashboard update.")


def run_sequence(rows: list[dict], delay: float, source: str) -> None:
    """Feed a list of reports through the pipeline with a slight delay each."""
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        if delay > 0:
            time.sleep(random.uniform(max(0.0, delay - 0.5), delay + 0.5))
        print(f"\n== report {i}/{total} ==")
        ts = datetime.now()
        stages = pipeline.process_report(row["text"], ts=ts, source=source)
        print_stages(stages)


def load_demo_rows() -> list[dict]:
    path = db.DATA_DIR / "demo_reports.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    # stage lines must appear live even when stdout is a pipe/log file
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="NEERWATCH runner")
    parser.add_argument("--replay", action="store_true",
                        help="replay the scripted Ward 9 outbreak from demo_reports.csv")
    parser.add_argument("--scenario", choices=["outbreak", "quiet"],
                        help="run a generated scenario instead of the random loop")
    parser.add_argument("--delay", type=float, default=2.5,
                        help="seconds between reports (2-3 simulates real chatter)")
    parser.add_argument("--port", type=int, default=None,
                        help="Streamlit port (default: first free port from 8501)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="skip launching Streamlit (terminal-only mode)")
    parser.add_argument("--keep-db", action="store_true",
                        help="keep existing reports/snapshots/alerts (default: fresh run)")
    args = parser.parse_args()

    # 1. initialise the SQLite database -------------------------------------
    db.init_db(reset=not args.keep_db)
    print(f"[db] initialised at {db.DB_PATH}")

    # train models up front so the first report isn't slow -------------------
    pipeline.ensure_models()

    # 2. launch the dashboard ------------------------------------------------
    streamlit_proc = None
    port = args.port or find_free_port()
    if not args.no_dashboard:
        streamlit_proc = start_dashboard(port)

    try:
        # 3-5. feed reports, print stages, persist to SQLite -----------------
        if args.replay:
            print("[runner] replaying demo_reports.csv (scripted Ward 9 outbreak)")
            run_sequence(load_demo_rows(), args.delay, source="demo")
            print("\n[runner] replay complete.")
        elif args.scenario:
            print(f"[runner] running '{args.scenario}' scenario")
            run_sequence(simulate.build_scenario(args.scenario), args.delay, source="scenario")
            print(f"\n[runner] scenario '{args.scenario}' complete.")
        else:
            print("[runner] live simulation loop (Ctrl+C to stop)")
            n = 0
            while True:
                n += 1
                text = simulate.next_random_report()
                print(f"\n== report {n} ==")
                stages = pipeline.process_report(text, ts=datetime.now(), source="sim")
                print_stages(stages)
                time.sleep(max(0.5, args.delay))
        if streamlit_proc is not None:
            print(f"[runner] dashboard still live on port {port}; Ctrl+C to exit.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[runner] shutting down.")
    finally:
        if streamlit_proc is not None and streamlit_proc.poll() is None:
            streamlit_proc.terminate()
            try:
                streamlit_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                streamlit_proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
