"""Measure a local installation with real processes (heuristic evaluator; no model).

    python scripts/perf/local_perf.py --avd backend/.venv/bin/avd --out benchmarks/reports/perf-local.json

Reports cold and warm startup, idle memory and CPU, database size, queue throughput, search and
recommendation latency (p50/p95 over repeated requests), export/import time and shutdown time.
These numbers describe the application around the evaluator. They say nothing about model
inference speed, which `avd bench-local` measures on hardware with a model.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


def rss_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def cpu_seconds(pid: int) -> float:
    parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")


def get(url: str, token: str | None = None) -> float:
    r = urllib.request.Request(url)
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    t0 = time.perf_counter()
    urllib.request.urlopen(r, timeout=30).read()
    return (time.perf_counter() - t0) * 1000


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avd", required=True)
    ap.add_argument("--port", type=int, default=8797)
    ap.add_argument("--out", default=None)
    ap.add_argument("--requests", type=int, default=50)
    a = ap.parse_args()
    env = {k: v for k, v in os.environ.items() if not k.startswith("AVD_")}
    # Latency is measured with repeated requests, which the default per-caller burst (60) would
    # throttle; the limiter stays active, with larger buckets.
    env.update(AVD_API_RATE_LIMIT_BURST="10000", AVD_API_RATE_LIMIT_IP_BURST="10000")
    work = Path(tempfile.mkdtemp(prefix="avd-perf-"))
    data, base = work / "data", f"http://127.0.0.1:{a.port}"
    res: dict[str, object] = {
        "host": {"os": platform.platform(), "cpus": os.cpu_count(), "python": platform.python_version()},
        "evaluator": "heuristic baseline (no model)",
        "note": "rate-limit buckets raised to 10000 for the latency loop; limiter still active",
    }

    def start(*extra: str) -> tuple[subprocess.Popen[bytes], float]:
        t0 = time.monotonic()
        p = subprocess.Popen(
            [a.avd, "start", "--data-dir", str(data), "--port", str(a.port), *extra],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        while True:
            try:
                urllib.request.urlopen(f"{base}/health", timeout=1)
                return p, round(time.monotonic() - t0, 2)
            except OSError:
                time.sleep(0.05)

    def idle() -> float:
        t0 = time.monotonic()
        while True:
            q = json.loads(urllib.request.urlopen(f"{base}/v1/status").read())["queue"]
            if not q.get("queued") and not q.get("running"):
                return round(time.monotonic() - t0, 2)
            time.sleep(0.1)

    def stop(p: subprocess.Popen[bytes]) -> float:
        t0 = time.monotonic()
        p.send_signal(signal.SIGINT)
        p.wait(timeout=60)
        return round(time.monotonic() - t0, 2)

    p, res["cold_start_s"] = start("--demo")  # fresh data dir: migrations + demo queueing
    try:
        _measure(a, env, data, base, work, res, p, start, stop, idle)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{exc.cmd} failed:\n{exc.stderr}") from exc
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            p.wait(timeout=60)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2) + "\n")


def _measure(a, env, data, base, work, res, p, start, stop, idle) -> None:  # type: ignore[no-untyped-def]
    res["demo_queue_drain_s_20_items"] = idle()
    cli_env = {**env, "AVD_DATABASE_URL": f"sqlite:///{data / 'avd.db'}"}
    t0 = time.monotonic()
    subprocess.run(
        [a.avd, "dataset", "import", "--offline", "--evaluate"],
        env=cli_env,
        capture_output=True,
        text=True,
        check=True,
    )
    res["pilot_import_s_29_items"] = round(time.monotonic() - t0, 2)
    res["pilot_queue_drain_s_29_items"] = idle()
    cpu0, t0 = cpu_seconds(p.pid), time.monotonic()
    time.sleep(10)
    res["idle"] = {
        "rss_mb": round(rss_mb(p.pid), 1),
        "cpu_percent_10s": round(100 * (cpu_seconds(p.pid) - cpu0) / (time.monotonic() - t0), 2),
    }
    token = json.loads(
        urllib.request.urlopen(urllib.request.Request(f"{base}/v1/users", method="POST")).read()
    )["token"]
    req = urllib.request.Request(
        f"{base}/v1/users/me/preferences",
        data=json.dumps({"interests": ["science", "programming"]}).encode(),
        method="PUT",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    urllib.request.urlopen(req).read()
    for name, url, tok in (
        ("search", f"{base}/v1/search?q=hash%20tables", None),
        ("recommendations", f"{base}/v1/recommendations", token),
        ("video_evaluation", f"{base}/v1/videos/syn_sci_001/evaluation", None),
    ):
        lat = [get(url, tok) for _ in range(a.requests)]
        res[f"{name}_ms"] = {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "n": len(lat)}
    res["rss_after_requests_mb"] = round(rss_mb(p.pid), 1)
    ex = work / "export.json"
    t0 = time.monotonic()
    out = subprocess.run(
        [a.avd, "evaluations", "export", "--out", str(ex)],
        env=cli_env,
        capture_output=True,
        text=True,
        check=True,
    )
    res["export"] = {
        "s": round(time.monotonic() - t0, 2),
        **json.loads(out.stdout),
        "kib": round(ex.stat().st_size / 1024, 1),
    }
    other = {**env, "AVD_DATABASE_URL": f"sqlite:///{work / 'other.db'}"}
    subprocess.run([a.avd, "migrate"], env=other, capture_output=True, check=True)
    t0 = time.monotonic()
    subprocess.run(
        [a.avd, "evaluations", "import", str(ex), "--origin", "perf"],
        env=other,
        capture_output=True,
        check=True,
    )
    res["import_s"] = round(time.monotonic() - t0, 2)
    res["shutdown_s"] = stop(p)
    res["database_mb"] = round((data / "avd.db").stat().st_size / 2**20, 2)
    p2, res["warm_start_s"] = start()
    res["warm_shutdown_s"] = stop(p2)


if __name__ == "__main__":
    main()
