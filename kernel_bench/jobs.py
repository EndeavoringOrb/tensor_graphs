import json
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = PROJECT_ROOT / "tensor_graphs_cpp" / "kernels"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "dirty_region_caches"

TIMEOUTS = {
    "build": 600,
    "test": 600,
    "infer": 1200,
    "bench": 1800,
    "analyze": 120,
}

jobs: dict = {}
worker_lock = threading.Lock()
active_job_id: str | None = None


def find_next_slot(backend: str) -> str:
    base = KERNELS_DIR / backend / "general" / "generated"
    ext = ".cu" if backend == "cuda" else ".hpp"
    os.makedirs(base, exist_ok=True)
    n = 0
    while True:
        path = base / f"{n:05d}{ext}"
        if not path.exists():
            return str(path)
        n += 1


def run_cmd(cmd: list[str], timeout: int) -> dict:
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": (time.time() - start) * 1000,
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": "TIMED OUT",
            "duration_ms": timeout * 1000,
        }


def parse_test_output(stdout: str) -> dict:
    result = {"python_ref": None, "kernel": None, "region_merge": None, "shape_prop": None}
    for line in stdout.splitlines():
        line = line.strip()
        if "Python Reference Tests Passed:" in line:
            parts = line.split(":")[-1].strip()
            if "/" in parts:
                p, t = parts.split("/")
                result["python_ref"] = {"passed": int(p), "total": int(t)}
        elif line.startswith("Tests Passed:") and "Non-Reference" not in stdout[:stdout.index(line) + 1]:
            parts = line.split(":")[-1].strip()
            if "/" in parts:
                p, t = parts.split("/")
                result["kernel"] = {"passed": int(p), "total": int(t)}
        elif "Region Merge Tests Passed:" in line:
            parts = line.split(":")[-1].strip()
            if "/" in parts:
                p, t = parts.split("/")
                result["region_merge"] = {"passed": int(p), "total": int(t)}
        elif "Shape Propagation Tests Passed:" in line:
            parts = line.split(":")[-1].strip()
            if "/" in parts:
                p, t = parts.split("/")
                result["shape_prop"] = {"passed": int(p), "total": int(t)}
    failures = 0
    for section in result.values():
        if section and section["passed"] < section["total"]:
            failures += 1
    result["failures"] = failures
    return result


def parse_analyze_output(stdout: str) -> dict:
    result = {
        "total_estimated_time_ms": None,
        "bucket_count": None,
        "top_chains": [],
        "op_type_breakdown": [],
        "missing_benchmarks": [],
        "raw_output": stdout,
    }

    for line in stdout.splitlines():
        if "Total Estimated Execution Time:" in line:
            ms_str = line.split(":")[-1].strip().replace(" ms", "")
            try:
                result["total_estimated_time_ms"] = float(ms_str)
            except ValueError:
                pass
        if "Analyzed" in line and "compiled buckets" in line:
            parts = line.split()
            for i, w in enumerate(parts):
                if w == "Analyzed" and i + 1 < len(parts):
                    try:
                        result["bucket_count"] = int(parts[i + 1])
                    except ValueError:
                        pass

    # Parse top chains table
    lines = stdout.splitlines()
    in_chains = False
    in_ops = False
    in_warnings = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("=" * 10):
            in_chains = True
            in_ops = False
            continue
        if stripped.startswith("-" * 10) and in_chains:
            continue
        if in_chains and "|" in stripped and not stripped.startswith("-"):
            parts = [p.strip() for p in stripped.split("|")]
            if len(parts) >= 4:
                chain_label = parts[0].strip()
                try:
                    count = int(parts[1].strip())
                except ValueError:
                    count = 0
                total_str = parts[2].strip().replace(" ms", "").replace("ms", "").strip()
                avg_str = parts[3].strip().replace(" ms", "").replace("ms", "").strip()
                try:
                    total_time = float(total_str) if total_str else None
                except ValueError:
                    total_time = None
                try:
                    avg_time = float(avg_str) if avg_str else None
                except ValueError:
                    avg_time = None
                if chain_label and not chain_label.startswith("Top") and not chain_label.startswith("Kernels"):
                    result["top_chains"].append({
                        "label": chain_label,
                        "count": count,
                        "total_time_ms": total_time,
                        "avg_time_ms": avg_time,
                    })
            continue
        if stripped.startswith("=" * 10) and len(result["top_chains"]):
            in_chains = False
        if stripped.startswith("Operation Type") or (stripped.startswith("-" * 10) and not in_chains and len(result["top_chains"])):
            in_ops = True
            continue
        if in_ops and "|" in stripped and not stripped.startswith("-"):
            parts = [p.strip() for p in stripped.split("|")]
            if len(parts) >= 2:
                op_name = parts[0].strip()
                time_str = parts[1].strip().replace(" ms", "").replace("ms", "").strip()
                try:
                    total_time = float(time_str) if time_str else None
                except ValueError:
                    total_time = None
                if op_name and op_name != "Operation Type":
                    result["op_type_breakdown"].append({
                        "op_name": op_name,
                        "total_time_ms": total_time,
                    })
            continue
        if "[Warning]" in stripped:
            in_warnings = True
        if in_warnings and stripped.startswith("Run bench.cpp"):
            break
        if in_warnings and "(" in stripped and "UID:" in stripped:
            result["missing_benchmarks"].append(stripped)

    return result


def load_benchmark_records() -> list:
    path = BENCHMARKS_DIR / "records.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_worker():
    """Background worker that processes jobs sequentially."""
    while True:
        job_id = None
        with worker_lock:
            for jid, job in jobs.items():
                if job["status"] == "queued":
                    job_id = jid
                    job["status"] = "running"
                    active_job_id = jid
                    break

        if not job_id:
            time.sleep(1)
            continue

        job = jobs[job_id]
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        steps = job["steps"]

        try:
            # Step 1: Write kernel file
            steps[0]["status"] = "running"
            kernel_path = find_next_slot(job["backend"])
            with open(kernel_path, "w") as f:
                f.write(job["source"])
            job["kernel_file"] = kernel_path
            steps[0]["status"] = "done"

            # Step 2: Clear dirty region caches
            steps[1]["status"] = "running"
            for f in CACHE_DIR.iterdir():
                if f.is_file():
                    f.unlink()
            steps[1]["status"] = "done"

            # Step 3: Build
            steps[2]["status"] = "running"
            res = run_cmd(["python", "build.py", "--cuda"], TIMEOUTS["build"])
            steps[2].update(res)
            steps[2]["status"] = "done" if res["exit_code"] == 0 else "failed"
            if res["exit_code"] != 0:
                job["status"] = "failed"
                job["error"] = f"Build failed with exit code {res['exit_code']}"
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
                with worker_lock:
                    active_job_id = None
                break

            # Step 4: Test
            steps[3]["status"] = "running"
            res = run_cmd(["./tensor_graphs_cpp/test"], TIMEOUTS["test"])
            steps[3].update(res)
            test_results = parse_test_output(res["stdout"])
            job["test_results"] = test_results
            if res["exit_code"] != 0 or test_results.get("failures", 0) > 0:
                steps[3]["status"] = "failed"
                job["status"] = "failed"
                job["error"] = "Tests failed"
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
                with worker_lock:
                    active_job_id = None
                break
            steps[3]["status"] = "done"

            # Steps 5-6: Inference + benchmark loop
            max_iterations = 20
            bench_done = False
            bench_combined = ""

            for i in range(max_iterations):
                # Infer
                steps[4]["status"] = "running"
                res = run_cmd(["./tensor_graphs_cpp/main"], TIMEOUTS["infer"])
                steps[4].update({
                    "stdout": res["stdout"],
                    "stderr": res["stderr"],
                    "exit_code": res["exit_code"],
                    "duration_ms": res["duration_ms"],
                })
                if res["exit_code"] != 0:
                    steps[4]["status"] = "failed"
                    job["status"] = "failed"
                    job["error"] = f"Inference failed (iteration {i + 1})"
                    job["completed_at"] = datetime.now(timezone.utc).isoformat()
                    with worker_lock:
                        active_job_id = None
                    break
                steps[4]["status"] = "done"

                # Bench
                steps[5]["status"] = "running"
                res = run_cmd(["./tensor_graphs_cpp/bench"], TIMEOUTS["bench"])
                bench_combined += res["stdout"] + "\n"
                steps[5].update({
                    "stdout": res["stdout"],
                    "stderr": res["stderr"],
                    "exit_code": res["exit_code"],
                    "duration_ms": res["duration_ms"],
                })
                job["benchmark_iterations"] = i + 1

                if "All calls already benchmarked or no new kernels to test" in res["stdout"]:
                    steps[5]["status"] = "done"
                    bench_done = True
                    break
                steps[5]["status"] = "done"

            if not bench_done:
                if job["status"] != "failed":
                    job["status"] = "failed"
                    job["error"] = f"Benchmark did not converge after {max_iterations} iterations"
                    job["completed_at"] = datetime.now(timezone.utc).isoformat()
                    with worker_lock:
                        active_job_id = None
                    break

            # Step 7: Analyze
            steps[6]["status"] = "running"
            res = run_cmd(
                ["python", "analyze_performance.py",
                 "--graph", str(CACHE_DIR / "gemma-3-270m-cpp.jsonl"),
                 "--records", str(BENCHMARKS_DIR / "records.jsonl")],
                TIMEOUTS["analyze"],
            )
            steps[6].update(res)
            analysis = parse_analyze_output(res["stdout"])
            job["analysis_output"] = analysis
            steps[6]["status"] = "done"

            job["status"] = "completed"

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)

        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        with worker_lock:
            active_job_id = None


def start_worker():
    t = threading.Thread(target=run_worker, daemon=True)
    t.start()
    return t


def create_job(source: str, backend: str = "cpu") -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "queued",
        "backend": backend,
        "source": source,
        "kernel_file": None,
        "started_at": None,
        "completed_at": None,
        "benchmark_iterations": 0,
        "test_results": None,
        "analysis_output": None,
        "error": None,
        "steps": [
            {"name": "write_kernel", "status": "pending"},
            {"name": "clear_caches", "status": "pending"},
            {"name": "build", "status": "pending"},
            {"name": "test", "status": "pending"},
            {"name": "infer", "status": "pending"},
            {"name": "benchmark_loop", "status": "pending"},
            {"name": "analyze", "status": "pending"},
        ],
    }
    with worker_lock:
        jobs[job_id] = job
    return job_id
