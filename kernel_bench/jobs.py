# File: kernel_bench/jobs.py
import json
import os
import subprocess
import threading
import time
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = PROJECT_ROOT / "tensor_graphs_cpp" / "kernels"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "dirty_region_caches"
HISTORY_FILE = PROJECT_ROOT / "jobs_history.jsonl"
GENERATED_DIR = PROJECT_ROOT / "tensor_graphs_cpp" / "generated"

TIMEOUTS = {
    "build": 600,
    "test": 600,
    "infer": 1200,
    "bench": 1800,
}

jobs: dict = {}
worker_lock = threading.Lock()


def get_hw_info():
    info = "not available"
    hwinfo_path = PROJECT_ROOT / "hwinfo.txt"
    if hwinfo_path.exists():
        info = hwinfo_path.read_text()
    return info


def save_job_history(job):
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(job) + "\n")


def load_job_history():
    history = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))
    return history


def find_next_slot(backend: str) -> str:
    base = KERNELS_DIR / backend / "general" / "generated"
    os.makedirs(base, exist_ok=True)
    ext = ".cu" if backend == "cuda" else ".hpp"
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
            cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT
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


def get_uid_for_file(rel_path: str):
    header_path = GENERATED_DIR / "kernel_uids.gen.hpp"
    if not header_path.exists():
        return None
    const_name = rel_path.replace("/", "_").replace("\\", "_").replace(".", "_").upper()
    content = header_path.read_text()
    match = re.search(
        rf"constexpr uint64_t {const_name} = (0x[0-9a-fA-F]+ULL);", content
    )
    if match:
        return match.group(1).replace("ULL", "")
    return None


def analyze_total_time(target_model: str):
    records_path = BENCHMARKS_DIR / "records.jsonl"
    cache_path = CACHE_DIR / f"{target_model}-cpp.jsonl"

    if not records_path.exists() or not cache_path.exists():
        return 0.0, set()

    bench_map = {}
    with open(records_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                key = (
                    r["kernelUid"],
                    tuple(r["outputShapes"][0]),
                    tuple(r["outputStrides"][0]),
                )
                bench_map[key] = r["runTime"]

    total_time = 0.0
    extracted_uids = set()
    with open(cache_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") != "compiled_bucket":
                continue

            for inst in entry["graph"]["instructions"]:
                uid = inst["fullKernelId"]
                node = entry["graph"]["nodesMap"][str(inst["nodeId"])]
                extracted_uids.add(uid)
                key = (uid, tuple(node["shape"]), tuple(node["strides"]))
                total_time += bench_map.get(key, 0.0)

    return total_time, extracted_uids


def get_benchmark_scores(uid_str):
    scores = []
    records_path = BENCHMARKS_DIR / "records.jsonl"
    if not records_path.exists() or not uid_str:
        return scores
    target_uid = f"0x{int(uid_str, 16):x}"
    with open(records_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["kernelUid"] == target_uid:
                    scores.append(r["runTime"])
    return scores


def run_worker():
    while True:
        job_id = None
        with worker_lock:
            for jid, job in jobs.items():
                if job["status"] == "queued":
                    job_id = jid
                    job["status"] = "running"
                    break

        if not job_id:
            time.sleep(1)
            continue

        job = jobs[job_id]
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        opname = job["opname"]
        target_model = job["target_model"]

        try:
            # 1. Write kernel
            kernel_path = find_next_slot(job["backend"])
            with open(kernel_path, "w") as f:
                f.write(job["source"])
            job["kernel_file"] = kernel_path
            rel_path = (
                Path(kernel_path)
                .relative_to(PROJECT_ROOT / "tensor_graphs_cpp")
                .as_posix()
            )

            # Clear specific cache
            cache_file = CACHE_DIR / f"{target_model}-cpp.jsonl"
            if cache_file.exists():
                cache_file.unlink()

            # 2. Compile
            build_res = run_cmd(
                (
                    ["python", "build.py", "--cuda"]
                    if job["backend"] == "cuda"
                    else ["python", "build.py"]
                ),
                TIMEOUTS["build"],
            )
            job["steps"]["compile"] = build_res
            if build_res["exit_code"] != 0:
                raise Exception("Compilation failed")

            uid_str = get_uid_for_file("kernels/" + rel_path)
            job["assigned_uid"] = uid_str

            # 3. Test No Records
            test_no_rec_res = run_cmd(
                [
                    str(PROJECT_ROOT / "tensor_graphs_cpp" / "test"),
                    opname,
                    "--no-records",
                ],
                TIMEOUTS["test"],
            )
            job["steps"]["test_no_records"] = test_no_rec_res
            if (
                test_no_rec_res["exit_code"] != 0
                or "FAILED" in test_no_rec_res["stdout"]
            ):
                raise Exception("Test without records failed")

            # 4. Main to build calls.jsonl
            run_cmd(
                [str(PROJECT_ROOT / "tensor_graphs_cpp" / "main"), target_model],
                TIMEOUTS["infer"],
            )
            calls_path = BENCHMARKS_DIR / "calls.jsonl"
            matched = False
            if calls_path.exists() and uid_str:
                uid_int = int(uid_str, 16)
                with open(calls_path) as f:
                    for line in f:
                        if f'"kernelUid":"0x{uid_int:x}"' in line:
                            matched = True
                            break
            job["steps"]["matched"] = matched

            # 5. Test with Records
            test_rec_res = run_cmd(
                [str(PROJECT_ROOT / "tensor_graphs_cpp" / "test"), opname],
                TIMEOUTS["test"],
            )
            job["steps"]["test_records"] = test_rec_res
            if test_rec_res["exit_code"] != 0 or "FAILED" in test_rec_res["stdout"]:
                raise Exception("Test with records failed")

            # 6. Benchmark
            bench_res = run_cmd(
                [str(PROJECT_ROOT / "tensor_graphs_cpp" / "bench"), opname],
                TIMEOUTS["bench"],
            )
            job["steps"]["bench"] = bench_res

            # 7. Main again to construct cache with optimized routes
            run_cmd(
                [str(PROJECT_ROOT / "tensor_graphs_cpp" / "main"), target_model],
                TIMEOUTS["infer"],
            )

            total_time, extracted_uids = analyze_total_time(target_model)
            if uid_str:
                job["steps"]["extracted"] = (
                    uid_str in extracted_uids
                    or f"0x{int(uid_str, 16):x}" in extracted_uids
                )
            job["total_estimated_time_ms"] = total_time
            job["benchmark_scores"] = get_benchmark_scores(uid_str)

            job["status"] = "completed"

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)

        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_job_history(job)


def start_worker():
    t = threading.Thread(target=run_worker, daemon=True)
    t.start()
    return t


def create_job(source: str, opname: str, backend: str, target_model: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "queued",
        "backend": backend,
        "target_model": target_model,
        "opname": opname,
        "source": source,
        "assigned_uid": None,
        "started_at": None,
        "completed_at": None,
        "total_estimated_time_ms": None,
        "benchmark_scores": [],
        "steps": {
            "compile": None,
            "test_no_records": None,
            "matched": False,
            "test_records": None,
            "bench": None,
            "extracted": False,
        },
    }
    with worker_lock:
        jobs[job_id] = job
    return job_id
