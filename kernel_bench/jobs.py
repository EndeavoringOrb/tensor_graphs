# kernel_bench/jobs.py
import json
import os
import subprocess
import threading
import time
import uuid
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = PROJECT_ROOT / "tensor_graphs_cpp" / "kernels"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "dirty_region_caches"
HISTORY_FILE = PROJECT_ROOT / "kernel_bench" / "jobs_history.jsonl"
REPORTS_FILE = PROJECT_ROOT / "kernel_bench" / "reports.jsonl"
GENERATED_DIR = PROJECT_ROOT / "tensor_graphs_cpp" / "generated"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.binary import (
    BinaryReader,
    load_cache_file,
)

TIMEOUTS = {
    "build": 600,
    "test": 600,
    "infer": 1200,
    "bench": 1800,
}

jobs: dict = {}
worker_lock = threading.Lock()
report_lock = threading.Lock()


def save_report(report_data):
    with report_lock:
        with open(REPORTS_FILE, "a") as f:
            f.write(json.dumps(report_data) + "\n")


def load_reports():
    reports = []
    if REPORTS_FILE.exists():
        with report_lock:
            with open(REPORTS_FILE, "r") as f:
                for line in f:
                    if line.strip():
                        reports.append(json.loads(line))
    return list(reversed(reports))


def get_hw_info():
    info = "not available"
    hwinfo_path = PROJECT_ROOT / "hwinfo.txt"
    if hwinfo_path.exists():
        info = hwinfo_path.read_text()
    return info


def save_job_history(job):
    print(f"[INFO] Saving job {job['job_id']} history to {HISTORY_FILE}")
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
        failed_path = base / f"{n:05d}{ext}.failed"
        if not path.exists() and not failed_path.exists():
            return str(path)
        n += 1


def run_cmd(cmd: list[str], timeout: int) -> dict:
    start = time.time()
    cmd_str = " ".join(cmd)
    print(f"[EXEC] Running: {cmd_str} (Timeout: {timeout}s)")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT
        )
        duration = (time.time() - start) * 1000
        print(f"[EXEC] Finished in {duration:.2f}ms with exit code {result.returncode}")
        if result.returncode != 0:
            print(f"[WARN] Command stderr: {result.stderr.strip()[:200]}...")

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": duration,
        }
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Command TIMED OUT after {timeout}s: {cmd_str}")
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
        uid = match.group(1).replace("ULL", "")
        print(f"[INFO] Resolved UID {uid} for {rel_path}")
        return uid
    return None


def analyze_total_time(target_model: str):
    print(f"[INFO] Analyzing total time for model: {target_model}")

    if target_model == "gemma-3-270m":
        cache_paths = ["gemma-3-270m-cpp.bin"]
    elif target_model == "flux-klein-4b":
        cache_paths = ["flux-text.bin", "flux-trans.bin", "flux-vae.bin"]
    else:
        cache_paths = [f"{target_model}-cpp.bin"]

    cache_paths = [CACHE_DIR / cache_path for cache_path in cache_paths]

    for check_path in cache_paths:
        if not check_path.exists():
            message = f"[WARN] {check_path} file missing, skipping analysis."
            print(message)
            return 0.0, set(), message

    total_time = 0.0
    extracted_uids = set()

    for cache_path in cache_paths:
        cache_entries = load_cache_file(cache_path)
        for entry in cache_entries:
            if entry.get("type") == "compiled_bucket":
                graph = entry["graph"]
                node_costs = graph.get("nodeCosts", {})

                for inst in graph["instructions"]:
                    uid = inst["fullKernelId"]
                    node_id = inst["nodeId"]
                    extracted_uids.add(uid)

                    runtime = node_costs.get(node_id, 0.0)
                    if runtime == float("inf"):
                        runtime = 0.0
                    total_time += runtime

    message = f"[INFO] Analysis complete. Total time: {total_time:.4f}ms, Unique UIDs: {len(extracted_uids)}"
    print(message)
    return total_time, extracted_uids, message


def get_benchmark_scores(uid_str):
    scores = []
    records_path = BENCHMARKS_DIR / "records.bin"
    if not records_path.exists() or not uid_str:
        return scores
    target_uid = int(uid_str, 16)
    with open(records_path, "rb") as f:
        br = BinaryReader(f)
        while True:
            r = br.read_record()
            if r is None:
                break
            if r["kernelId"] == target_uid:
                scores.append(r["runTime"])
    return scores


def run_worker():
    print("[SYSTEM] Worker thread started and waiting for jobs...")
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

        print(f"\n[JOB {job_id}] Processing op: {opname} for model: {target_model}")

        try:
            kernel_path = find_next_slot(job["backend"])
            print(f"[JOB {job_id}] Writing kernel source to {kernel_path}")
            with open(kernel_path, "w") as f:
                f.write(job["source"])
            job["kernel_file"] = kernel_path
            job["agent_file_path"] = (
                Path(kernel_path).relative_to(KERNELS_DIR).as_posix()
            )

            del job["source"]

            rel_path = (
                Path(kernel_path)
                .relative_to(PROJECT_ROOT / "tensor_graphs_cpp")
                .as_posix()
            )

            cache_file = CACHE_DIR / f"{target_model}-cpp.bin"
            if cache_file.exists():
                print(f"[JOB {job_id}] Clearing existing cache: {cache_file}")
                cache_file.unlink()

            python_path = ".venv/Scripts/python.exe" if os.name == "nt" else "python"
            print(f"[JOB {job_id}] Step 1/7: Compiling...")
            build_res = run_cmd(
                (
                    [python_path, "build.py", "--cuda"]
                    if job["backend"] == "cuda"
                    else [python_path, "build.py"]
                ),
                TIMEOUTS["build"],
            )
            job["steps"]["compile"] = build_res
            if build_res["exit_code"] != 0:
                raise Exception(
                    f"Compilation failed.\nSTDOUT:\n{build_res['stdout']}\nSTDERR:\n{build_res['stderr']}"
                )

            uid_str = get_uid_for_file("kernels/" + rel_path)
            job["assigned_uid"] = uid_str

            print(f"[JOB {job_id}] Step 2/7: Testing without records...")
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
                raise Exception(
                    f"Test without records failed.\nSTDOUT:\n{test_no_rec_res['stdout']}\nSTDERR:\n{test_no_rec_res['stderr']}"
                )

            print(f"[JOB {job_id}] Step 3/7: Running inference to build calls.bin...")
            run_cmd(
                [
                    str(PROJECT_ROOT / "tensor_graphs_cpp" / "write_ref_tensors"),
                    target_model,
                ],
                TIMEOUTS["infer"],
            )
            run_cmd(
                [
                    str(PROJECT_ROOT / "tensor_graphs_cpp" / "main"),
                    target_model,
                    "--only-plan",
                ],
                TIMEOUTS["infer"],
            )
            calls_path = BENCHMARKS_DIR / "calls.bin"
            matched = False
            if calls_path.exists() and uid_str:
                uid_int = int(uid_str, 16)
                with open(calls_path, "rb") as f:
                    br = BinaryReader(f)
                    while True:
                        r = br.read_record()
                        if r is None:
                            break
                        if r["kernelId"] == uid_int:
                            matched = True
                            break
            print(f"[JOB {job_id}] UID Match Result: {matched}")
            job["steps"]["matched"] = matched
            if not matched:
                raise Exception(
                    "Kernel UID not matched in inference plan (calls.bin). The kernel might not be utilized or the operation signature/shapes are incorrect."
                )

            print(f"[JOB {job_id}] Step 4/7: Testing with records...")
            test_rec_res = run_cmd(
                [str(PROJECT_ROOT / "tensor_graphs_cpp" / "test"), opname],
                TIMEOUTS["test"],
            )
            job["steps"]["test_records"] = test_rec_res
            if test_rec_res["exit_code"] != 0 or "FAILED" in test_rec_res["stdout"]:
                raise Exception(
                    f"Test with records failed.\nSTDOUT:\n{test_rec_res['stdout']}\nSTDERR:\n{test_rec_res['stderr']}"
                )

            print(f"[JOB {job_id}] Step 5/7: Benchmarking kernel...")
            bench_res = run_cmd(
                [str(PROJECT_ROOT / "tensor_graphs_cpp" / "bench"), opname],
                TIMEOUTS["bench"],
            )
            job["steps"]["bench"] = bench_res
            if bench_res["exit_code"] != 0:
                raise Exception(
                    f"Benchmark failed.\nSTDOUT:\n{bench_res['stdout']}\nSTDERR:\n{bench_res['stderr']}"
                )

            print(
                f"[JOB {job_id}] Step 6/7: Regenerating cache with optimized routes..."
            )
            run_cmd(
                [
                    str(PROJECT_ROOT / "tensor_graphs_cpp" / "main"),
                    target_model,
                    "--only-plan",
                ],
                TIMEOUTS["infer"],
            )

            print(f"[JOB {job_id}] Step 7/7: Final time analysis...")
            total_time, extracted_uids, message = analyze_total_time(target_model)

            is_extracted = False
            if uid_str:
                is_extracted = (
                    uid_str in extracted_uids
                    or f"0x{int(uid_str, 16):x}" in extracted_uids
                )

            job["steps"]["extracted"] = is_extracted

            if not is_extracted:
                raise Exception(
                    "Kernel was not extracted in the final optimized graph. It may be functionally valid but benchmarks slower than existing alternatives or unoptimized defaults."
                )

            job["total_estimated_time_ms"] = total_time
            job["benchmark_scores"] = get_benchmark_scores(uid_str)

            job["status"] = "completed"
            print(f"[SUCCESS] Job {job_id} completed successfully.")

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            print(f"[ERROR] Job {job_id} failed: {e}")

            if job.get("kernel_file") and os.path.exists(job["kernel_file"]):
                failed_path = job["kernel_file"] + ".failed"
                try:
                    os.rename(job["kernel_file"], failed_path)
                    job["kernel_file"] = failed_path
                    if "agent_file_path" in job:
                        job["agent_file_path"] += ".failed"
                    print(f"[INFO] Renamed failed kernel to {failed_path}")
                except Exception as rename_err:
                    print(f"[WARN] Failed to rename {job['kernel_file']}: {rename_err}")

        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_job_history(job)


def start_worker():
    t = threading.Thread(target=run_worker, daemon=True)
    t.start()
    return t


def create_job(source: str, opname: str, backend: str, target_model: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    print(f"[SYSTEM] Creating job {job_id} for {opname} ({backend})")
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
