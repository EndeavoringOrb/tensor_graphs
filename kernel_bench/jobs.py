# File: kernel_bench/jobs.py
import json
import os
import subprocess
import threading
import time
import uuid
import re
import struct
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = PROJECT_ROOT / "tensor_graphs_cpp" / "kernels"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "dirty_region_caches"
HISTORY_FILE = PROJECT_ROOT / "kernel_bench" / "jobs_history.jsonl"
GENERATED_DIR = PROJECT_ROOT / "tensor_graphs_cpp" / "generated"


class BinaryReader:
    def __init__(self, f):
        self.f = f

    def read_u8(self):
        buf = self.f.read(1)
        if not buf:
            return None
        return struct.unpack("<B", buf)[0]

    def read_u32(self):
        buf = self.f.read(4)
        if not buf:
            return None
        return struct.unpack("<I", buf)[0]

    def read_u64(self):
        buf = self.f.read(8)
        if not buf:
            return None
        return struct.unpack("<Q", buf)[0]

    def read_i32(self):
        buf = self.f.read(4)
        if not buf:
            return None
        return struct.unpack("<i", buf)[0]

    def read_float(self):
        buf = self.f.read(4)
        if not buf:
            return None
        return struct.unpack("<f", buf)[0]

    def read_string(self):
        size = self.read_u32()
        if size is None:
            return None
        if size == 0:
            return ""
        return self.f.read(size).decode("utf-8", errors="ignore")

    def read_vector(self, read_func):
        size = self.read_u32()
        if size is None:
            return None
        return [read_func() for _ in range(size)]

    def read_dtype(self):
        return self.read_u32()

    def read_backend(self):
        return self.read_u32()

    def read_map(self, read_key, read_val):
        size = self.read_u32()
        if size is None:
            return None
        return {read_key(): read_val() for _ in range(size)}

    def read_record(self):
        kernelUid = self.read_u64()
        if kernelUid is None:
            return None
        buildContextId = self.read_u64()
        hwTag = self.read_string()
        inputShapes = self.read_vector(lambda: self.read_vector(self.read_u32))
        outputShapes = self.read_vector(lambda: self.read_vector(self.read_u32))
        inputStrides = self.read_vector(lambda: self.read_vector(self.read_u64))
        outputStrides = self.read_vector(lambda: self.read_vector(self.read_u64))
        inputDTypes = self.read_vector(self.read_dtype)
        outputDTypes = self.read_vector(self.read_dtype)
        inputConstants = self.read_vector(lambda: self.f.read(self.read_u32()))
        backends = self.read_vector(self.read_backend)
        inputBackends = self.read_vector(lambda: self.read_vector(self.read_backend))
        runTime = self.read_float()
        return {
            "kernelUid": kernelUid,
            "outputShapes": outputShapes,
            "outputStrides": outputStrides,
            "runTime": runTime,
            "inputShapes": inputShapes,
            "inputStrides": inputStrides,
            "inputDTypes": inputDTypes,
            "outputDTypes": outputDTypes,
            "inputConstants": inputConstants,
            "backends": backends,
            "inputBackends": inputBackends,
            "hwTag": hwTag,
            "buildContextId": buildContextId,
        }

    def read_op_instruction(self):
        return {
            "nodeId": self.read_u32(),
            "logicalNodeId": self.read_u32(),
            "fullKernelId": self.read_u64(),
            "cachedKernelIds": self.read_vector(self.read_u64),
            "inputNodeIds": self.read_vector(self.read_u32),
            "inplaceInputIndex": self.read_i32(),
            "viewInputIndex": self.read_i32(),
            "backend": self.read_backend(),
            "outputStorageType": self.read_u32(),
        }

    def read_tensor_node(self):
        _id = self.read_u32()
        opType = self.read_u32()
        opName = self.read_string()
        dtype = self.read_dtype()
        parentIds = self.read_vector(self.read_u32)
        shape = self.read_vector(self.read_u32)
        strides = self.read_vector(self.read_u64)
        viewOffset = self.read_u64()
        backend = self.read_backend()
        storageType = self.read_u32()
        contentHash = self.read_string()
        return {
            "id": _id,
            "opType": opType,
            "opName": opName,
            "dtype": dtype,
            "parentIds": parentIds,
            "shape": shape,
            "strides": strides,
            "viewOffset": viewOffset,
            "backend": backend,
            "storageType": storageType,
            "contentHash": contentHash,
        }

    def read_compiled_graph(self):
        return {
            "instructions": self.read_vector(self.read_op_instruction),
            "refCounts": self.read_map(self.read_u32, self.read_u32),
            "nodesMap": {
                str(k): v
                for k, v in self.read_map(self.read_u32, self.read_tensor_node).items()
            },
            "nodeCosts": self.read_map(self.read_u32, self.read_float),
            "physicalToLogicalNodeMap": self.read_map(self.read_u32, self.read_u32),
            "constStaging": self.read_vector(
                lambda: (self.read_u32(), self.f.read(self.read_u32()))
            ),
        }


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
    records_path = BENCHMARKS_DIR / "records.bin"
    cache_path = CACHE_DIR / f"{target_model}-cpp.bin"

    if not records_path.exists() or not cache_path.exists():
        print("[WARN] records.bin or cache file missing, skipping analysis.")
        return 0.0, set()

    bench_map = {}
    with open(records_path, "rb") as f:
        br = BinaryReader(f)
        while True:
            r = br.read_record()
            if r is None:
                break
            key = (
                r["kernelUid"],
                tuple(r["outputShapes"][0]),
                tuple(r["outputStrides"][0]),
            )
            bench_map[key] = r["runTime"]

    total_time = 0.0
    extracted_uids = set()
    with open(cache_path, "rb") as f:
        br = BinaryReader(f)
        while True:
            t = br.read_u8()
            if t is None:
                break
            if t == 1:  # Compiled Bucket
                br.read_string()  # key
                graph = br.read_compiled_graph()
                for inst in graph["instructions"]:
                    uid = inst["fullKernelId"]
                    node = graph["nodesMap"][str(inst["nodeId"])]
                    extracted_uids.add(uid)
                    key = (uid, tuple(node["shape"]), tuple(node["strides"]))
                    total_time += bench_map.get(key, 0.0)
            elif t == 0:  # Metadata
                br.read_u32()
                br.read_u32()
                br.read_map(br.read_u32, br.read_backend)
            elif t == 2:  # Constants
                count = br.read_u32()
                for _ in range(count):
                    br.read_u32()
                    f.read(br.read_u32())
            else:
                break

    print(
        f"[INFO] Analysis complete. Total time: {total_time:.4f}ms, Unique UIDs: {len(extracted_uids)}"
    )
    return total_time, extracted_uids


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
            if r["kernelUid"] == target_uid:
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
            # 1. Write kernel
            kernel_path = find_next_slot(job["backend"])
            print(f"[JOB {job_id}] Writing kernel source to {kernel_path}")
            with open(kernel_path, "w") as f:
                f.write(job["source"])
            job["kernel_file"] = kernel_path
            rel_path = (
                Path(kernel_path)
                .relative_to(PROJECT_ROOT / "tensor_graphs_cpp")
                .as_posix()
            )

            # Clear specific cache
            cache_file = CACHE_DIR / f"{target_model}-cpp.bin"
            if cache_file.exists():
                print(f"[JOB {job_id}] Clearing existing cache: {cache_file}")
                cache_file.unlink()

            # 2. Compile
            python_path = ".venv/Scripts/python.exe"
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
                raise Exception("Compilation failed")

            uid_str = get_uid_for_file("kernels/" + rel_path)
            job["assigned_uid"] = uid_str

            # 3. Test No Records
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
                raise Exception("Test without records failed")

            # 4. Main to build calls.bin
            print(f"[JOB {job_id}] Step 3/7: Running inference to build calls.bin...")
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
                        if r["kernelUid"] == uid_int:
                            matched = True
                            break
            print(f"[JOB {job_id}] UID Match Result: {matched}")
            job["steps"]["matched"] = matched

            # 5. Test with Records
            print(f"[JOB {job_id}] Step 4/7: Testing with records...")
            test_rec_res = run_cmd(
                [str(PROJECT_ROOT / "tensor_graphs_cpp" / "test"), opname],
                TIMEOUTS["test"],
            )
            job["steps"]["test_records"] = test_rec_res
            if test_rec_res["exit_code"] != 0 or "FAILED" in test_rec_res["stdout"]:
                raise Exception("Test with records failed")

            # 6. Benchmark
            print(f"[JOB {job_id}] Step 5/7: Benchmarking kernel...")
            bench_res = run_cmd(
                [str(PROJECT_ROOT / "tensor_graphs_cpp" / "bench"), opname],
                TIMEOUTS["bench"],
            )
            job["steps"]["bench"] = bench_res

            # 7. Main again to construct cache with optimized routes
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
            total_time, extracted_uids = analyze_total_time(target_model)
            if uid_str:
                job["steps"]["extracted"] = (
                    uid_str in extracted_uids
                    or f"0x{int(uid_str, 16):x}" in extracted_uids
                )
            job["total_estimated_time_ms"] = total_time
            job["benchmark_scores"] = get_benchmark_scores(uid_str)

            job["status"] = "completed"
            print(f"[SUCCESS] Job {job_id} completed successfully.")

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            print(f"[ERROR] Job {job_id} failed: {e}")

            # Rename the faulty file to prevent it from breaking subsequent compilations
            if job.get("kernel_file") and os.path.exists(job["kernel_file"]):
                failed_path = job["kernel_file"] + ".failed"
                try:
                    os.rename(job["kernel_file"], failed_path)
                    job["kernel_file"] = failed_path
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
