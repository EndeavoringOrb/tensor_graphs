# File: kernel_bench/jobs.py
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = PROJECT_ROOT / "tensor_graphs_cpp" / "kernels"
CORE_DIR = PROJECT_ROOT / "tensor_graphs_cpp" / "core"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
CACHE_DIR = PROJECT_ROOT / "dirty_region_caches"
VERSIONS_DIR = PROJECT_ROOT / "versions"
HISTORY_FILE = PROJECT_ROOT / "kernel_bench" / "jobs_history.jsonl"
REPORTS_FILE = PROJECT_ROOT / "kernel_bench" / "reports.jsonl"
SUGGESTIONS_FILE = PROJECT_ROOT / "kernel_bench" / "suggestions.jsonl"
GENERATED_DIR = PROJECT_ROOT / "tensor_graphs_cpp" / "generated"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.binary import BinaryReader, load_cache_file
from utils.common import format_op_name, load_uids_from_cpp, natural_sort_key

LLAMA_CPP_TARGETS = {
    "gemma-3-270m": {
        "pp512": {"tps": 3439.99, "stdev": 127.06},
        "tg128": {"tps": 69.16, "stdev": 1.28},
    }
}

TIMEOUTS = {
    "build": 600,
    "test": 120,
    "bench_model": 900,
    "bench": 180,
    "analysis": 120,
}

STATIC_MIN_COMPILE_TIME = 90.0


def extractRegisteredKernelNames(source_text: str) -> list:
    if not source_text:
        return []
    matches = re.findall(r'REGISTER_KERNEL(?:_VIEW)?\s*\(\s*["\']([^"\']+)["\']', source_text)
    return list(dict.fromkeys(matches))


jobs: dict = {}
worker_lock = threading.Lock()
report_lock = threading.Lock()
suggestion_lock = threading.Lock()


def getPythonExe() -> str:
    if os.name == "nt":
        venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def getHwInfo() -> str:
    info = "not available"
    hwinfo_path = PROJECT_ROOT / "hwinfo.txt"
    if hwinfo_path.exists():
        info = hwinfo_path.read_text()
    return info


def saveReport(report_data: dict) -> None:
    with report_lock, open(REPORTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(report_data) + "\n")

    desc = report_data.get("issue_description", "")
    title = report_data.get("title") or (f"Issue: {desc[:50]}..." if len(desc) > 50 else (desc or "Issue Report"))
    sug = {
        "title": title,
        "category": "report",
        "suggested_endpoint": report_data.get("suggested_endpoint", ""),
        "description": desc,
        "proposed_changes": report_data.get("proposed_changes", ""),
        "agent_id": report_data.get("agent_id", "reporter"),
        "priority": report_data.get("priority", "high"),
        "timestamp": report_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "status": "open",
    }
    saveSuggestion(sug)


def loadReports() -> list:
    reports = []
    if REPORTS_FILE.exists():
        with report_lock, open(REPORTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    reports.append(json.loads(line))
    return list(reversed(reports))


def saveSuggestion(suggestion_data: dict) -> str:
    suggestion_id = uuid.uuid4().hex[:12]
    suggestion_data["suggestion_id"] = suggestion_id
    if "timestamp" not in suggestion_data:
        suggestion_data["timestamp"] = datetime.now(timezone.utc).isoformat()
    if "status" not in suggestion_data:
        suggestion_data["status"] = "open"
    if "priority" not in suggestion_data:
        suggestion_data["priority"] = "medium"
    with suggestion_lock, open(SUGGESTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(suggestion_data) + "\n")
    return suggestion_id


def loadSuggestions() -> list:
    suggestions = []
    if SUGGESTIONS_FILE.exists():
        with suggestion_lock, open(SUGGESTIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "priority" not in item:
                        item["priority"] = "medium"
                    suggestions.append(item)
    return list(reversed(suggestions))


def updateSuggestionStatus(suggestion_id: str, new_status: str) -> bool:
    updated = False
    if not SUGGESTIONS_FILE.exists():
        return False
    with suggestion_lock:
        lines = []
        with open(SUGGESTIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("suggestion_id") == suggestion_id:
                        item["status"] = new_status
                        item["updated_at"] = datetime.now(timezone.utc).isoformat()
                        updated = True
                    lines.append(json.dumps(item))
        if updated:
            with open(SUGGESTIONS_FILE, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
    return updated


def saveJobHistory(job: dict) -> None:
    job_record = dict(job)
    if "source" in job_record:
        del job_record["source"]
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(job_record) + "\n")


def loadJobHistory() -> list:
    history = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))
    return history


def getActiveJobs() -> list:
    with worker_lock:
        active = [
            dict(j) for j in jobs.values()
            if j.get("status") in ("queued", "running")
        ]
    for j in active:
        if "source" in j:
            del j["source"]
    return active


def getAllJobs() -> list:
    active = getActiveJobs()
    history = loadJobHistory()
    seen = {j.get("job_id") for j in active}
    combined = list(active)
    for j in reversed(history):
        if j.get("job_id") not in seen:
            combined.append(j)
            seen.add(j.get("job_id"))
    return combined


def getNextVersionId() -> int:
    VERSIONS_DIR.mkdir(exist_ok=True)
    existing = []
    for item in VERSIONS_DIR.iterdir():
        if item.is_dir() and item.name.isdigit():
            existing.append(int(item.name))
    return max(existing, default=-1) + 1


def parseBenchModelMetrics(content: str) -> dict:
    metrics = {}
    pattern = re.compile(
        r"\|\s*([^\s|]+)\s*\|\s*([\d.]+)\s*(?:\+\-|\+\/\-|\±)\s*([\d.]+)\s*\|"
    )
    for test_name, mean_str, stdev_str in pattern.findall(content):
        test_key = test_name.strip()
        metrics[test_key] = {
            "tps": float(mean_str),
            "stdev": float(stdev_str),
        }
    return metrics


def getFallbackCause(name: str, path: str = "") -> str:
    name_upper = name.upper()
    if "COPY_TO" in name_upper:
        return "Host <-> Device memory transfer"
    if "CAST" in name_upper:
        return "Missing GPU type cast kernel"
    if "NEGATE" in name_upper:
        return "Missing GPU elementwise negate kernel"
    if "ARANGE" in name_upper:
        return "Missing GPU range generation kernel"
    if "CONCAT" in name_upper:
        return "Missing GPU concatenation kernel"
    if "POWER" in name_upper:
        return "Missing GPU power/exponentiation kernel"
    if any(k in name_upper for k in ["MUL", "DIVIDE", "ADD", "SUB", "EQ", "LT"]):
        return "Missing GPU strided/broadcast kernel (requiresContiguous mismatch)"
    return "Missing GPU implementation (CPU reference fallback)"


def parseCacheAnalysis(content: str) -> dict:
    result = {
        "total_estimated_time_ms": 0.0,
        "top_ops": [],
        "kernels": [],
        "all_kernels": [],
        "cpu_fallbacks": [],
        "summary": {},
    }
    time_match = re.search(r"Total Estimated Execution Time:\s*([\d.]+)\s*ms", content)
    total_time = float(time_match.group(1)) if time_match else 0.0
    result["total_estimated_time_ms"] = total_time

    count_map = {}
    top_kernel_pattern = re.compile(
        r"^([a-zA-Z0-9_]+(?:\s*\[[^\]]+\])?)(?:\([^)]*\))?(?:\s*\[[^\]]*\])?\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*ms",
        re.MULTILINE,
    )
    for raw_name, cnt, _ in top_kernel_pattern.findall(content):
        clean_raw = raw_name.strip()
        count_map[clean_raw] = int(cnt)

    op_matches = re.findall(
        r"^([a-zA-Z0-9_]+(?:\s*\[[^\]]+\])?)\s*\|\s*([\d.]+)\s*ms",
        content,
        re.MULTILINE,
    )

    all_kernels = []
    fallbacks = []

    for full_op_str, time_str in op_matches:
        t_val = float(time_str)
        full_op_str = full_op_str.strip()

        path_match = re.search(r"\[([^\]]+)\]", full_op_str)
        kernel_path = path_match.group(1).strip() if path_match else ""
        kernel_name = re.sub(r"\s*\[[^\]]+\]", "", full_op_str).strip()

        is_fallback = "REF_" in kernel_name or "reference" in kernel_path.lower()
        if is_fallback:
            device = "CPU (Reference)"
            category = "fallback"
        elif "cuda" in kernel_path.lower() or "cublas" in kernel_path.lower() or kernel_path.endswith(".cu") or "CUDA" in kernel_name or "CuBLAS" in kernel_name:
            device = "CUDA"
            category = "cuda"
        else:
            device = "CPU"
            category = "cpu"

        cause = getFallbackCause(kernel_name, kernel_path) if is_fallback else ""
        pct = round((t_val / total_time * 100), 2) if total_time > 0 else 0.0
        count = count_map.get(full_op_str, count_map.get(kernel_name, 0))

        k_dict = {
            "op": kernel_name,
            "name": kernel_name,
            "path": kernel_path,
            "kernel_id": "",
            "time_ms": t_val,
            "count": count,
            "percentage": pct,
            "is_fallback": is_fallback,
            "device": device,
            "category": category,
            "cause": cause,
        }
        all_kernels.append(k_dict)
        if is_fallback:
            fallbacks.append(k_dict)

    result["top_ops"] = all_kernels[:20]
    result["kernels"] = all_kernels
    result["all_kernels"] = all_kernels
    result["cpu_fallbacks"] = fallbacks

    fallback_time = sum(f["time_ms"] for f in fallbacks)
    result["summary"] = {
        "total_time_ms": total_time,
        "total_kernels": len(all_kernels),
        "fallback_count": len(fallbacks),
        "fallback_time_ms": round(fallback_time, 4),
        "fallback_percentage": round(fallback_time / total_time * 100, 2) if total_time > 0 else 0.0,
    }

    return result


def getAllVersions() -> list:
    VERSIONS_DIR.mkdir(exist_ok=True)
    versions_list = []
    subdirs = sorted(
        [d for d in VERSIONS_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    for v_dir in subdirs:
        v_id = int(v_dir.name)
        idea_file = v_dir / "idea.md"
        bench_file = v_dir / "bench_model.log"
        cache_file = v_dir / "cache_analysis.log"
        readme_file = v_dir / "README.md"
        build_file = v_dir / "build.log"

        idea_text = idea_file.read_text(encoding="utf-8").strip() if idea_file.exists() else ""
        metrics = {}
        if bench_file.exists():
            metrics = parseBenchModelMetrics(bench_file.read_text(encoding="utf-8", errors="ignore"))

        cache_summary = {}
        if cache_file.exists():
            cache_summary = parseCacheAnalysis(cache_file.read_text(encoding="utf-8", errors="ignore"))

        readme_text = readme_file.read_text(encoding="utf-8").strip() if readme_file.exists() else ""

        has_build = build_file.exists()
        has_bench = bench_file.exists()
        has_cache = cache_file.exists()

        status = "completed" if (has_build and has_bench and has_cache) else "partial"

        pp_metric = metrics.get("pp512", {})
        tg_metric = metrics.get("tg128", {})
        target = LLAMA_CPP_TARGETS.get("gemma-3-270m", {})

        target_beaten = False
        if pp_metric and tg_metric and target:
            target_beaten = (
                pp_metric.get("tps", 0) > target["pp512"]["tps"]
                and tg_metric.get("tps", 0) > target["tg128"]["tps"]
            )

        versions_list.append({
            "version": v_id,
            "path": str(v_dir.relative_to(PROJECT_ROOT)),
            "idea": idea_text,
            "status": status,
            "metrics": metrics,
            "total_estimated_time_ms": cache_summary.get("total_estimated_time_ms", 0.0),
            "top_ops": cache_summary.get("top_ops", [])[:5],
            "cpu_fallbacks": cache_summary.get("cpu_fallbacks", [])[:3],
            "target_beaten": target_beaten,
            "has_readme": readme_file.exists(),
        })
    return versions_list


def getVersionDetails(version_id: int) -> dict:
    v_dir = VERSIONS_DIR / str(version_id)
    if not v_dir.exists():
        return {"error": f"Version {version_id} not found"}

    files = {}
    for filename in ["idea.md", "build.log", "test.log", "bench_model.log", "cache_analysis.log", "README.md"]:
        f_path = v_dir / filename
        if f_path.exists():
            try:
                files[filename] = f_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                files[filename] = f"[Error reading file: {e}]"
        else:
            files[filename] = None

    metrics = parseBenchModelMetrics(files.get("bench_model.log") or "")
    cache_analysis = parseCacheAnalysis(files.get("cache_analysis.log") or "")

    return {
        "version": version_id,
        "files": files,
        "metrics": metrics,
        "cache_analysis": cache_analysis,
    }


def findNextSlot(backend: str, category: str = "generated") -> str:
    base = KERNELS_DIR / backend / "general" / category
    base.mkdir(parents=True, exist_ok=True)
    ext = ".cu" if backend in ("cuda", "cublas") else ".hpp"
    n = 0
    while True:
        path = base / f"{n:05d}{ext}"
        failed_path = base / f"{n:05d}{ext}.failed"
        if not path.exists() and not failed_path.exists():
            return str(path)
        n += 1


def runCmd(cmd: list[str], timeout: int, log_path: Path = None, append_log: bool = False) -> dict:
    start_time = time.time()
    cmd_str = " ".join(cmd)
    print(f"[EXEC] Running: {cmd_str} (Timeout: {timeout}s)")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=PROJECT_ROOT
        )
        duration_ms = (time.time() - start_time) * 1000
        output_text = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append_log else "w"
            with open(log_path, mode, encoding="utf-8") as f:
                f.write(output_text)

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": duration_ms,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        duration_ms = timeout * 1000
        out = (e.stdout.decode("utf-8", errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or ""))
        err = (e.stderr.decode("utf-8", errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append_log else "w"
            with open(log_path, mode, encoding="utf-8") as f:
                f.write(f"{out}\n{err}\n[TIMED OUT after {timeout}s]\n")
        return {
            "exit_code": -1,
            "stdout": out,
            "stderr": f"TIMED OUT after {timeout}s: {err}",
            "duration_ms": duration_ms,
            "timed_out": True,
        }


def generateVersionReadme(v_id: int, idea: str, metrics: dict, cache_analysis: dict, target_model: str) -> str:
    target = LLAMA_CPP_TARGETS.get(target_model, {}).get("pp512", {})
    tg_target = LLAMA_CPP_TARGETS.get(target_model, {}).get("tg128", {})
    pp_curr = metrics.get("pp512", {"tps": 0.0, "stdev": 0.0})
    tg_curr = metrics.get("tg128", {"tps": 0.0, "stdev": 0.0})

    top_ops_md = ""
    for op in cache_analysis.get("top_ops", [])[:10]:
        top_ops_md += f"- `{op['op']}`: {op['time_ms']:.2f} ms\n"

    cpu_fallbacks_md = ""
    for op in cache_analysis.get("cpu_fallbacks", [])[:5]:
        cpu_fallbacks_md += f"- `{op['op']}`: {op['time_ms']:.2f} ms\n"
    if not cpu_fallbacks_md:
        cpu_fallbacks_md = "None detected (all operations accelerated on GPU).\n"

    return f"""# Version {v_id}: {idea}

## Idea
{idea}

## Results
| Model | Test | Tokens/sec (Achieved) | Tokens/sec (Target Llama.cpp) | Status |
|---|---|---|---|---|
| {target_model} | pp512 | {pp_curr.get('tps', 0.0):.2f} ± {pp_curr.get('stdev', 0.0):.2f} | {target.get('tps', 0.0):.2f} ± {target.get('stdev', 0.0):.2f} | {'BEATEN' if pp_curr.get('tps', 0) > target.get('tps', 0) else 'BELOW'} |
| {target_model} | tg128 | {tg_curr.get('tps', 0.0):.2f} ± {tg_curr.get('stdev', 0.0):.2f} | {tg_target.get('tps', 0.0):.2f} ± {tg_target.get('stdev', 0.0):.2f} | {'BEATEN' if tg_curr.get('tps', 0) > tg_target.get('tps', 0) else 'BELOW'} |

## Cache & Performance Analysis
- **Total Estimated Execution Time**: {cache_analysis.get('total_estimated_time_ms', 0.0):.2f} ms

### Top Time-Consuming Operations:
{top_ops_md if top_ops_md else "No operation breakdown available."}

### CPU Reference Fallbacks (Roundtrip Overhead):
{cpu_fallbacks_md}

## Conclusions & Next Bottlenecks
Continue iterating to convert remaining high-cost CPU reference fallbacks or naive CUDA kernels into optimized GPU kernels.
"""


def runWorker():
    print("[SYSTEM] KernelBench worker thread started and listening for jobs...")
    while True:
        job_id = None
        with worker_lock:
            for jid, j in jobs.items():
                if j["status"] == "queued":
                    job_id = jid
                    j["status"] = "running"
                    break

        if not job_id:
            time.sleep(1)
            continue

        job = jobs[job_id]
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        v_id = job.get("version")
        if v_id is None:
            v_id = getNextVersionId()
            job["version"] = v_id

        v_dir = VERSIONS_DIR / str(v_id)
        v_dir.mkdir(parents=True, exist_ok=True)

        idea_text = job.get("idea", "Iterative optimization").strip()
        (v_dir / "idea.md").write_text(idea_text + "\n", encoding="utf-8")

        print(f"\n[JOB {job_id}] Processing Version {v_id}: {idea_text}")

        try:
            # Step 1: Write kernel source if supplied
            if job.get("source"):
                backend = job.get("backend", "cuda").lower()
                ext = ".cu" if backend in ("cuda", "cublas") else ".hpp"
                gen_dir = KERNELS_DIR / "generated" / str(v_id)
                gen_dir.mkdir(parents=True, exist_ok=True)
                kernel_path = (gen_dir / f"kernel{ext}").resolve()

                kernel_path.write_text(job["source"], encoding="utf-8")
                job["kernel_file"] = str(kernel_path)
                job["agent_file_path"] = str(kernel_path.relative_to(KERNELS_DIR))
                print(f"[JOB {job_id}] Staged candidate kernel at {kernel_path}")

            python_exe = getPythonExe()
            target_model = job.get("target_model", "gemma-3-270m")
            pp = job.get("pp", 512)
            tg = job.get("tg", 128)
            min_compile_time = STATIC_MIN_COMPILE_TIME

            # Step 2: Build test, bench, and bench_model
            job["step"] = "build"
            print(f"[JOB {job_id}] Step 1/7: Building test, bench, and bench_model...")
            build_log = v_dir / "build.log"
            build_res = runCmd(
                [python_exe, "build.py", "--targets", "test", "bench", "bench_model", "--log-level", "DEBUG", "--opencl", "0"],
                TIMEOUTS["build"],
                log_path=build_log,
            )
            job["steps"]["build"] = build_res
            if build_res["exit_code"] != 0:
                raise Exception(f"Build failed with exit code {build_res['exit_code']}.\nCheck {build_log}")

            # Step 3: Clear dirty region caches before first compilation
            job["step"] = "clear_cache"
            print(f"[JOB {job_id}] Step 2/7: Clearing dirty region caches...")
            CACHE_DIR.mkdir(exist_ok=True)
            for cache_file in CACHE_DIR.glob("*.bin"):
                cache_file.unlink(missing_ok=True)

            # Step 4: First pass of bench_model to compile & populate calls.bin
            job["step"] = "populate_calls"
            print(f"[JOB {job_id}] Step 3/7: Running bench_model to populate calls.bin...")
            bench_model_bin = str(PROJECT_ROOT / "tensor_graphs_cpp" / "bench_model")
            bench_bin = str(PROJECT_ROOT / "tensor_graphs_cpp" / "bench")
            test_bin = str(PROJECT_ROOT / "tensor_graphs_cpp" / "test")

            pop_res = runCmd(
                [bench_model_bin, "--min-compile-time", str(min_compile_time), "--pp", str(pp), "--tg", str(tg), "--iters", "1", "--warmup", "0"],
                TIMEOUTS["bench_model"],
            )
            job["steps"]["populate_calls"] = pop_res

            # Step 5: Test submitted kernel using fused kernel testing on calls.bin shapes
            kernel_names = []
            if job.get("kernel_name"):
                kernel_names = [job["kernel_name"]]
            elif job.get("source"):
                kernel_names = extractRegisteredKernelNames(job["source"])
            elif job.get("kernel_file") and Path(job["kernel_file"]).exists():
                kernel_names = extractRegisteredKernelNames(Path(job["kernel_file"]).read_text(encoding="utf-8", errors="ignore"))

            if kernel_names:
                job["step"] = "test_kernel"
                print(f"[JOB {job_id}] Step 4/7: Testing submitted kernel(s) {kernel_names} on shapes in calls.bin...")
                test_log = v_dir / "test.log"
                for kname in kernel_names:
                    test_res = runCmd([test_bin, kname], TIMEOUTS["test"], log_path=test_log, append_log=True)
                    job["steps"][f"test_{kname}"] = test_res
                    if test_res["exit_code"] != 0:
                        raise Exception(
                            f"Fused kernel test failed for kernel '{kname}' (exit code {test_res['exit_code']}).\n"
                            f"Test output:\n{test_res.get('stdout', '')}\n{test_res.get('stderr', '')}"
                        )
                print(f"[JOB {job_id}] Kernel test passed successfully!")

            # Step 6: Benchmark newly added calls using bench (with timeout)
            job["step"] = "bench_kernels"
            print(f"[JOB {job_id}] Step 5/7: Running bench to populate records.bin (timeout {TIMEOUTS['bench']}s)...")
            bench_res = runCmd([bench_bin], TIMEOUTS["bench"])
            job["steps"]["bench"] = bench_res

            # Step 7: Final bench_model run with new records
            job["step"] = "bench_model"
            print(f"[JOB {job_id}] Step 6/7: Running bench_model with new records...")
            bench_log = v_dir / "bench_model.log"
            # Remove dirty region cache between compilations so bench_model replans fresh with new records
            for cache_file in CACHE_DIR.glob("*.bin"):
                cache_file.unlink(missing_ok=True)
            bench_cmd = [
                bench_model_bin,
                "--min-compile-time", str(min_compile_time),
                "--pp", str(pp),
                "--tg", str(tg),
            ]
            bench_final_res = runCmd(
                bench_cmd,
                TIMEOUTS["bench_model"],
                log_path=bench_log,
            )
            job["steps"]["bench_model"] = bench_final_res
            if bench_final_res["exit_code"] != 0:
                raise Exception(f"bench_model failed with exit code {bench_final_res['exit_code']}.\nCheck {bench_log}")

            # Step 8: Analyze performance cache before subsequent compilations delete it
            job["step"] = "cache_analysis"
            print(f"[JOB {job_id}] Step 7/7: Running cache analysis...")
            cache_analysis_log = v_dir / "cache_analysis.log"
            cache_file_target = CACHE_DIR / f"bench_{target_model}-pp{pp}-tg{tg}.bin"
            if not cache_file_target.exists():
                candidate_caches = list(CACHE_DIR.glob(f"*{target_model}*.bin"))
                if candidate_caches:
                    cache_file_target = candidate_caches[0]

            if cache_file_target.exists():
                runCmd(
                    [python_exe, "utils/analyze_performance.py", "--graph", str(cache_file_target)],
                    TIMEOUTS["analysis"],
                    log_path=cache_analysis_log,
                )
                try:
                    shutil.copy2(cache_file_target, v_dir / cache_file_target.name)
                except Exception:
                    pass
            else:
                cache_analysis_log.write_text(f"Cache file {cache_file_target} not generated.\n", encoding="utf-8")

            # Parse results
            metrics = parseBenchModelMetrics(bench_log.read_text(encoding="utf-8", errors="ignore"))
            cache_analysis = parseCacheAnalysis(cache_analysis_log.read_text(encoding="utf-8", errors="ignore"))

            job["metrics"] = metrics
            job["cache_analysis"] = cache_analysis

            # Generate README.md
            readme_text = generateVersionReadme(v_id, idea_text, metrics, cache_analysis, target_model)
            (v_dir / "README.md").write_text(readme_text, encoding="utf-8")

            pp_res = metrics.get(f"pp{pp}", {})
            tg_res = metrics.get(f"tg{tg}", {})
            target_info = LLAMA_CPP_TARGETS.get(target_model, {})
            target_beaten = False
            if pp_res and tg_res and target_info:
                target_beaten = (
                    pp_res.get("tps", 0) > target_info.get(f"pp{pp}", {}).get("tps", 0)
                    and tg_res.get("tps", 0) > target_info.get(f"tg{tg}", {}).get("tps", 0)
                )

            # Check for speedup across previous versions
            prev_versions = getAllVersions()
            best_prev_pp = max((v.get("metrics", {}).get(f"pp{pp}", {}).get("tps", 0.0) for v in prev_versions if v.get("version") != v_id), default=0.0)
            best_prev_tg = max((v.get("metrics", {}).get(f"tg{tg}", {}).get("tps", 0.0) for v in prev_versions if v.get("version") != v_id), default=0.0)
            curr_pp = pp_res.get("tps", 0.0)
            curr_tg = tg_res.get("tps", 0.0)
            has_speedup = (curr_pp > best_prev_pp * 1.001) or (curr_tg > best_prev_tg * 1.001) or target_beaten

            job["speedup"] = has_speedup
            if job.get("kernel_file") and os.path.exists(job["kernel_file"]):
                if has_speedup:
                    job["retained"] = True
                    print(f"[JOB {job_id}] Kernel demonstrated speedup (pp: {curr_pp:.2f} vs {best_prev_pp:.2f}, tg: {curr_tg:.2f} vs {best_prev_tg:.2f})! Retained in {job['kernel_file']}.")
                else:
                    job["retained"] = False
                    print(f"[JOB {job_id}] No speedup demonstrated. Cleaning up {job['kernel_file']} to avoid clutter.")
                    try:
                        os.remove(job["kernel_file"])
                        parent = Path(job["kernel_file"]).parent
                        if parent.exists() and not any(parent.iterdir()):
                            parent.rmdir()
                    except Exception:
                        pass

            job["target_beaten"] = target_beaten
            job["status"] = "completed"
            job["step"] = "done"
            print(f"[SUCCESS] Version {v_id} completed successfully! Metrics: {metrics}. Target beaten: {target_beaten}")

        except Exception as err:
            job["status"] = "failed"
            job["error"] = str(err)
            print(f"[ERROR] Version {v_id} job {job_id} failed: {err}")

            if job.get("kernel_file") and os.path.exists(job["kernel_file"]):
                try:
                    os.remove(job["kernel_file"])
                    parent = Path(job["kernel_file"]).parent
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                except Exception:
                    pass

        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        saveJobHistory(job)


def startWorker():
    t = threading.Thread(target=runWorker, daemon=True)
    t.start()
    return t


def createJob(
    source: str = "",
    opname: str = "",
    backend: str = "cuda",
    target_model: str = "gemma-3-270m",
    idea: str = "",
    filename: str = "",
    pp: int = 512,
    tg: int = 128,
    version: int = None,
    kernel_name: str = "",
    min_compile_time: float = STATIC_MIN_COMPILE_TIME,
) -> str:
    job_id = uuid.uuid4().hex[:12]
    if not idea:
        idea = f"Optimize {opname}" if opname else "Bench iteration"

    job = {
        "job_id": job_id,
        "status": "queued",
        "step": "pending",
        "version": version,
        "idea": idea,
        "backend": backend,
        "target_model": target_model,
        "opname": opname,
        "kernel_name": kernel_name or opname,
        "source": source,
        "filename": filename,
        "pp": pp,
        "tg": tg,
        "min_compile_time": STATIC_MIN_COMPILE_TIME,
        "started_at": None,
        "completed_at": None,
        "metrics": {},
        "target_beaten": False,
        "steps": {
            "build": None,
            "populate_calls": None,
            "bench": None,
            "bench_model": None,
        },
    }
    with worker_lock:
        jobs[job_id] = job
    return job_id
