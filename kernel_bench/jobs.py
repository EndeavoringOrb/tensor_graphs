"""Kernel-only experiments with a durable queue and an exclusive GPU worker."""
import hashlib
import io
import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

from psycopg.types.json import Jsonb

from .storage import fingerprint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOAL_CACHE = PROJECT_ROOT / "dirty_region_caches/embeddinggemma-2-image-current.bin.dirty-cache"
WORKER_LOCK = 724189002
TIMEOUTS = {"build": 600, "discovery": 1800, "evaluate": 900}
logger = logging.getLogger(__name__)


def getPythonExe():
    path = PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not path.is_file():
        raise RuntimeError(f"Project Python environment is missing: {path}")
    return str(path)


def stripComments(source):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)


def extractRegisteredKernelNames(source_text):
    return re.findall(r'\bREGISTER_KERNEL\s*\(\s*"([^"\n]+)"', stripComments(source_text))


def sourceArchive():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        paths = [PROJECT_ROOT / "build.py"]
        for directory in ("tensor_graphs_cpp", "docs/hwinfo/box"):
            paths.extend(sorted((PROJECT_ROOT / directory).rglob("*")))
        for path in paths:
            relative = path.relative_to(PROJECT_ROOT)
            if (path.is_file() and not relative.as_posix().startswith("tensor_graphs_cpp/generated/")
                    and not relative.as_posix().startswith("tensor_graphs_cpp/kernels/generated/kernel_bench/")
                    and path.suffix in (".py", ".cpp", ".hpp", ".h", ".cu", ".cl", ".txt")):
                info = zipfile.ZipInfo(relative.as_posix())
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
    return output.getvalue()


def hardwareIdentity():
    identity = {"machine": platform.machine(), "processor": platform.processor(),
                "python": platform.python_version(), "system": platform.system()}
    if shutil.which("nvidia-smi"):
        result = runCmd(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], PROJECT_ROOT, 30)
        if result["exit_code"] != 0:
            raise RuntimeError("GPU identity query failed: " + result["output"])
        identity["gpus"] = result["output"].strip()
    return identity


class JobFailure(RuntimeError):
    def __init__(self, message, details, result=None):
        super().__init__(message)
        self.details = details
        self.result = result or {}


def readableOutput(output):
    output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    lines = output.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(line for line in lines if not re.match(r"\s*linting \[\d+/\d+\]:", line))


def runCmd(command, cwd, timeout, payload=None, env=None):
    started_at = time.monotonic()
    logger.info("Starting %s (timeout %ss)", " ".join(command), timeout)
    process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env,
                               start_new_session=os.name != "nt")
    streams = {"stdout": [], "stderr": []}

    def readOutput(stream, name):
        # Read both pipes concurrently; preserve raw text before any log cleanup.
        while chunk := stream.read1(8192):
            streams[name].append(chunk)
            readable = readableOutput(chunk.decode("utf-8", errors="replace"))
            if readable.strip():
                logger.info("[%s %s] %s", process.pid, name, readable)

    readers = [threading.Thread(target=readOutput, args=(stream, name), daemon=True)
               for name, stream in (("stdout", process.stdout), ("stderr", process.stderr))]
    for reader in readers:
        reader.start()
    timed_out = False
    try:
        if payload is not None:
            process.stdin.write(json.dumps(payload).encode())
        process.stdin.close()
        while True:
            remaining = timeout - (time.monotonic() - started_at)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                process.wait(timeout=min(15, remaining))
                break
            except subprocess.TimeoutExpired:
                logger.info("[%s] Still running after %.0fs", process.pid, time.monotonic() - started_at)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    finally:
        for reader in readers:
            reader.join()
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
    stdout, stderr = (b"".join(streams[name]).decode("utf-8", errors="replace") for name in ("stdout", "stderr"))
    output = readableOutput(stdout + "\n" + stderr)
    if timed_out:
        output += f"\nTimed out after {timeout}s"
    logger.info("[%s] Finished in %.1fs (exit %s)", process.pid, time.monotonic() - started_at, process.returncode)
    return {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr,
            "output": output, "timed_out": timed_out, "command": command}


def runNative(workspace, payload, timeout):
    output_path = workspace / "native_result.json"
    output_path.unlink(missing_ok=True)
    result = runCmd([getPythonExe(), "-u", str(Path(__file__).with_name("native_runner.py"))],
                    workspace, timeout, {**payload, "output_path": str(output_path)})
    for stream in ("stdout", "stderr"):
        (workspace / f"native.{stream}").write_text(result[stream])
    (workspace / "native_command.json").write_text(json.dumps(result["command"]))
    envelope = json.loads(output_path.read_text()) if output_path.is_file() else {}
    if result["exit_code"] != 0 or envelope.get("error"):
        details = envelope.get("error_details", {})
        details.update({"exit_code": result["exit_code"],
                        "signal": -result["exit_code"] if result["exit_code"] < 0 else None,
                        "diagnostic": result["output"]})
        if result["timed_out"]:
            details["kind"] = "timeout"
        else:
            details.setdefault("kind", "native_crash" if result["exit_code"] < 0 else "native_failure")
        details.setdefault("stage", "native")
        raise JobFailure(envelope.get("error") or f"Native {details['stage']} failed (exit {result['exit_code']})",
                         details, envelope.get("result", {}))
    if not envelope:
        raise JobFailure("Native worker returned no structured result", {"kind": "missing_result", "stage": "native"})
    return envelope.get("result", envelope), result["output"]


def initializeGoal(store, cache_path):
    cache_path = Path(cache_path).resolve()
    logger.info("Preparing goal cache %s", cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Goal cache is missing: {cache_path}. Set KERNEL_BENCH_GOAL_CACHE to an existing compiled cache.")
    cache_data = cache_path.read_bytes()
    archive = sourceArchive()
    cache_hash = hashlib.sha256(cache_data).hexdigest()
    hardware = hardwareIdentity()
    goal_hash = fingerprint({"cache": cache_hash, "sources": hashlib.sha256(archive).hexdigest(), "hardware": hardware})
    with store.connection() as lock_connection:
        lock_connection.autocommit = True
        logger.info("Waiting for the discovery/GPU worker database lock")
        lock_connection.execute("SELECT pg_advisory_lock(%s)", (WORKER_LOCK,))
        lock_connection.execute("SELECT pg_advisory_lock(%s)", (WORKER_LOCK + 2,))
        logger.info("Acquired discovery/GPU worker database lock")
        try:
            existing = store.query("SELECT id FROM kernel_bench.goals WHERE fingerprint = %s", (goal_hash,))
            if existing:
                logger.info("Reusing stored discovery for goal %s", existing[0]["id"])
                return existing[0]["id"]
            with tempfile.TemporaryDirectory(prefix="kernel-bench-discover-") as temp_path:
                workspace = Path(temp_path)
                with zipfile.ZipFile(io.BytesIO(archive)) as source_zip:
                    source_zip.extractall(workspace)
                result = runCmd([getPythonExe(), "build.py", "--targets", "kernel_bench_bindings", "--opencl", "0"],
                                workspace, TIMEOUTS["build"], env={**os.environ, "KERNEL_BENCH_SERVER_JOB": "1"})
                if result["exit_code"] != 0:
                    raise RuntimeError(result["output"])
                goal_file = workspace / "goal.bin"
                goal_file.write_bytes(cache_data)
                logger.info("Discovering reference matches by CPU graph saturation; no model inference is run")
                discovered, discovery_log = runNative(workspace, {"action": "discover", "cache_path": str(goal_file)}, TIMEOUTS["discovery"])
                uid_metadata = json.loads((workspace / "tensor_graphs_cpp/generated/kernel_uids.json").read_text())
            for kernel in discovered["kernels"]:
                kernel["source_path"] = "tensor_graphs_cpp/" + uid_metadata[str(kernel["kernel_id"])]["path"]
            metadata = {"kernels": discovered["kernels"], "discovery_log": discovery_log,
                        "build_log": result["output"], "scope": "kernel", "cache_path": str(cache_path), "hardware": hardware}
            groups = {}
            for call in discovered["calls"]:
                expression = call["reference_expression"]
                configuration = {key: call[key] for key in ("input_shapes", "input_strides", "input_dtypes", "input_constants",
                                                            "output_shape", "output_strides", "output_dtype")}
                config_hash = fingerprint(configuration)
                group = groups.setdefault((expression, config_hash), {"configuration": configuration, "kernels": {}})
                if call["kernel_matches"]:
                    group["kernels"][str(call["kernel_id"])] = call["kernel_name"]
            with store.connection() as connection:
                goal_id = connection.execute("""INSERT INTO kernel_bench.goals
                    (fingerprint, name, cache_sha256, cache_data, source_archive, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                    (goal_hash, cache_path.name, cache_hash, cache_data, archive, Jsonb(metadata))).fetchone()["id"]
                for (expression, config_hash), group in groups.items():
                    target_id = hashlib.sha256(expression.encode()).hexdigest()
                    connection.execute("INSERT INTO kernel_bench.targets(id, expression) VALUES (%s, %s) ON CONFLICT DO NOTHING", (target_id, expression))
                    connection.execute("""INSERT INTO kernel_bench.configurations
                        (goal_id, target_id, fingerprint, configuration, matching_kernels)
                        VALUES (%s, %s, %s, %s, %s)""",
                        (goal_id, target_id, config_hash, Jsonb(group["configuration"]), Jsonb(group["kernels"])))
            logger.info("Stored goal %s: %s configurations", goal_id, len(groups))
            return goal_id
        finally:
            lock_connection.execute("SELECT pg_advisory_unlock(%s)", (WORKER_LOCK + 2,))
            lock_connection.execute("SELECT pg_advisory_unlock(%s)", (WORKER_LOCK,))


def createJob(store, goal_id, source, target_id, idea="", backend="cuda", mode="evaluate"):
    if mode not in ("lint", "compile", "evaluate"):
        raise ValueError("mode must be lint, compile or evaluate")
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be cpu or cuda")
    if not source or len(source.encode()) > 1024 * 1024:
        raise ValueError("Provide a kernel source file smaller than 1 MiB")
    names = extractRegisteredKernelNames(source)
    registrations = re.findall(r"\bREGISTER_(?:REF_)?KERNEL(?:_VIEW)?\s*\(", stripComments(source))
    if len(names) != 1 or len(registrations) != 1:
        raise ValueError("Submit exactly one REGISTER_KERNEL with a literal, unique name")
    if not store.getConfigurations(goal_id, target_id):
        raise ValueError("Unknown target for this goal")
    metadata = store.getGoal(goal_id)["metadata"]
    if any(kernel["name"] == names[0] for kernel in metadata["kernels"]):
        raise ValueError("Candidate kernel name is already registered in the baseline")
    job_id = uuid.uuid4()
    store.execute("""INSERT INTO kernel_bench.jobs(id, goal_id, target_id, idea, source, kernel_name, backend, mode)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""", (job_id, goal_id, target_id, idea, source, names[0], backend, mode))
    return str(job_id)


def evaluateJob(store, job, workspace):
    goal = store.query("SELECT source_archive FROM kernel_bench.goals WHERE id = %s", (job["goal_id"],))[0]
    if not (workspace / "build.py").is_file():
        with zipfile.ZipFile(io.BytesIO(bytes(goal["source_archive"]))) as archive:
            archive.extractall(workspace)
    staging = workspace / "tensor_graphs_cpp/kernels/generated/kernel_bench"
    staging.mkdir(parents=True, exist_ok=True)
    for path in staging.iterdir():
        if path.is_file():
            path.unlink()
    candidate_path = staging / ("candidate.cu" if job["backend"] == "cuda" else "candidate.hpp")
    candidate_path.write_text(job["source"])
    diagnostic_dir = workspace / "build_diagnostics"
    if diagnostic_dir.exists():
        shutil.rmtree(diagnostic_dir)
    diagnostic_dir.mkdir()
    artifact_ids = []
    def saveArtifact(name, data, media_type="text/plain"):
        artifact_ids.append(store.addArtifact(job["id"], name, data, media_type))

    saveArtifact("submitted_source", job["source"].encode())
    mode = job.get("mode", "evaluate")
    build_result = None
    for name in ("native_result.json", "native.stdout", "native.stderr", "native_command.json"):
        (workspace / name).unlink(missing_ok=True)
    try:
        store.addEvent(job["id"], "lint", "Validating candidate source")
        build_env = {**os.environ, "KERNEL_BENCH_SERVER_JOB": "1",
                     "KERNEL_BENCH_DIAGNOSTICS_DIR": str(diagnostic_dir.resolve())}
        lint_command = [getPythonExe(), "build.py", "--lint-candidate", str(candidate_path.relative_to(workspace))]
        baseline_marker = workspace / ".baseline_linted"
        if not baseline_marker.exists() and mode != "lint":
            candidate_path.unlink()
            baseline = runCmd([getPythonExe(), "build.py", "--lint"], workspace, TIMEOUTS["build"], env=build_env)
            candidate_path.write_text(job["source"])
            build_result = baseline
            if baseline["exit_code"] != 0:
                raise JobFailure("Baseline lint failed", {"kind": "lint_error", "stage": "lint", "diagnostic": baseline["output"]})
            baseline_marker.touch()
        result = runCmd(lint_command, workspace, TIMEOUTS["build"], env=build_env)
        if result["exit_code"] == 0 and mode != "lint":
            store.addEvent(job["id"], "build", "Candidate lint passed; compiling native bindings")
            result = runCmd([getPythonExe(), "build.py", "--targets", "kernel_bench_bindings", "--opencl", "0", "--no-lint"],
                            workspace, TIMEOUTS["build"], env=build_env)
        build_result = result
        store.addEvent(job["id"], "lint" if "--lint-candidate" in result["command"] else "build", result["output"])
        if result["exit_code"] != 0:
            compiler_records = []
            for path in sorted(diagnostic_dir.glob("compiler-*.json")):
                record = json.loads(path.read_text())
                if record["exit_code"] is None:
                    for stream in ("stdout", "stderr"):
                        stream_path = path.with_suffix("." + stream)
                        if stream_path.is_file():
                            record[stream] = stream_path.read_text(errors="replace")
                compiler_records.append(record)
            failed = [record for record in compiler_records if record["exit_code"] != 0]
            diagnostic = "\n".join(record["stderr"] or record["stdout"] for record in failed) or result["output"]
            kind = "timeout" if result["timed_out"] else "compilation_error" if failed else "lint_error" if "Violation" in diagnostic or "DETECTED" in diagnostic else "build_error"
            diagnostic_lines = diagnostic.splitlines()
            summary = next((line.strip(" │") for line in diagnostic_lines if "contains a manual" in line), None)
            if summary is None:
                summary = next((line.strip(" │") for line in diagnostic_lines
                                if "error:" in line.lower() or "DETECTED" in line or "Violation" in line),
                               f"Build failed with exit code {result['exit_code']}")
            raise JobFailure(summary[:1000], {"kind": kind, "stage": "lint" if "--lint-candidate" in result["command"] else "build", "exit_code": result["exit_code"],
                                             "signal": -result["exit_code"] if result["exit_code"] < 0 else None,
                                             "diagnostic": diagnostic, "compiler_diagnostics": failed})
        validation_fingerprint = fingerprint({"source": job["source"], "goal": store.getGoal(job["goal_id"])["fingerprint"],
                                              "backend": job["backend"], "build_command": [arg.replace(str(workspace), "<workspace>") for arg in build_result["command"]],
                                              "metadata": json.loads((diagnostic_dir / "build_metadata.json").read_text())
                                              if (diagnostic_dir / "build_metadata.json").exists() else {}})
        if mode == "lint":
            store.completeJob(job["id"], {"mode": mode, "passed_stages": ["submission", "lint"], "measurements": [],
                                          "validation_id": str(job["id"]), "source_fingerprint": hashlib.sha256(job["source"].encode()).hexdigest(),
                                          "build_fingerprint": validation_fingerprint})
            return
        configurations = store.getConfigurations(job["goal_id"], job["target_id"])
        expression = store.query("SELECT expression FROM kernel_bench.targets WHERE id = %s", (job["target_id"],))[0]["expression"]
        native_stage = "registration" if mode == "compile" else "test_bench"
        store.addEvent(job["id"], native_stage, "Validating executable registration and target" if mode == "compile" else
                       "Testing covered goal configurations, then benchmarking against existing kernels")
        result, log = runNative(workspace, {"action": "compile" if mode == "compile" else "evaluate", "kernel_name": job["kernel_name"],
                                           "reference_expression": expression,
                                           "configurations": [{"id": row["id"], "configuration": row["configuration"]} for row in configurations]}, TIMEOUTS["evaluate"])
        store.addEvent(job["id"], native_stage, log)
        result.update({"mode": mode, "validation_id": str(job["id"]), "source_fingerprint": hashlib.sha256(job["source"].encode()).hexdigest(),
                       "build_fingerprint": validation_fingerprint,
                       "passed_stages": ["submission", "lint", "build", "load", "registration", "signature"] +
                       (["correctness", "benchmark", "comparison"] if mode == "evaluate" else [])})
    except JobFailure as error:
        error.details["artifact_ids"] = artifact_ids
        metadata_path = diagnostic_dir / "build_metadata.json"
        if metadata_path.is_file():
            error.details["build_capabilities"] = json.loads(metadata_path.read_text())
        raise
    finally:
        # Persist actual generated files before removing the candidate or workspace.
        saveArtifact(candidate_path.relative_to(workspace).as_posix(), candidate_path.read_bytes())
        for directory in (workspace / "tensor_graphs_cpp/generated", diagnostic_dir):
            if directory.exists():
                for path in sorted(directory.rglob("*")):
                    if path.is_file() and (directory == diagnostic_dir or path.suffix in (".hpp", ".h", ".cpp", ".cu", ".json")):
                        saveArtifact(path.relative_to(workspace).as_posix(), path.read_bytes(),
                                     "application/json" if path.suffix == ".json" else "text/plain")
        if build_result:
            for stream in ("stdout", "stderr"):
                saveArtifact(f"build.{stream}", build_result[stream].encode())
            saveArtifact("build_command.json", json.dumps(build_result["command"]).encode(), "application/json")
        for name in ("native_result.json", "native.stdout", "native.stderr", "native_command.json"):
            native_path = workspace / name
            if native_path.is_file():
                saveArtifact(name, native_path.read_bytes(), "application/json" if name.endswith(".json") else "text/plain")
        candidate_path.unlink(missing_ok=True)
    store.completeJob(job["id"], result)


def runWorker(store, stop_event, validation=False):
    worker_lock = WORKER_LOCK + 1 if validation else WORKER_LOCK
    with tempfile.TemporaryDirectory(prefix="kernel-bench-worker-") as temp_path:
        while not stop_event.is_set():
            with store.connection() as lock_connection:
                lock_connection.autocommit = True
                acquired = lock_connection.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (worker_lock,)).fetchone()["acquired"]
                if not acquired:
                    stop_event.wait(1)
                    continue
                try:
                    if stop_event.is_set():
                        break
                    store.execute("UPDATE kernel_bench.jobs SET status = 'queued', step = 'recovered' WHERE status = 'running' AND (mode <> 'evaluate') = %s", (validation,))
                    job = store.nextJob(validation)
                    if job is not None:
                        workspace = Path(temp_path) / str(job["goal_id"])
                        workspace.mkdir(exist_ok=True)
                        try:
                            if job.get("mode", "evaluate") != "lint":
                                # Shared across processes: builds and measurements never overlap.
                                lock_connection.execute("SELECT pg_advisory_lock(%s)", (WORKER_LOCK + 2,))
                            try:
                                evaluateJob(store, job, workspace)
                            finally:
                                if job.get("mode", "evaluate") != "lint":
                                    lock_connection.execute("SELECT pg_advisory_unlock(%s)", (WORKER_LOCK + 2,))
                        except Exception as error:  # noqa: BLE001 - persist native/build failures and keep the queue alive
                            details = getattr(error, "details", {})
                            message = str(error)
                            if details.get("diagnostic"):
                                message += "\n" + details["diagnostic"]
                            if details.get("artifact_ids"):
                                message += "\nArtifact IDs: " + ", ".join(details["artifact_ids"])
                            store.addEvent(job["id"], "error", message)
                            store.failJob(job["id"], str(error),
                                          getattr(error, "details", {"kind": "worker_failure", "stage": job.get("step", "worker")}),
                                          getattr(error, "result", {}))
                finally:
                    lock_connection.execute("SELECT pg_advisory_unlock(%s)", (worker_lock,))
            if job is None:
                stop_event.wait(1)


def startWorker(store, stop_event=None, validation=False):
    stop_event = stop_event if stop_event is not None else threading.Event()
    thread = threading.Thread(target=runWorker, args=(store, stop_event, validation), name="kernel-bench-validation-worker" if validation else "kernel-bench-worker", daemon=True)
    thread.start()
    return thread
