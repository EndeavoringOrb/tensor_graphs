"""Kernel-only experiments with a durable queue and an exclusive GPU worker."""
from contextlib import nullcontext
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
from .policy import NATIVE_INTERFACE_VERSION, baselineEligible, configurationErrors

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
        paths = [PROJECT_ROOT / "build.py", PROJECT_ROOT / "kernel_linter_rules.py", PROJECT_ROOT / "kernel_linter.py",
                 PROJECT_ROOT / "embeddinggemma2.py", PROJECT_ROOT / "goal_validation.py", PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "uv.lock"]
        for directory in ("tensor_graphs_cpp", "docs/hwinfo/box", "tests/embeddinggemma-2", "tests/gemma-3-270m"):
            paths.extend(sorted((PROJECT_ROOT / directory).rglob("*")))
        for path in paths:
            relative = path.relative_to(PROJECT_ROOT)
            if (path.is_file() and not relative.as_posix().startswith("tensor_graphs_cpp/generated/")
                    and not relative.as_posix().startswith("tensor_graphs_cpp/kernels/generated/kernel_bench/")
                    and path.suffix in (".py", ".cpp", ".hpp", ".h", ".cu", ".cl", ".txt", ".toml", ".lock", ".png", ".mp4", ".wav")):
                info = zipfile.ZipInfo(relative.as_posix())
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
    return output.getvalue()


def hardwareIdentity():
    identity = {"machine": platform.machine(), "processor": platform.processor(),
                "python": platform.python_version(), "system": platform.system(),
                "hostname": platform.node(), "cpu_count": os.cpu_count()}
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.is_file():
        identity["memory_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        identity["cpu_model"] = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                                     if line.startswith("model name")), platform.processor())
    if shutil.which("nvidia-smi"):
        result = runCmd(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"], PROJECT_ROOT, 30)
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
    job_context = (env or {}).get("KERNEL_BENCH_JOB_ID") or (payload or {}).get("job_id")
    context = f"job={job_context} workspace={cwd}" if job_context else str(cwd)
    logger.info("[%s] Starting %s (timeout %ss)", context, " ".join(command), timeout)
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
    (logger.error if timed_out or process.returncode else logger.info)(
        "[%s pid=%s] Finished in %.1fs (exit %s)", context, process.pid, time.monotonic() - started_at, process.returncode)
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


def initializeGoal(store, cache_path, promotion_recipe="environment", already_locked=False):
    cache_path = Path(cache_path).resolve()
    logger.info("Preparing goal cache %s", cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Goal cache is missing: {cache_path}. Set KERNEL_BENCH_GOAL_CACHE to an existing compiled cache.")
    cache_data = cache_path.read_bytes()
    archive = sourceArchive()
    cache_hash = hashlib.sha256(cache_data).hexdigest()
    hardware = hardwareIdentity()
    from .promotion import promotionConfig
    promotion_config = promotionConfig() if promotion_recipe == "environment" else promotion_recipe
    goal_hash = fingerprint({"cache": cache_hash, "sources": hashlib.sha256(archive).hexdigest(), "hardware": hardware,
                             "native_interface_version": NATIVE_INTERFACE_VERSION, "promotion_recipe": promotion_config})
    with (nullcontext(None) if already_locked else store.connection()) as lock_connection:
        if not already_locked:
            lock_connection.autocommit = True
            logger.info("Waiting for the discovery/GPU worker database lock")
            lock_connection.execute("SELECT pg_advisory_lock(%s)", (WORKER_LOCK,))
            lock_connection.execute("SELECT pg_advisory_lock(%s)", (WORKER_LOCK + 2,))
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
                        "build_log": result["output"], "scope": "kernel", "cache_path": str(cache_path), "hardware": hardware,
                        "enabled_capabilities": discovered["enabled_capabilities"],
                        "native_interface_version": NATIVE_INTERFACE_VERSION,
                        "promotion_recipe": promotion_config,
                        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()}
            kernels_by_id = {kernel["kernel_id"]: kernel for kernel in discovered["kernels"]}
            groups = {}
            invalid_configurations = {}
            for call in discovered["calls"]:
                expression = call["reference_expression"]
                configuration = {key: call[key] for key in ("input_shapes", "input_strides", "input_dtypes", "input_constants",
                                                            "output_shape", "output_strides", "output_dtype")}
                config_hash = fingerprint(configuration)
                errors = configurationErrors(configuration)
                if errors:
                    invalid_configurations[config_hash] = {"reference_expression": expression,
                        "configuration": configuration, "errors": errors}
                    continue
                group = groups.setdefault((expression, config_hash), {"configuration": configuration, "kernels": {}, "rejections": {}})
                if not call["kernel_matches"]:
                    group["rejections"][call["kernel_name"]] = call.get("match_reasons", ["not_executable_on_this_configuration"])
                if call["kernel_matches"] and baselineEligible(kernels_by_id[call["kernel_id"]]):
                    group["kernels"][str(call["kernel_id"])] = call["kernel_name"]
            metadata["invalid_configurations"] = list(invalid_configurations.values())
            metadata["configuration_rejections"] = {expression + ":" + config_hash: group["rejections"]
                for (expression, config_hash), group in groups.items() if group["rejections"]}
            if invalid_configurations:
                logger.error("Discovery quarantined %s invalid configurations; inspect goal metadata", len(invalid_configurations))
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
            if not already_locked:
                lock_connection.execute("SELECT pg_advisory_unlock(%s)", (WORKER_LOCK + 2,))
                lock_connection.execute("SELECT pg_advisory_unlock(%s)", (WORKER_LOCK,))


def createJob(store, goal_id, source, target_id, idea="", backend="cuda", mode="evaluate", agent_id=None):
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
    conflicts = [kernel for kernel in metadata["kernels"] if kernel["name"] == names[0]]
    if conflicts:
        raise ValueError(f"Candidate name {names[0]!r} conflicts with baseline registration "
                         f"{conflicts[0]['kernel_id']} ({conflicts[0].get('source_path', 'source unknown')})")
    job_id = uuid.uuid4()
    store.execute("""INSERT INTO kernel_bench.jobs(id, goal_id, target_id, idea, source, kernel_name, backend, mode, agent_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""", (job_id, goal_id, target_id, idea, source, names[0], backend, mode, agent_id))
    return str(job_id)


def requireNativeInterface(store, goal_id):
    version = store.getGoal(goal_id)["metadata"].get("native_interface_version")
    if version != NATIVE_INTERFACE_VERSION:
        raise JobFailure(f"Goal {goal_id} has incompatible native interface {version}; expected {NATIVE_INTERFACE_VERSION}. Use the current goal.",
                         {"kind": "incompatible_native_interface", "stage": "snapshot"})


def evaluateJob(store, job, workspace):
    requireNativeInterface(store, job["goal_id"])
    goal = store.query("SELECT source_archive FROM kernel_bench.goals WHERE id = %s", (job["goal_id"],))[0]
    if not (workspace / "build.py").is_file():
        with zipfile.ZipFile(io.BytesIO(bytes(goal["source_archive"]))) as archive:
            archive.extractall(workspace)
    if not (workspace / "kernel_linter.py").is_file():
        for name in ("kernel_linter.py", "kernel_linter_rules.py"):
            shutil.copyfile(PROJECT_ROOT / name, workspace / name)
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
                     "KERNEL_BENCH_JOB_ID": str(job["id"]),
                     "KERNEL_BENCH_DIAGNOSTICS_DIR": str(diagnostic_dir.resolve())}
        lint_command = [getPythonExe(), "kernel_linter.py", str(candidate_path.relative_to(workspace))]
        baseline_marker = workspace / ".baseline_linted"
        if not baseline_marker.exists() and mode != "lint":
            candidate_path.unlink()
            baseline = runCmd([getPythonExe(), "kernel_linter.py", "--repository"], workspace, TIMEOUTS["build"], env=build_env)
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
        store.addEvent(job["id"], "lint" if "kernel_linter.py" in result["command"] else "build", result["output"])
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
            raise JobFailure(summary[:1000], {"kind": kind, "stage": "lint" if "kernel_linter.py" in result["command"] else "build", "exit_code": result["exit_code"],
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
                                           "job_id": str(job["id"]),
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
    if mode == "evaluate":
        from .promotion import promotionEligibility
        result["promotion_eligibility"] = promotionEligibility({**job, "status": "completed", "result": result},
            store.getGoal(job["goal_id"])["metadata"].get("promotion_recipe"))
    store.completeJob(job["id"], result)
    if mode == "evaluate":
        from .promotion import enqueuePromotion
        try:
            enqueuePromotion(store, job)
        except Exception as error:  # noqa: BLE001 - retain successful benchmark evidence if promotion submission fails
            store.addEvent(job["id"], "promotion_error", str(error))
            logger.exception("Could not enqueue promotion for %s", job["id"])


def calibrateNext(store, workspace_root, goal_id):
    # Called only by the exclusive evaluation worker, while holding the hardware lock.
    store.execute("""INSERT INTO kernel_bench.baselines(configuration_id)
        SELECT id FROM kernel_bench.configurations WHERE goal_id = %s ON CONFLICT DO NOTHING""", (goal_id,))
    rows = store.query("""SELECT c.*, t.expression FROM kernel_bench.baselines b
        JOIN kernel_bench.configurations c ON c.id = b.configuration_id
        JOIN kernel_bench.targets t ON t.id = c.target_id
        WHERE b.status IN ('pending', 'running') AND (c.goal_id = %s OR b.requested)
        ORDER BY c.goal_id, c.id LIMIT 1""", (goal_id,))
    if not rows:
        return False
    calibrateConfiguration(store, rows[0], workspace_root)
    return True


def calibrateConfiguration(store, row, workspace_root):
    store.execute("UPDATE kernel_bench.baselines SET status = 'running' WHERE configuration_id = %s", (row["id"],))
    workspace = workspace_root / ("baseline-" + str(row["goal_id"]))
    workspace.mkdir(exist_ok=True)
    try:
        requireNativeInterface(store, row["goal_id"])
        if not row["matching_kernels"]:
            store.execute("UPDATE kernel_bench.baselines SET status = 'unavailable', error = %s WHERE configuration_id = %s",
                          ("No matching executable kernels with a reference factory", row["id"]))
            return True
        marker = workspace / ".baseline_built"
        build_identity = fingerprint({"goal": store.getGoal(row["goal_id"])["fingerprint"],
                                      "native_interface_version": NATIVE_INTERFACE_VERSION})
        if not marker.exists() or marker.read_text() != build_identity:
            goal = store.query("SELECT source_archive FROM kernel_bench.goals WHERE id = %s", (row["goal_id"],))[0]
            with zipfile.ZipFile(io.BytesIO(bytes(goal["source_archive"]))) as archive:
                archive.extractall(workspace)
            build_env = {**os.environ, "KERNEL_BENCH_SERVER_JOB": "1", "KERNEL_BENCH_DIAGNOSTICS_DIR": str(workspace / "build_diagnostics")}
            build = runCmd([getPythonExe(), "build.py", "--targets", "kernel_bench_bindings", "--opencl", "0"],
                           workspace, TIMEOUTS["build"], env=build_env)
            if build["exit_code"] != 0:
                raise RuntimeError("Baseline build failed: " + build["output"])
            runNative(workspace, {"action": "preflight"}, TIMEOUTS["evaluate"])
            marker.write_text(build_identity)
        result, _log = runNative(workspace, {"action": "baseline", "reference_expression": row["expression"],
                                            "configuration": {"id": row["id"], "configuration": row["configuration"]}}, TIMEOUTS["evaluate"])
        metadata_path = workspace / "build_diagnostics/build_metadata.json"
        identity = {"hardware": hardwareIdentity(), "goal_fingerprint": store.getGoal(row["goal_id"])["fingerprint"],
                    "toolchain": json.loads(metadata_path.read_text()) if metadata_path.exists() else {},
                    "enabled_capabilities": result["enabled_capabilities"],
                    "methodology": {"seed": 42, "rounds": 2, "iterations": 8, "competitors": "matching non-reference kernels with reference factories"}}
        store.execute("""UPDATE kernel_bench.baselines SET status = 'measured', measurements = %s,
            winner = %s, identity = %s, measured_at = now(), error = NULL, error_details = NULL WHERE configuration_id = %s""",
            (Jsonb(result["measurements"]), Jsonb(result["winner"]), Jsonb(identity), row["id"]))
    except Exception as error:  # noqa: BLE001 - retain per-configuration failures and continue calibration
        status = "unavailable" if isinstance(error, JobFailure) and error.details.get("kind") == "no_matching_kernels" else "failed"
        details = getattr(error, "details", {"kind": "calibration_infrastructure_failure"})
        details = {**details, "configuration_id": row["id"], "configuration": row["configuration"]}
        logger.error("Baseline goal=%s configuration=%s failed: %s", row["goal_id"], row["id"], error)
        store.execute("UPDATE kernel_bench.baselines SET status = %s, error = %s, error_details = %s, measured_at = now() WHERE configuration_id = %s",
                      (status, str(error), Jsonb(details), row["id"]))
        if details.get("kind") in ("incompatible_native_interface", "calibration_infrastructure_failure", "native_crash"):
            store.execute("""UPDATE kernel_bench.baselines b SET status = 'failed', error = %s,
                error_details = %s, measured_at = now() FROM kernel_bench.configurations c
                WHERE c.id = b.configuration_id AND c.goal_id = %s AND b.status IN ('pending', 'running')""",
                          ("Calibration batch stopped: " + str(error), Jsonb(details), row["goal_id"]))
    return True


def runWorker(store, stop_event, validation=False, goal_id=None):
    worker_lock = WORKER_LOCK
    lane = "evaluation"
    heartbeat_stop = threading.Event()
    def heartbeat(heartbeat_stop):
        while not heartbeat_stop.is_set():
            try:
                store.execute("UPDATE kernel_bench.workers SET heartbeat_at = now() WHERE lane = %s", (lane,))
            except Exception:
                logger.exception("Worker heartbeat failed")
            heartbeat_stop.wait(5)
    # Heartbeat ownership begins only after acquiring the shared queue lock.
    heartbeat_thread = None
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
                    store.execute("DELETE FROM kernel_bench.workers WHERE lane = 'validation'")
                    store.execute("""INSERT INTO kernel_bench.workers(lane, stage) VALUES (%s, 'idle')
                        ON CONFLICT(lane) DO UPDATE SET heartbeat_at = now(), job_id = NULL, stage = 'idle'""", (lane,))
                    if heartbeat_thread is None:
                        heartbeat_stop = threading.Event()
                        heartbeat_thread = threading.Thread(target=heartbeat, args=(heartbeat_stop,), daemon=True)
                        heartbeat_thread.start()
                    store.execute("UPDATE kernel_bench.jobs SET status = 'queued', step = 'recovered' WHERE status = 'running'", ())
                    job = store.nextJob(validation)
                    if job is None:
                        store.execute("UPDATE kernel_bench.workers SET stage = 'baseline' WHERE lane = %s", (lane,))
                        lock_connection.execute("SELECT pg_advisory_lock(%s)", (WORKER_LOCK + 2,))
                        try:
                            calibrateNext(store, Path(temp_path), goal_id() if callable(goal_id) else goal_id)
                        finally:
                            lock_connection.execute("SELECT pg_advisory_unlock(%s)", (WORKER_LOCK + 2,))
                    if job is not None:
                        store.execute("UPDATE kernel_bench.workers SET job_id = %s, stage = 'running' WHERE lane = %s", (job["id"], lane))
                        workspace = Path(temp_path) / str(job["goal_id"])
                        workspace.mkdir(exist_ok=True)
                        try:
                            if job.get("mode", "evaluate") != "lint":
                                # Shared across processes: builds and measurements never overlap.
                                lock_connection.execute("SELECT pg_advisory_lock(%s)", (WORKER_LOCK + 2,))
                            try:
                                if job.get("mode") == "goal_setup":
                                    from .admin import prepareGoal
                                    prepareGoal(store, job)
                                elif job.get("mode") == "promote":
                                    from .promotion import promoteJob
                                    promoteJob(store, job)
                                else:
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
                                          getattr(error, "result", None) or store.getJob(job["id"])["result"])
                finally:
                    heartbeat_stop.set()
                    if heartbeat_thread is not None:
                        heartbeat_thread.join(timeout=10)
                        heartbeat_thread = None
                    store.execute("UPDATE kernel_bench.workers SET job_id = NULL, stage = 'idle', heartbeat_at = now() WHERE lane = %s", (lane,))
                    lock_connection.execute("SELECT pg_advisory_unlock(%s)", (worker_lock,))
            if job is None:
                stop_event.wait(1)


def startWorker(store, stop_event=None, validation=False, goal_id=None):
    stop_event = stop_event if stop_event is not None else threading.Event()
    thread = threading.Thread(target=runWorker, args=(store, stop_event, validation, goal_id), name="kernel-bench-validation-worker" if validation else "kernel-bench-worker", daemon=True)
    thread.start()
    return thread
