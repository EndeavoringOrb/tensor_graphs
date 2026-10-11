"""Authenticated kernel-level API. Start with python -m kernel_bench.app."""
import argparse
import hmac
import io
import logging
import os
import signal
import threading
import uuid
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, send_file
from psycopg.types.json import Jsonb
from werkzeug.middleware.proxy_fix import ProxyFix

from .jobs import (
    DEFAULT_GOAL_CACHE,
    PROJECT_ROOT,
    createJob,
    initializeGoal,
    startWorker,
)
from .storage import Store
from .contract import CONTRACT, decodeConstants, targetKernels
from kernel_linter_rules import getRules

logger = logging.getLogger(__name__)

TOOLS = [
    {"name": "kernel_contract", "method": "GET", "path": "/api/kernel-contract", "description": "Native signatures, registration parameters, configuration encoding and CPU/CUDA skeletons"},
    {"name": "linter_rules", "method": "GET", "path": "/api/linter/rules", "description": "Shared build linter rule identifiers, descriptions and scopes"},
    {"name": "kernel_names", "method": "GET", "path": "/api/kernels/list?name=...", "description": "Exact baseline name lookup and availability; independent candidate jobs may reuse names"},
    {"name": "get_status", "method": "GET", "path": "/api/status", "description": "Goal identity and kernel-level scope"},
    {"name": "list_targets", "method": "GET", "path": "/api/targets", "description": "All reference expressions matching the saturated goal graph"},
    {"name": "get_target", "method": "GET", "path": "/api/targets/{target_id}", "description": "Exact workload configurations, constants, strides, and matching kernels"},
    {"name": "read_source", "method": "GET", "path": "/api/source?path=...", "description": "Read baseline kernel/core/reference/hardware source snapshot"},
    {"name": "lint_iteration", "method": "POST", "path": "/api/iteration/lint", "description": "Validate candidate lint without building or benchmarking; returns a queued validation job"},
    {"name": "submit_iteration", "method": "POST", "path": "/api/iteration/submit", "description": "Submit target_id, source, backend (cpu/cuda), idea, and mode (lint/compile/evaluate, default evaluate); one REGISTER_KERNEL"},
    {"name": "get_job", "method": "GET", "path": "/api/jobs/{job_id}", "description": "Correctness coverage, measurements, and per-configuration speedups"},
    {"name": "get_job_logs", "method": "GET", "path": "/api/jobs/{job_id}/logs", "description": "Build/test/benchmark diagnostics; filters: after_id, tail (1–1000), stage, literal grep (max 200 characters)"},
    {"name": "list_job_artifacts", "method": "GET", "path": "/api/jobs/{job_id}/artifacts", "description": "Durable source, build commands, capabilities and full diagnostics"},
    {"name": "read_job_artifact", "method": "GET", "path": "/api/jobs/{job_id}/artifacts/{artifact_id}", "description": "Download an artifact by job-scoped ID"},
    {"name": "list_jobs", "method": "GET", "path": "/api/jobs?target_id=...", "description": "Shared trial history; targets have no reservations"},
]


def createApp(database_url=None, goal_cache=None, token=None, start_worker=True, goal_id=None, stop_event=None):
    load_dotenv(PROJECT_ROOT / ".env")
    token = token or os.environ.get("BENCH_SECRET_TOKEN", "").strip()
    if not token:
        token_file = PROJECT_ROOT / ".bench_token"
        if token_file.is_file():
            token = token_file.read_text().strip()
    if not token:
        raise ValueError("Set BENCH_SECRET_TOKEN before exposing KernelBench")
    store = Store(database_url or os.environ.get("KERNEL_BENCH_DATABASE_URL"))
    store.initialize()
    goal_id = goal_id or initializeGoal(store, goal_cache or os.environ.get("KERNEL_BENCH_GOAL_CACHE", DEFAULT_GOAL_CACHE))
    app = Flask(__name__)
    stop_event = stop_event if stop_event is not None else threading.Event()
    app.extensions["kernel_bench_stop"] = stop_event
    if os.environ.get("BENCH_TRUST_PROXY") == "1":
        app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    app.extensions["kernel_bench_store"] = store
    app.config["KERNEL_BENCH_GOAL_ID"] = goal_id

    def authorized(candidate):
        return bool(candidate) and hmac.compare_digest(candidate, token)

    @app.before_request
    def authenticate():
        if stop_event.is_set():
            return jsonify({"error": "KernelBench is draining for shutdown"}), 503, {"Retry-After": "60"}
        if request.path in ("/login", "/welcome", "/api/agent", "/api/agent/index"):
            return None
        candidate = request.headers.get("X-Bench-Token", "") or request.cookies.get("bench_token", "")
        if authorized(candidate):
            return None
        if request.path.startswith("/api/"):
            return jsonify({"error": "Authentication required: X-Bench-Token"}), 401
        return redirect("/login")

    @app.errorhandler(ValueError)
    @app.errorhandler(TypeError)
    def invalidRequest(error):
        stage = "submission" if request.path in ("/api/iteration/submit", "/api/kernels/test") else "request"
        return jsonify({"error": str(error), "error_details": {
            "kind": "submission_error" if stage == "submission" else "invalid_request",
            "stage": stage, "artifact_ids": []}}), 400

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            if authorized(request.form.get("token", "").strip()):
                response = redirect("/")
                response.set_cookie("bench_token", token, httponly=True, samesite="Strict", secure=request.is_secure)
                return response
            return render_template("login.html", error="Invalid token", next_param="/"), 401
        return render_template("login.html", error=None, next_param="/")

    @app.get("/")
    def index():
        return render_template("kernel.html")

    @app.get("/logout")
    def logout():
        response = redirect("/login")
        response.delete_cookie("bench_token")
        return response

    @app.get("/api/status")
    def status():
        goal = store.getGoal(goal_id)
        metadata = goal.pop("metadata")
        return jsonify({"scope": "kernel", "goal": goal, "target_count": len(store.getTargets(goal_id)),
                        "hardware": metadata.get("hardware", {}),
                        "pipeline": ["build", "test", "benchmark", "compare"]})

    @app.get("/api/targets")
    @app.get("/api/analyze")
    def targets():
        rows = store.getTargets(goal_id)
        search = request.args.get("search", "").lower()
        if search:
            rows = [row for row in rows if search in row["reference_expression"].lower()]
        metadata = store.getGoal(goal_id)["metadata"]
        for row in rows:
            row["matching_kernels"] = {key: value for group in row["matching_kernels"] for key, value in group.items()}
            row["source_paths"] = sorted({kernel["source_path"] for kernel in targetKernels(metadata, row["reference_expression"]) if kernel.get("source_path")})
        return jsonify({"goal_id": goal_id, "targets": rows})

    @app.get("/api/targets/<target_id>")
    def target(target_id):
        configurations = store.getConfigurations(goal_id, target_id)
        if not configurations:
            return jsonify({"error": "Unknown target for this goal"}), 404
        expression = store.query("SELECT expression FROM kernel_bench.targets WHERE id = %s", (target_id,))[0]["expression"]
        metadata = store.getGoal(goal_id)["metadata"]
        kernels = targetKernels(metadata, expression)
        for row in configurations:
            row["configuration"]["decoded_input_constants"] = decodeConstants(row["configuration"])
        return jsonify({"id": target_id, "reference_expression": expression, "configurations": configurations,
                        "kernels": kernels, "source_paths": sorted({kernel["source_path"] for kernel in kernels if kernel.get("source_path")}),
                        "contract": CONTRACT, "enabled_capabilities": metadata.get("enabled_capabilities"),
                        "capabilities_status": "known" if "enabled_capabilities" in metadata else "unknown_legacy_snapshot",
                        "best_recorded_measurements": store.getBestMeasurements(goal_id, target_id)})

    @app.get("/api/kernels/list")
    def kernels():
        rows = store.getGoal(goal_id)["metadata"]["kernels"]
        name = request.args.get("name")
        if name is not None:
            rows = [kernel for kernel in rows if kernel["name"] == name]
        return jsonify({"kernels": rows, "name": name, "available": not rows if name else None,
                        "availability_reason": ("baseline_registration_conflict" if rows else "available") if name else "name_not_requested",
                        "uniqueness_scope": "goal baseline; candidate names may be reused across independent jobs"})

    @app.get("/api/linter/rules")
    def linterRules():
        return jsonify({"rules": getRules()})

    @app.get("/api/kernel-contract")
    def kernelContract():
        from .onboarding import SKELETONS
        return jsonify({**CONTRACT, "skeletons": SKELETONS,
                        "enabled_capabilities": store.getGoal(goal_id)["metadata"].get("enabled_capabilities")})

    def readArchive():
        row = store.query("SELECT source_archive FROM kernel_bench.goals WHERE id = %s", (goal_id,))[0]
        return zipfile.ZipFile(io.BytesIO(bytes(row["source_archive"])))

    @app.get("/api/source/list")
    @app.get("/api/core/list")
    def sources():
        with readArchive() as archive:
            return jsonify({"files": archive.namelist()})

    @app.get("/api/source")
    @app.get("/api/kernels/read_source")
    @app.get("/api/core/read_source")
    def source():
        path = request.args.get("path", "")
        with readArchive() as archive:
            if path not in archive.namelist():
                return jsonify({"error": "Unknown baseline source path"}), 404
            return jsonify({"path": path, "content": archive.read(path).decode()})

    @app.post("/api/iteration/lint")
    @app.post("/api/iteration/submit")
    @app.post("/api/kernels/test")
    def submit():
        data = request.get_json()
        if not isinstance(data, dict):
            raise TypeError("Expected a JSON object")
        for key in ("source", "target_id", "idea", "backend", "mode"):
            if key in data and not isinstance(data[key], str):
                raise TypeError(f"{key} must be a string")
        mode = "lint" if request.path == "/api/iteration/lint" else data.get("mode", "evaluate")
        job_id = createJob(store, goal_id, data.get("source", ""), data.get("target_id", ""), data.get("idea", ""), data.get("backend", "cuda"), mode)
        return jsonify({"job_id": job_id, "validation_id": job_id, "mode": mode, "status": "queued"}), 202

    @app.get("/api/jobs")
    @app.get("/api/history")
    @app.get("/api/versions")
    def jobs():
        target_id = request.args.get("target_id", "")
        rows = store.query("""SELECT id AS job_id, target_id, idea, kernel_name, backend, mode, status, step,
            result, error, error_details, created_at, completed_at FROM kernel_bench.jobs
            WHERE goal_id = %s AND (%s = '' OR target_id = %s) ORDER BY created_at DESC LIMIT 200""", (goal_id, target_id, target_id))
        return jsonify({"jobs": rows})

    def currentJob(job_id):
        uuid.UUID(job_id)
        row = store.getJob(job_id)
        return row if row and row["goal_id"] == goal_id else None

    @app.get("/api/jobs/<job_id>")
    def job(job_id):
        row = currentJob(job_id)
        if row is None:
            return jsonify({"error": "Unknown job for this goal"}), 404
        row["job_id"] = row.pop("id")
        return jsonify(row)

    @app.get("/api/jobs/<job_id>/logs")
    def logs(job_id):
        if currentJob(job_id) is None:
            return jsonify({"error": "Unknown job for this goal"}), 404
        clauses = ["job_id = %s"]
        parameters = [job_id]
        after_id = request.args.get("after_id")
        if after_id is not None:
            after_id = int(after_id)
            if after_id < 0:
                raise ValueError("after_id must be nonnegative")
            clauses.append("id > %s")
            parameters.append(after_id)
        stage = request.args.get("stage")
        if stage:
            if len(stage) > 100:
                raise ValueError("stage must be at most 100 characters")
            clauses.append("step = %s")
            parameters.append(stage)
        literal = request.args.get("grep")
        if literal is not None:
            if len(literal) > 200:
                raise ValueError("grep must be at most 200 characters")
            clauses.append("strpos(message, %s) > 0")
            parameters.append(literal)
        tail = request.args.get("tail")
        order = "ASC"
        limit = ""
        if tail is not None:
            tail = int(tail)
            if not 1 <= tail <= 1000:
                raise ValueError("tail must be between 1 and 1000")
            order = "DESC"
            limit = " LIMIT %s"
            parameters.append(tail)
        events = store.query("SELECT id, step, message, created_at FROM kernel_bench.events WHERE " +
                             " AND ".join(clauses) + f" ORDER BY id {order}" + limit, tuple(parameters))
        if tail is not None:
            events.reverse()
        return jsonify({"events": events, "next_after_id": events[-1]["id"] if events else after_id})

    @app.get("/api/jobs/<job_id>/artifacts")
    def artifacts(job_id):
        if currentJob(job_id) is None:
            return jsonify({"error": "Unknown job for this goal"}), 404
        rows = store.query("""SELECT id, name, media_type, octet_length(data) AS size_bytes, created_at
            FROM kernel_bench.artifacts WHERE job_id = %s ORDER BY name""", (job_id,))
        for row in rows:
            row["artifact_id"] = row.pop("id")
            row["url"] = f"/api/jobs/{job_id}/artifacts/{row['artifact_id']}"
        return jsonify({"artifacts": rows})

    @app.get("/api/jobs/<job_id>/artifacts/<artifact_id>")
    def artifact(job_id, artifact_id):
        if currentJob(job_id) is None:
            return jsonify({"error": "Unknown job for this goal"}), 404
        uuid.UUID(artifact_id)
        rows = store.query("SELECT name, media_type, data FROM kernel_bench.artifacts WHERE job_id = %s AND id = %s",
                           (job_id, artifact_id))
        if not rows:
            return jsonify({"error": "Unknown artifact for this job"}), 404
        row = rows[0]
        return send_file(io.BytesIO(bytes(row["data"])), mimetype=row["media_type"],
                         as_attachment=True, download_name=Path(row["name"]).name)


    @app.get("/api/benchmarks/records")
    def measurements():
        rows = store.query("""SELECT m.* FROM kernel_bench.measurements m JOIN kernel_bench.jobs j ON j.id = m.job_id
            WHERE j.goal_id = %s ORDER BY j.created_at DESC LIMIT 1000""", (goal_id,))
        return jsonify({"records": rows})

    def starterFiles():
        from .onboarding import starterFiles as makeStarterFiles
        with readArchive() as archive:
            return makeStarterFiles(archive)

    @app.get("/welcome")
    @app.get("/api/agent")
    @app.get("/api/agent/index")
    def agent():
        from .onboarding import INSTRUCTIONS
        base_url = request.url_root.rstrip("/")
        return jsonify({"scope": "kernel", "goal_id": goal_id, "tools": TOOLS,
                        "instructions": INSTRUCTIONS,
                        "authentication": {"header": "X-Bench-Token", "environment_variable": "BENCH_SECRET_TOKEN"},
                        "setup": {"base_url": base_url, "environment_variable": "BENCH_SERVER_URL",
                                  "download": base_url + "/api/starter.zip",
                                  "install": "uv sync",
                                  "client": "from agent import KernelBenchClient; client = KernelBenchClient()"},
                        "downloads": {name: base_url + "/api/starter/" + name
                                      for name in ("pyproject.toml", "agent.py", "AGENTS.md", ".env.example")},
                        "hardware": {"status": "/api/status", "files": "/api/source/list",
                                     "source_prefix": "docs/hwinfo/box/"}})

    @app.get("/api/starter.zip")
    def starterArchive():
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, content in starterFiles().items():
                archive.writestr(name, content)
        output.seek(0)
        return send_file(output, mimetype="application/zip", as_attachment=True, download_name="kernel-bench-starter.zip")

    @app.get("/api/starter/<path:name>")
    def starterFile(name):
        files = starterFiles()
        if name not in files:
            return jsonify({"error": "Unknown starter file"}), 404
        return send_file(io.BytesIO(files[name]), mimetype="text/plain", as_attachment=True,
                         download_name=Path(name).name)

    @app.get("/api/tools")
    def tools():
        return jsonify({"tools": TOOLS})

    @app.route("/api/<kind>", methods=["GET", "POST"])
    def feedback(kind):
        if kind not in ("reports", "suggestions"):
            return jsonify({"error": "Unknown endpoint"}), 404
        singular = "report" if kind == "reports" else "suggestion"
        if request.method == "POST":
            data = request.get_json()
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            store.execute("INSERT INTO kernel_bench.feedback(kind, data) VALUES (%s, %s)", (singular, Jsonb(data)))
            return jsonify({"status": "saved"}), 201
        return jsonify({kind: store.query("SELECT * FROM kernel_bench.feedback WHERE kind = %s ORDER BY id DESC LIMIT 200", (singular,))})

    if start_worker:
        app.extensions["kernel_bench_worker"] = startWorker(store, stop_event)
        app.extensions["kernel_bench_validation_worker"] = startWorker(store, stop_event, validation=True)
    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="KernelBench kernel-level server")
    parser.add_argument("--goal-cache", type=Path)
    parser.add_argument("--host", default=os.environ.get("BENCH_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BENCH_PORT", "8080")))
    args = parser.parse_args()
    serveApp(args)


def serveApp(args):
    from waitress import create_server

    stop_event = threading.Event()

    def requestStop(signum, frame):
        stop_event.set()

    previous_handlers = {signum: signal.signal(signum, requestStop) for signum in (signal.SIGTERM, signal.SIGINT)}
    app = None
    server = None
    server_thread = None
    try:
        # Install handlers before discovery: a stop during startup waits for its native process too.
        app = createApp(goal_cache=args.goal_cache, stop_event=stop_event)
        if not stop_event.is_set():
            server = create_server(app, host=args.host, port=args.port, threads=4,
                                   trusted_proxy="127.0.0.1" if os.environ.get("BENCH_TRUST_PROXY") == "1" else None,
                                   trusted_proxy_headers={"x-forwarded-proto", "x-forwarded-host"}
                                   if os.environ.get("BENCH_TRUST_PROXY") == "1" else set())
            server_thread = threading.Thread(target=server.run, name="kernel-bench-http", daemon=True)
            server_thread.start()
            while not stop_event.wait(1):
                if not server_thread.is_alive() or not app.extensions["kernel_bench_worker"].is_alive() or not app.extensions["kernel_bench_validation_worker"].is_alive():
                    raise RuntimeError("KernelBench HTTP server or worker exited unexpectedly")
    finally:
        stop_event.set()
        logger.info("Draining KernelBench; waiting for active startup/job processes")
        if app is not None:
            app.extensions["kernel_bench_worker"].join()
            app.extensions["kernel_bench_validation_worker"].join()
        if server is not None:
            server.task_dispatcher.shutdown()
            server.asyncore.close_all(map=server._map)
        if server_thread is not None:
            server_thread.join()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
