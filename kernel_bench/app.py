"""Authenticated kernel-level API. Start with python -m kernel_bench.app."""
import argparse
import hmac
import io
import logging
import os
import uuid
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, send_file
from psycopg.types.json import Jsonb

from .jobs import (
    DEFAULT_GOAL_CACHE,
    PROJECT_ROOT,
    createJob,
    initializeGoal,
    startWorker,
)
from .storage import Store

TOOLS = [
    {"name": "get_status", "method": "GET", "path": "/api/status", "description": "Goal identity and kernel-level scope"},
    {"name": "list_targets", "method": "GET", "path": "/api/targets", "description": "All reference expressions matching the saturated goal graph"},
    {"name": "get_target", "method": "GET", "path": "/api/targets/{target_id}", "description": "Exact workload configurations, constants, strides, and matching kernels"},
    {"name": "read_source", "method": "GET", "path": "/api/source?path=...", "description": "Read baseline kernel/core/reference/hardware source snapshot"},
    {"name": "submit_iteration", "method": "POST", "path": "/api/iteration/submit", "description": "Submit target_id, source, backend (cpu/cuda), and idea; one REGISTER_KERNEL"},
    {"name": "get_job", "method": "GET", "path": "/api/jobs/{job_id}", "description": "Correctness coverage, measurements, and per-configuration speedups"},
    {"name": "get_job_logs", "method": "GET", "path": "/api/jobs/{job_id}/logs", "description": "Build/test/benchmark diagnostics from PostgreSQL"},
    {"name": "list_jobs", "method": "GET", "path": "/api/jobs?target_id=...", "description": "Shared trial history; targets have no reservations"},
]


def createApp(database_url=None, goal_cache=None, token=None, start_worker=True, goal_id=None):
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
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    app.extensions["kernel_bench_store"] = store
    app.config["KERNEL_BENCH_GOAL_ID"] = goal_id

    def authorized(candidate):
        return bool(candidate) and hmac.compare_digest(candidate, token)

    @app.before_request
    def authenticate():
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
        return jsonify({"error": str(error)}), 400

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
        return jsonify({"goal_id": goal_id, "targets": rows})

    @app.get("/api/targets/<target_id>")
    def target(target_id):
        configurations = store.getConfigurations(goal_id, target_id)
        if not configurations:
            return jsonify({"error": "Unknown target for this goal"}), 404
        expression = store.query("SELECT expression FROM kernel_bench.targets WHERE id = %s", (target_id,))[0]["expression"]
        kernels = [kernel for kernel in store.getGoal(goal_id)["metadata"]["kernels"] if kernel["reference_expression"] == expression]
        return jsonify({"id": target_id, "reference_expression": expression, "configurations": configurations,
                        "kernels": kernels, "best_recorded_measurements": store.getBestMeasurements(goal_id, target_id)})

    @app.get("/api/kernels/list")
    def kernels():
        return jsonify({"kernels": store.getGoal(goal_id)["metadata"]["kernels"]})

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

    @app.post("/api/iteration/submit")
    @app.post("/api/kernels/test")
    def submit():
        data = request.get_json()
        if not isinstance(data, dict):
            raise TypeError("Expected a JSON object")
        for key in ("source", "target_id", "idea", "backend"):
            if key in data and not isinstance(data[key], str):
                raise TypeError(f"{key} must be a string")
        job_id = createJob(store, goal_id, data.get("source", ""), data.get("target_id", ""), data.get("idea", ""), data.get("backend", "cuda"))
        return jsonify({"job_id": job_id, "status": "queued"}), 202

    @app.get("/api/jobs")
    @app.get("/api/history")
    @app.get("/api/versions")
    def jobs():
        target_id = request.args.get("target_id", "")
        rows = store.query("""SELECT id AS job_id, target_id, idea, kernel_name, backend, status, step,
            result, error, created_at, completed_at FROM kernel_bench.jobs
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
        return jsonify({"events": store.query("SELECT id, step, message, created_at FROM kernel_bench.events WHERE job_id = %s ORDER BY id", (job_id,))})

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
        app.extensions["kernel_bench_stop"] = startWorker(store)
    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="KernelBench kernel-level server")
    parser.add_argument("--goal-cache", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app = createApp(goal_cache=args.goal_cache)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
