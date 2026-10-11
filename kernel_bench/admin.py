"""Operator-only selection of immutable discovery snapshots."""
import json
import uuid
from pathlib import Path

from psycopg.types.json import Jsonb

import hmac
import os

from flask import jsonify, render_template, request

from .promotion import enqueueGoalValidation, promotionConfig


def installAdmin(app, store):
    @app.get("/admin")
    def adminPage():
        return render_template("admin.html")

    @app.post("/api/admin/goals/prepare")
    @app.get("/api/admin/goals")
    @app.post("/api/admin/goals/activate")
    def adminGoals():
        token = os.environ.get("KERNEL_BENCH_ADMIN_TOKEN", "")
        candidate = request.headers.get("X-Admin-Token", "")
        if not token or not candidate or not hmac.compare_digest(token, candidate):
            return jsonify({"error": "Operator authentication required: X-Admin-Token"}), 403
        catalog_path = os.environ.get("KERNEL_BENCH_GOAL_CATALOG")
        catalog = json.loads(Path(catalog_path).read_text()) if catalog_path else {}
        if request.method == "GET":
            return jsonify({"presets": [{"id": key, "name": value["name"]} for key, value in catalog.items()],"active_goal_id": app.config["KERNEL_BENCH_GOAL_ID"], "goals": store.query(
                "SELECT id, name, definition_id, COALESCE(metadata->'promotion_recipe' <> 'null'::jsonb, false) AS promotion_enabled, created_at FROM kernel_bench.goals ORDER BY id DESC")})
        data = request.get_json()
        if request.path.endswith("/prepare"):
            preset_id = data.get("preset_id") if isinstance(data, dict) else None
            if not isinstance(preset_id, str) or preset_id not in catalog:
                raise ValueError("Select an operator-configured goal preset")
            preset = catalog[preset_id]
            cache_path = str(Path(preset["cache_path"]).resolve(strict=True))
            recipe = promotionConfig(preset["promotion_config"])
            active_id = app.config["KERNEL_BENCH_GOAL_ID"]
            targets = store.getTargets(active_id)
            job_id = uuid.uuid4()
            store.execute("""INSERT INTO kernel_bench.jobs(id, goal_id, target_id, idea, source, kernel_name, backend, mode, result)
                VALUES (%s, %s, %s, 'Prepare operator goal preset', '', %s, 'cpu', 'goal_setup', %s)""",
                (job_id, active_id, targets[0]["id"], preset["name"], Jsonb({"preset_id": preset_id, "cache_path": cache_path, "recipe": recipe})))
            return jsonify({"job_id": str(job_id)}), 202
        goal_id = data.get("goal_id") if isinstance(data, dict) else None
        if isinstance(goal_id, bool) or not isinstance(goal_id, int):
            raise ValueError("goal_id must be an integer")
        if store.getGoal(goal_id) is None:
            return jsonify({"error": "Unknown discovery snapshot"}), 404
        # This changes routing only; all model validation stays in the shared queue.
        job_id = enqueueGoalValidation(store, goal_id)
        store.execute("INSERT INTO kernel_bench.active_goal(singleton, goal_id) VALUES (true, %s) ON CONFLICT(singleton) DO UPDATE SET goal_id = EXCLUDED.goal_id, updated_at = now()", (goal_id,))
        app.config["KERNEL_BENCH_GOAL_ID"] = goal_id
        return jsonify({"goal_id": goal_id, "validation_job_id": job_id})


def prepareGoal(store, job):
    from .jobs import initializeGoal
    config = job["result"]
    goal_id = initializeGoal(store, config["cache_path"], promotion_recipe=config["recipe"], already_locked=True)
    validation_id = enqueueGoalValidation(store, goal_id)
    store.execute("INSERT INTO kernel_bench.active_goal(singleton, goal_id) VALUES (true, %s) ON CONFLICT(singleton) DO UPDATE SET goal_id = EXCLUDED.goal_id, updated_at = now()", (goal_id,))
    store.completeJob(job["id"], {**config, "goal_id": goal_id, "validation_job_id": validation_id, "measurements": []})
