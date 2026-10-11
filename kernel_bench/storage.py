"""PostgreSQL is the authoritative store; files in workers are disposable."""
import hashlib
import json
import uuid
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def fingerprint(data):
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class Store:
    def __init__(self, database_url):
        if not database_url:
            raise ValueError("Set KERNEL_BENCH_DATABASE_URL to an authenticated PostgreSQL connection")
        self.database_url = database_url

    @contextmanager
    def connection(self):
        with psycopg.connect(self.database_url, row_factory=dict_row, connect_timeout=10) as connection:
            yield connection

    def initialize(self):
        with self.connection() as connection:
            # Serialize first-time DDL across server processes.
            connection.execute("SELECT pg_advisory_xact_lock(724189001)")
            connection.execute(Path(__file__).with_name("schema.sql").read_text())

    def query(self, sql, parameters=()):
        with self.connection() as connection:
            return connection.execute(sql, parameters).fetchall()

    def execute(self, sql, parameters=()):
        with self.connection() as connection:
            connection.execute(sql, parameters)

    def getGoal(self, goal_id):
        rows = self.query("SELECT id, name, definition_id, fingerprint, cache_sha256, metadata, created_at FROM kernel_bench.goals WHERE id = %s", (goal_id,))
        if not rows:
            raise ValueError("Unknown goal")
        return rows[0]

    def getConfigurations(self, goal_id, target_id):
        return self.query("SELECT * FROM kernel_bench.configurations WHERE goal_id = %s AND target_id = %s ORDER BY id", (goal_id, target_id))

    def getBestMeasurements(self, goal_id, target_id):
        return self.query("""
            SELECT DISTINCT ON (m.configuration_id) m.configuration_id, m.kernel_id,
                   m.kernel_name, m.median_ms, m.job_id, j.completed_at
            FROM kernel_bench.measurements m
            JOIN kernel_bench.jobs j ON j.id = m.job_id
            WHERE j.goal_id = %s AND j.target_id = %s AND j.status = 'completed'
            ORDER BY m.configuration_id, m.median_ms, j.completed_at DESC
        """, (goal_id, target_id))

    def getTargets(self, goal_id):
        return self.query("""
            SELECT t.id, t.expression AS reference_expression, count(*) AS configuration_count,
                   jsonb_agg(DISTINCT c.matching_kernels) AS matching_kernels
            FROM kernel_bench.configurations c JOIN kernel_bench.targets t ON t.id = c.target_id
            WHERE c.goal_id = %s GROUP BY t.id, t.expression ORDER BY t.expression
        """, (goal_id,))

    def getJob(self, job_id):
        rows = self.query("SELECT * FROM kernel_bench.jobs WHERE id = %s", (job_id,))
        return rows[0] if rows else None

    def addEvent(self, job_id, step, message):
        with self.connection() as connection:
            connection.execute("UPDATE kernel_bench.jobs SET step = %s WHERE id = %s", (step, job_id))
            connection.execute("INSERT INTO kernel_bench.events(job_id, step, message) VALUES (%s, %s, %s)", (job_id, step, message))

    def nextJob(self, validation=False):
        rows = self.query("""
            UPDATE kernel_bench.jobs SET status = 'running', started_at = now(), error = NULL, error_details = NULL
            WHERE id = (SELECT id FROM kernel_bench.jobs WHERE status = 'queued'
                        ORDER BY created_at, id FOR UPDATE SKIP LOCKED LIMIT 1)
            RETURNING *
        """, ())
        return rows[0] if rows else None

    def completeJob(self, job_id, result):
        # Measurements and completion become visible together.
        with self.connection() as connection:
            for measurement in result["measurements"]:
                connection.execute("""
                    INSERT INTO kernel_bench.measurements
                    (job_id, configuration_id, kernel_id, kernel_name, is_candidate, samples_ms, median_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (job_id, measurement["configuration_id"], str(measurement["kernel_id"]),
                      measurement["kernel_name"], measurement["is_candidate"],
                      measurement["samples_ms"], measurement["median_ms"]))
            connection.execute("""UPDATE kernel_bench.jobs SET status = 'completed', step = 'completed',
                result = %s, completed_at = now() WHERE id = %s""", (Jsonb(result), job_id))

    def addArtifact(self, job_id, name, data, media_type="text/plain"):
        rows = self.query("""INSERT INTO kernel_bench.artifacts(id, job_id, name, media_type, data)
            VALUES (%s, %s, %s, %s, %s) ON CONFLICT (job_id, name)
            DO UPDATE SET data = EXCLUDED.data, media_type = EXCLUDED.media_type
            RETURNING id""", (uuid.uuid4(), job_id, name, media_type, data))
        return str(rows[0]["id"])

    def failJob(self, job_id, error, error_details=None, result=None):
        self.execute("""UPDATE kernel_bench.jobs SET status = 'failed', error = %s,
            error_details = %s, result = %s, completed_at = now() WHERE id = %s""",
            (error, Jsonb(error_details or {}), Jsonb(result or {}), job_id))

    def queueInfo(self, goal_id, job=None):
        # Queues are server-wide, matching worker scheduling across all goals.
        rows = self.query("""SELECT mode, count(*) FILTER (WHERE status = 'queued') AS queued,
            count(*) FILTER (WHERE status = 'running') AS running FROM kernel_bench.jobs
            WHERE status IN ('queued', 'running') GROUP BY mode""")
        depths = {mode: {"queued": 0, "running": 0} for mode in ("lint", "compile", "evaluate", "promote", "goal_setup")}
        for row in rows:
            depths[row["mode"]] = {key: row[key] for key in ("queued", "running")}
        statistics = self.query("""SELECT mode, stage, count(*) AS sample_count,
            avg(seconds) AS mean_seconds, min(seconds) AS minimum_seconds, max(seconds) AS maximum_seconds
            FROM (SELECT * FROM kernel_bench.stage_durations ORDER BY created_at DESC LIMIT 1000) recent
            GROUP BY mode, stage ORDER BY mode, stage""")
        result = {"depth_by_mode": depths, "workers": self.query("""SELECT w.*, COALESCE(j.step, w.stage) AS running_stage, j.stage_started_at,
                      w.heartbeat_at < now() - interval '30 seconds' AS stale
                      FROM kernel_bench.workers w LEFT JOIN kernel_bench.jobs j ON j.id = w.job_id"""),
                  "ordering": "created_at ASC, id ASC in one shared queue; calibration runs only when the queue is idle",
                  "recent_stage_durations": statistics, "estimated_completion_range_seconds": None}
        if job is not None:
            position = None
            if job["status"] == "queued":
                position = self.query("""SELECT count(*) + 1 AS position FROM kernel_bench.jobs
                    WHERE status = 'queued'
                    AND (created_at, id) < (%s, %s)""", (job["created_at"], job["id"]))[0]["position"]
            result["position"] = position
        if (job is None or job["status"] in ("queued", "running")) and not any(worker["stage"] == "baseline" for worker in result["workers"]):
            history = self.query("""SELECT mode, count(*) AS count, min(seconds) AS low, max(seconds) AS high
                FROM (SELECT mode, EXTRACT(EPOCH FROM completed_at - started_at) AS seconds
                      FROM kernel_bench.jobs WHERE status = 'completed' AND started_at IS NOT NULL
                      ORDER BY completed_at DESC LIMIT 100) recent GROUP BY mode""")
            by_mode = {row["mode"]: row for row in history if row["count"] >= 5}
            active_modes = [mode for mode, counts in depths.items() if counts["queued"] + counts["running"]]
            if all(mode in by_mode for mode in active_modes):
                # Historical estimate includes both lanes because heavy builds share the hardware lock.
                upper = sum((depths[mode]["queued"] + depths[mode]["running"]) * float(by_mode[mode]["high"]) for mode in active_modes)
                result["estimated_completion_range_seconds"] = [0, upper]
                result["estimate_method"] = "0 to sum of active jobs times recent observed maximum duration per mode; at least 5 completed samples per active mode; not a guarantee"
        return result

    def getBaselines(self, goal_id, target_id):
        return self.query("""SELECT c.id AS configuration_id, CASE WHEN b.status = 'running' THEN 'pending' ELSE COALESCE(b.status, 'pending') END AS status,
            COALESCE(b.status = 'running', false) AS calibrating,
            b.measurements, b.winner, b.identity, b.error, b.measured_at
            FROM kernel_bench.configurations c LEFT JOIN kernel_bench.baselines b ON b.configuration_id = c.id
            WHERE c.goal_id = %s AND c.target_id = %s ORDER BY c.id""", (goal_id, target_id))
