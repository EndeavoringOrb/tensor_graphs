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
        rows = self.query("SELECT id, name, fingerprint, cache_sha256, metadata, created_at FROM kernel_bench.goals WHERE id = %s", (goal_id,))
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

    def nextJob(self):
        rows = self.query("""
            UPDATE kernel_bench.jobs SET status = 'running', started_at = now(), error = NULL, error_details = NULL
            WHERE id = (SELECT id FROM kernel_bench.jobs WHERE status = 'queued'
                        ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
            RETURNING *
        """)
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
