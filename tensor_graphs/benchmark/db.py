import sqlite3
import json
import uuid
import os
from datetime import datetime
from typing import Optional, List, Dict, Any


class BenchmarkDB:
    def __init__(self, db_path: str = "benchmarks.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # A. The Math (Definitions)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS canonical_graphs (
                    id TEXT PRIMARY KEY,
                    human_name TEXT,
                    structural_hash TEXT UNIQUE,
                    atomic_graph_json TEXT
                )
            """
            )

            # B. The Code (Solutions)
            # UPDATED: Added recipe_json column
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS implementations (
                    id TEXT PRIMARY KEY,
                    canonical_graph_id TEXT,
                    type TEXT, -- KERNEL, GRAPH_RECIPE
                    name TEXT,
                    backend TEXT,
                    source_hash TEXT,
                    recipe_json TEXT, -- Serialized assignments {node_name: backend}
                    requirements TEXT, -- JSON
                    FOREIGN KEY (canonical_graph_id) REFERENCES canonical_graphs (id)
                )
            """
            )

            # C. The Context (Environment)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS environments (
                    id TEXT PRIMARY KEY,
                    hardware_name TEXT,
                    memory_bytes INTEGER,
                    platform_info TEXT, -- JSON
                    libs_info TEXT -- JSON
                )
            """
            )

            # D. The Input (Workload)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS workloads (
                    id TEXT PRIMARY KEY,
                    canonical_graph_id TEXT,
                    axes_hash TEXT,
                    axes_json TEXT, -- JSON
                    input_descriptors TEXT, -- JSON
                    FOREIGN KEY (canonical_graph_id) REFERENCES canonical_graphs (id)
                )
            """
            )

            # E. The Stats (Trace)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_traces (
                    id TEXT PRIMARY KEY,
                    implementation_id TEXT,
                    workload_id TEXT,
                    environment_id TEXT,
                    status TEXT, -- PASSED, INCORRECT_NUMERICAL, RUNTIME_ERROR, etc.
                    latency_ms REAL,
                    speedup_factor REAL,
                    max_relative_error REAL,
                    timestamp TEXT,
                    full_log TEXT,
                    FOREIGN KEY (implementation_id) REFERENCES implementations (id),
                    FOREIGN KEY (workload_id) REFERENCES workloads (id),
                    FOREIGN KEY (environment_id) REFERENCES environments (id)
                )
            """
            )
            conn.commit()

    def add_canonical_graph(
        self,
        structural_hash: str,
        human_name: Optional[str] = None,
        atomic_graph_json: Optional[str] = None,
    ) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM canonical_graphs WHERE structural_hash = ?",
                (structural_hash,),
            )
            row = cursor.fetchone()
            if row:
                return row[0]

            graph_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO canonical_graphs (id, human_name, structural_hash, atomic_graph_json) VALUES (?, ?, ?, ?)",
                (graph_id, human_name, structural_hash, atomic_graph_json),
            )
            conn.commit()
            return graph_id

    def add_implementation(
        self,
        canonical_graph_id: str,
        impl_type: str,
        name: str,
        backend: str,
        source_hash: str,
        requirements: Optional[Dict[str, Any]] = None,
        recipe_json: Optional[str] = None,  # UPDATED: Accept recipe_json
    ) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check for existing implementation
            cursor.execute(
                "SELECT id FROM implementations WHERE canonical_graph_id = ? AND type = ? AND name = ?",
                (canonical_graph_id, impl_type, name),
            )
            row = cursor.fetchone()
            if row:
                return row[0]

            impl_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO implementations (id, canonical_graph_id, type, name, backend, source_hash, requirements, recipe_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    impl_id,
                    canonical_graph_id,
                    impl_type,
                    name,
                    backend,
                    source_hash,
                    json.dumps(requirements or {}),
                    recipe_json,
                ),
            )
            conn.commit()
            return impl_id

    def add_environment(
        self,
        hardware_name: str,
        memory_bytes: int,
        platform_info: Dict[str, Any],
        libs_info: Dict[str, Any],
    ) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Simple dedupe based on hardware name (MVP)
            cursor.execute(
                "SELECT id FROM environments WHERE hardware_name = ?", (hardware_name,)
            )
            row = cursor.fetchone()
            if row:
                return row[0]

            env_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO environments (id, hardware_name, memory_bytes, platform_info, libs_info) VALUES (?, ?, ?, ?, ?)",
                (
                    env_id,
                    hardware_name,
                    memory_bytes,
                    json.dumps(platform_info),
                    json.dumps(libs_info),
                ),
            )
            conn.commit()
            return env_id

    def add_workload(
        self,
        canonical_graph_id: str,
        axes_hash: str,
        axes_json: Dict[str, Any],
        input_descriptors: Optional[Dict[str, Any]] = None,
    ) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM workloads WHERE canonical_graph_id = ? AND axes_hash = ?",
                (canonical_graph_id, axes_hash),
            )
            row = cursor.fetchone()
            if row:
                return row[0]

            workload_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO workloads (id, canonical_graph_id, axes_hash, axes_json, input_descriptors) VALUES (?, ?, ?, ?, ?)",
                (
                    workload_id,
                    canonical_graph_id,
                    axes_hash,
                    json.dumps(axes_json),
                    json.dumps(input_descriptors or {}),
                ),
            )
            conn.commit()
            return workload_id

    def add_benchmark_trace(
        self,
        implementation_id: str,
        workload_id: str,
        environment_id: str,
        status: str,
        latency_ms: float,
        max_relative_error: float = 0.0,
        speedup_factor: float = 1.0,
        full_log: str = "",
    ) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            trace_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            cursor.execute(
                """INSERT INTO benchmark_traces 
                   (id, implementation_id, workload_id, environment_id, status, latency_ms, speedup_factor, max_relative_error, timestamp, full_log) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace_id,
                    implementation_id,
                    workload_id,
                    environment_id,
                    status,
                    latency_ms,
                    speedup_factor,
                    max_relative_error,
                    timestamp,
                    full_log,
                ),
            )
            conn.commit()
            return trace_id

    def get_best_implementation(
        self, structural_hash: str, workload_axes_hash: str, environment_id: str
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Join traces with implementations and workloads to find the fastest PASSED implementation
            query = """
                SELECT i.*, t.latency_ms, t.max_relative_error
                FROM implementations i
                JOIN canonical_graphs g ON i.canonical_graph_id = g.id
                JOIN workloads w ON w.canonical_graph_id = g.id
                JOIN benchmark_traces t ON t.implementation_id = i.id AND t.workload_id = w.id
                WHERE g.structural_hash = ? 
                  AND w.axes_hash = ?
                  AND t.environment_id = ?
                  AND t.status = 'PASSED'
                ORDER BY t.latency_ms ASC
                LIMIT 1
            """
            cursor.execute(query, (structural_hash, workload_axes_hash, environment_id))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_op_preference(
        self, op_type: str, shape_str: str, env_id: str
    ) -> Optional[str]:
        """
        Returns 'KERNEL' or 'GRAPH_RECIPE' (Atomic) based on what was fastest for this op_type.
        """
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT i.type, t.latency_ms
                FROM implementations i
                JOIN canonical_graphs g ON i.canonical_graph_id = g.id
                JOIN benchmark_traces t ON t.implementation_id = i.id
                WHERE g.human_name = ?
                  AND t.environment_id = ?
                  AND t.status = 'PASSED'
                ORDER BY t.latency_ms ASC
                LIMIT 1
            """
            cursor.execute(query, (op_type, env_id))
            row = cursor.fetchone()
            if row:
                return row["type"]
            return None
