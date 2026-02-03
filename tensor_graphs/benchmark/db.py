import sqlite3
import json
import math
import uuid
from typing import Optional, Dict, Any, List
from ..ir.dtypes import DType
from ..ir.graph import GraphEncoder


class BenchmarkDB:
    def __init__(self, db_path: str = "benchmarks.db"):
        self.db_path = db_path
        self._init_db()

    def add_environment(
        self,
        hardware_name: str,
        memory_bytes: int,
        platform_info: str,
        libs_info: str,
    ) -> str:
        import uuid
        import datetime

        env_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO kernel_benchmarks (id, op_type, backend, dtype, shape_json, attrs_json, latency_ms, timestamp)
                VALUES (?, 'ENVIRONMENT', 'SYSTEM', 'INFO', ?, ?, ?, ?)
            """,
                (
                    env_id,
                    json.dumps(
                        {
                            "hardware_name": hardware_name,
                            "memory_bytes": memory_bytes,
                            "platform_info": platform_info,
                            "libs_info": libs_info,
                        }
                    ),
                    None,
                    datetime.datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        return env_id

    def add_canonical_graph(self, structural_hash: str) -> str:
        import uuid
        import datetime

        graph_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO kernel_benchmarks (id, op_type, backend, dtype, shape_json, attrs_json, latency_ms, timestamp)
                VALUES (?, 'GRAPH', 'CANONICAL', 'METADATA', ?, ?, ?, ?)
            """,
                (
                    graph_id,
                    json.dumps({"structural_hash": structural_hash}),
                    None,
                    datetime.datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        return graph_id

    def add_implementation(
        self,
        graph_id: str,
        impl_type: str,
        impl_name: str,
        backend: str,
        source_hash: str,
        requirements: Dict[str, Any],
    ) -> str:
        import uuid
        import datetime

        impl_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO kernel_benchmarks (id, op_type, backend, dtype, shape_json, attrs_json, latency_ms, timestamp)
                VALUES (?, 'IMPLEMENTATION', ?, 'METADATA', ?, ?, ?, ?)
            """,
                (
                    impl_id,
                    backend,
                    json.dumps(
                        {
                            "graph_id": graph_id,
                            "impl_type": impl_type,
                            "impl_name": impl_name,
                            "source_hash": source_hash,
                            "requirements": requirements,
                        }
                    ),
                    None,
                    datetime.datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        return impl_id

    def add_workload(
        self, graph_id: str, workload_axes_hash: str, axes_json: Dict[str, Any]
    ) -> str:
        import uuid
        import datetime

        workload_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO kernel_benchmarks (id, op_type, backend, dtype, shape_json, attrs_json, latency_ms, timestamp)
                VALUES (?, 'WORKLOAD', 'METADATA', 'METADATA', ?, ?, ?, ?)
            """,
                (
                    workload_id,
                    json.dumps(
                        {
                            "graph_id": graph_id,
                            "workload_axes_hash": workload_axes_hash,
                            "axes_json": axes_json,
                        }
                    ),
                    None,
                    datetime.datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        return workload_id

    def add_benchmark_trace(
        self,
        impl_id: str,
        workload_id: str,
        env_id: str,
        status: str,
        latency_ms: float,
    ) -> None:
        import datetime

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO kernel_benchmarks (id, op_type, backend, dtype, shape_json, attrs_json, latency_ms, timestamp)
                VALUES (?, 'TRACE', 'METADATA', 'METADATA', ?, ?, ?, ?)
            """,
                (
                    str(uuid.uuid4()),
                    json.dumps(
                        {
                            "impl_id": impl_id,
                            "workload_id": workload_id,
                            "env_id": env_id,
                            "status": status,
                        }
                    ),
                    latency_ms,
                    datetime.datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Kernel Benchmarks Table
            # Stores raw timing for specific op/backend/shape configurations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kernel_benchmarks (
                    id TEXT PRIMARY KEY,
                    op_type TEXT,
                    backend TEXT,
                    dtype TEXT,
                    shape_json TEXT, -- List of ints
                    attrs_json TEXT, -- Dict
                    latency_ms REAL,
                    timestamp TEXT
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_op_backend ON kernel_benchmarks(op_type, backend)"
            )
            conn.commit()

    def _serialize_attrs(self, attrs: Optional[Dict[str, Any]]) -> str:
        """Standardized serialization for attributes to ensure DB consistency."""
        if not attrs:
            return json.dumps({}, sort_keys=True)
        return json.dumps(attrs, cls=GraphEncoder, sort_keys=True)

    def add_benchmark(
        self,
        op_type: str,
        backend: str,
        dtype: str,
        shape: list,
        attrs: dict,
        latency_ms: float,
    ):
        import datetime

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO kernel_benchmarks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    op_type,
                    backend,
                    dtype,
                    json.dumps(shape),
                    self._serialize_attrs(attrs),
                    latency_ms,
                    datetime.datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def estimate_latency(
        self,
        op_type: str,
        backend: str,
        dtype: str,
        shape: tuple,
        attrs: Optional[dict] = None,
    ) -> Optional[float]:
        """
        Estimates latency.
        1. Exact Match
        2. Nearest Neighbor (Log-Space Euclidean Distance on Shape) + Complexity Scaling
        """
        shape_list = list(shape) if shape else []
        attrs_json = self._serialize_attrs(attrs)

        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 1. Exact Match
            cursor.execute(
                """
                SELECT latency_ms FROM kernel_benchmarks 
                WHERE op_type=? AND backend=? AND dtype=? AND shape_json=? AND attrs_json=?
                ORDER BY timestamp DESC LIMIT 1
            """,
                (
                    op_type,
                    backend,
                    dtype,
                    json.dumps(shape_list, cls=GraphEncoder),
                    attrs_json,
                ),
            )
            row = cursor.fetchone()
            if row:
                return row["latency_ms"]

            # 2. Heuristic Interpolation (Fallback)
            # Fetch all entries for this op/backend/dtype
            cursor.execute(
                """
                SELECT shape_json, latency_ms FROM kernel_benchmarks 
                WHERE op_type=? AND backend=? AND dtype=?
            """,
                (op_type, backend, dtype),
            )

            rows = cursor.fetchall()
            if not rows:
                return None

            best_dist = float("inf")
            best_ref = None

            target_vol = math.prod(x for x in shape_list if x is not None) or 1
            target_log_dims = [math.log(x if x else 1) for x in shape_list]

            for r in rows:
                ref_shape = json.loads(r["shape_json"])
                ref_latency = r["latency_ms"]

                # Check rank match
                if len(ref_shape) != len(shape_list):
                    continue

                # Distance
                ref_log_dims = [math.log(x if x else 1) for x in ref_shape]
                dist = sum((a - b) ** 2 for a, b in zip(target_log_dims, ref_log_dims))

                if dist < best_dist:
                    best_dist = dist
                    best_ref = (ref_shape, ref_latency)

            if best_ref:
                ref_shape, ref_latency = best_ref
                ref_vol = math.prod(x if x else 1 for x in ref_shape) or 1
                # Linear scaling by volume (FLOPs/Bytes proxy)
                return ref_latency * (target_vol / ref_vol)

            return None
