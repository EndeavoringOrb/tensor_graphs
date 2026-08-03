import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

from algos.bufferize import bufferize
from algos.core import Buffer, Node
from algos.iter_dispatch import graphs, iter_dispatch_orders
from algos.malloc import malloc
from flask import Flask, jsonify, render_template
from pydantic import TypeAdapter

# Include root directory in sys.path to access utils package
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.analyze_performance import load_uids_from_cpp
from utils.binary import load_cache_file

app = Flask(__name__)

buffer_adapter = TypeAdapter(Buffer)
node_adapter = TypeAdapter(Node)


def discover_compiled_graphs() -> dict[str, str]:
    """Finds all compiled .bin cache files in dirty_region_caches and benchmarks."""
    compiled_files = {}

    search_dirs = [
        ROOT_DIR / "dirty_region_caches",
        ROOT_DIR / "benchmarks",
    ]

    for s_dir in search_dirs:
        if s_dir.exists():
            for bin_path in s_dir.glob("*.bin"):
                if bin_path.name in ("calls.bin", "records.bin", "bucket_counts.bin"):
                    continue
                display_name = f"[Compiled] {bin_path.name}"
                compiled_files[display_name] = str(bin_path)

    return compiled_files

def load_compiled_graph_orders(filepath: str) -> list[dict]:
    """Converts a compiled graph binary cache file into the visualization schedule format."""
    if not os.path.exists(filepath):
        return []

    cache_entries = load_cache_file(filepath, string_enums=True)
    uid_map = load_uids_from_cpp()

    all_orders = []

    for entry in cache_entries:
        if entry.get("type") != "compiled_bucket":
            continue

        graph = entry["graph"]
        instructions = graph.get("instructions", [])
        node_costs = graph.get("nodeCosts", {})

        ordered_nodes = []
        buffers_map = {}
        allocated_map = {}

        eclass_finish_time = {}
        engine_finish_time = {}

        for idx, inst in enumerate(instructions):
            eclass_id = inst.get("eclassId", idx)
            cost = node_costs.get(eclass_id, 1.0)
            if cost == float("inf") or cost <= 0:
                cost = 1.0
            cost = round(float(cost), 4)

            kernel_id = inst.get("kernelId", 0)
            info = (
                uid_map.get(kernel_id)
                or uid_map.get(str(kernel_id))
                or uid_map.get(hex(kernel_id).lower())
            )

            if isinstance(info, dict):
                k_name = info.get("name", f"Kernel_{hex(kernel_id)}")
            elif isinstance(info, str):
                k_name = info
            else:
                k_name = f"Kernel_{hex(kernel_id)}"

            # Clean operation name
            op_clean = k_name
            if "_" in k_name:
                parts = k_name.split("_")
                op_clean = parts[0].upper()

            node_name = f"Inst {idx}: {k_name}"
            debug_origin = inst.get("debugOrigin", "")
            if debug_origin:
                node_name += f" [{debug_origin}]"

            out_buf = inst.get("outBuffer", {})
            mem_idx = out_buf.get("memSpaceIdx", 1)
            mem_type = str(out_buf.get("memSpaceType", "CPP"))
            engine_key = f"Engine(idx={mem_idx}, engine_type={mem_type})"

            children = inst.get("children", [])
            children_finish = max(
                [eclass_finish_time.get(c, 0.0) for c in children], default=0.0
            )
            engine_free = engine_finish_time.get(engine_key, 0.0)

            start_time = round(max(children_finish, engine_free), 4)
            end_time = round(start_time + cost, 4)

            eclass_finish_time[eclass_id] = end_time
            engine_finish_time[engine_key] = end_time

            node_obj = {
                "name": node_name,
                "op": op_clean,
                "start": start_time,
                "birth": start_time,
                "death": end_time,
                "cost": cost,
                "duration": cost,
                "engine": engine_key,
                "size": out_buf.get("size", 0),
                "mem_space": {"idx": mem_idx, "handle_type": mem_type},
                "children": [str(c) for c in children],
            }
            ordered_nodes.append(node_obj)

            buf_id = out_buf.get("id", idx)
            buf_key = (buf_id, mem_idx)

            alloc_buf = {
                "idx": buf_id,
                "node_name": node_name,
                "op": op_clean,
                "start": start_time,
                "end": end_time,
                "offset": out_buf.get("offset", -1),
                "size": out_buf.get("size", 0),
                "mem_space_idx": mem_idx,
                "mem_space_handle": mem_type,
            }
            allocated_map[buf_key] = alloc_buf

            unalloc_buf = dict(alloc_buf)
            unalloc_buf["offset"] = -1
            buffers_map[buf_key] = unalloc_buf

            # Extend lifetimes of input buffers consumed by this instruction
            for in_buf in inst.get("inBuffers", []):
                in_buf_id = in_buf.get("id")
                in_mem_idx = in_buf.get("memSpaceIdx", 1)
                in_key = (in_buf_id, in_mem_idx)
                if in_key in allocated_map:
                    allocated_map[in_key]["end"] = max(
                        allocated_map[in_key]["end"], end_time
                    )
                    buffers_map[in_key]["end"] = max(
                        buffers_map[in_key]["end"], end_time
                    )

        all_orders.append(
            {
                "ordered": ordered_nodes,
                "buffers": list(buffers_map.values()),
                "allocated": list(allocated_map.values()),
            }
        )

    return all_orders

@app.route("/")
def index():
    prototype_names = list(graphs.keys())
    compiled_map = discover_compiled_graphs()
    all_names = prototype_names + list(compiled_map.keys())
    return render_template("index.html", graph_names=all_names)


@app.route("/api/graph/<path:name>")
def get_graph_data(name):
    # 1. Prototype Synthetic Graphs
    if name in graphs:
        graph = graphs[name]
        all_orders = []
        mem_cap = {1: 1024, 2: 1024}

        for ordered in iter_dispatch_orders(graph):
            buffers, node_to_buffer = bufferize(ordered)

            buffer_to_nodes: dict[int, list[str]] = defaultdict(list)
            for node_name, buf_idx in node_to_buffer.items():
                buffer_to_nodes[buf_idx].append(node_name)

            fresh_buffers = [
                Buffer(b.idx, b.mem_space, b.size, b.start, b.end) for b in buffers
            ]

            buf_by_mem_idx = defaultdict(list)
            for buf in fresh_buffers:
                buf.idx = len(buf_by_mem_idx[buf.mem_space.idx])
                buf_by_mem_idx[buf.mem_space.idx].append(buf)

            allocated_buffers = []
            for mem_idx, bufs in buf_by_mem_idx.items():
                cap = mem_cap.get(mem_idx, None)
                allocated_for_space = malloc(cap, bufs, [])
                allocated_buffers.extend(allocated_for_space)

            all_orders.append(
                {
                    "ordered": [
                        node_adapter.dump_json(node).decode() for node in ordered
                    ],
                    "buffers": [
                        buffer_adapter.dump_json(buf).decode() for buf in buffers
                    ],
                    "allocated": [
                        buffer_adapter.dump_json(buf).decode()
                        for buf in allocated_buffers
                    ],
                }
            )

        return jsonify({"graph_name": name, "orders": all_orders})

    # 2. Compiled Graph Files
    compiled_map = discover_compiled_graphs()
    if name in compiled_map:
        filepath = compiled_map[name]
        orders = load_compiled_graph_orders(filepath)
        if not orders:
            return jsonify({"error": "Failed to parse compiled graph"}), 500
        return jsonify({"graph_name": name, "orders": orders})

    return "Not found", 404


if __name__ == "__main__":
    app.run(debug=True)
