from collections import defaultdict

from algos.bufferize import bufferize
from algos.core import Buffer, Node
from algos.iter_dispatch import graphs, iter_dispatch_orders
from algos.malloc import malloc
from flask import Flask, jsonify, render_template
from pydantic import TypeAdapter

app = Flask(__name__)

buffer_adapter = TypeAdapter(Buffer)
node_adapter = TypeAdapter(Node)


@app.route("/")
def index():
    return render_template("index.html", graph_names=list(graphs.keys()))


@app.route("/api/graph/<name>")
def get_graph_data(name):
    graph = graphs.get(name)
    if not graph:
        return "Not found", 404

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

        # Group buffers by mem_space.idx before calling malloc
        buf_by_mem_idx = defaultdict(list)
        for buf in fresh_buffers:
            buf.idx = len(buf_by_mem_idx[buf.mem_space.idx])
            buf_by_mem_idx[buf.mem_space.idx].append(buf)

        # Allocate memory separately for each memory space index
        allocated_buffers = []
        for mem_idx, bufs in buf_by_mem_idx.items():
            cap = mem_cap.get(mem_idx, None)
            allocated_for_space = malloc(cap, bufs, [])
            allocated_buffers.extend(allocated_for_space)

        all_orders.append(
            {
                "ordered": [node_adapter.dump_json(node).decode() for node in ordered],
                "buffers": [buffer_adapter.dump_json(buf).decode() for buf in buffers],
                "allocated": [
                    buffer_adapter.dump_json(buf).decode() for buf in allocated_buffers
                ],
            }
        )

    return jsonify({"graph_name": name, "orders": all_orders})


if __name__ == "__main__":
    app.run(debug=True)
