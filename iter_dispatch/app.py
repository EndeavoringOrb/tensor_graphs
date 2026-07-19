from flask import Flask, render_template, jsonify
from algos.iter_dispatch import (
    Engine,
    EngineType,
    MemSpace,
    Handle,
    Node,
    Op,
    iter_dispatch_orders,
    get_schedule,
)

# --- DATASET ---
cpu = Engine(0, EngineType.CPU)
gpu = Engine(1, EngineType.QUALCOMM_IGPU)
storage = MemSpace(0, Handle.STORAGE)
ram_cpu = MemSpace(1, Handle.CPP)
ram_gpu = MemSpace(2, Handle.OPENCL)

graphs = {
    "cpu a+b": [
        Node("0", Op.ADD, ["1", "2"], ram_cpu, cpu),
        Node("1", Op.COPYTO, ["a"], ram_cpu, cpu),
        Node("2", Op.COPYTO, ["b"], ram_cpu, cpu),
        Node("a", Op.INPUT, [], storage, cpu),
        Node("b", Op.INPUT, [], storage, cpu),
    ],
    "cpu,gpu (a^2 + b^2)": [
        Node("0", Op.ADD, ["1", "2"], ram_cpu, cpu),
        Node("1", Op.SQRT, ["3"], ram_cpu, cpu),
        Node("3", Op.COPYTO, ["a"], ram_cpu, cpu),
        Node("a", Op.INPUT, [], storage, cpu),
        Node("2", Op.COPYTO, ["4"], ram_cpu, cpu),
        Node("4", Op.SQRT, ["5"], ram_gpu, gpu),
        Node("5", Op.COPYTO, ["b"], ram_gpu, cpu),
        Node("b", Op.INPUT, [], storage, cpu),
    ],
}

# --- FLASK ROUTES ---
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", graph_names=list(graphs.keys()))


@app.route("/api/graph/<name>")
def get_graph_data(name):
    graph_nodes = graphs.get(name)
    if not graph_nodes:
        return "Not found", 404

    all_orders = []
    for order in iter_dispatch_orders(graph_nodes):
        schedule = get_schedule(order)
        all_orders.append(schedule)

    return jsonify({"graph_name": name, "orders": all_orders})


if __name__ == "__main__":
    app.run(debug=True)
