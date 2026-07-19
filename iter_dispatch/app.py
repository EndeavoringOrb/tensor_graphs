from flask import Flask, render_template, jsonify
from algos.iter_dispatch import iter_dispatch_orders, get_schedule, graphs

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
