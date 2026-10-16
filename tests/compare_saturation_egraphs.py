"""Compare full/seeded saturation dumps, allowing class and buffer IDs to change.

Usage: .venv/bin/python tests/compare_saturation_egraphs.py REFERENCE_PREFIX CANDIDATE_PREFIX
The matching C++ dump fixture is tensor_graphs_cpp/tests/saturation_equivalence.cpp.
Color refinement only prunes the search; exact graph isomorphism makes the decision.
"""

import re
import shlex
import sys
from collections import Counter
from pathlib import Path

import networkx as nx


def readGraph(path, bucket, graph):
    node_idx = 0
    for line in Path(path).read_text().splitlines():
        parts = shlex.split(line)
        if parts[0] == "B":
            base = ("b", int(parts[2]))
            graph.add_node(base, label=("base",))
            graph.add_edge((bucket, "c", int(parts[1])), base, slots="base")
            continue
        if parts[0] == "C":
            graph.add_node(
                (bucket, "c", int(parts[1])), label=("class", bucket, parts[2])
            )
        else:
            owner = int(parts[1])
            node = (bucket, "n", node_idx)
            node_idx += 1
            content = parts[2]
            scatter = re.search(r"scatter-into:b([0-9]+)", content)
            if scatter:
                content = (
                    content[: scatter.start()]
                    + "scatter-into:base"
                    + content[scatter.end() :]
                )
                base = ("b", int(scatter.group(1)))
                graph.add_node(base, label=("base",))
                graph.add_edge(node, base, slots="scatter-base")
            graph.add_node(node, label=("expression", bucket, content))
            graph.add_edge((bucket, "c", owner), node, slots="member")
            children = {}
            for index, child in enumerate(parts[3:]):
                children.setdefault(int(child), []).append(index)
            for child, indices in children.items():
                graph.add_edge(node, (bucket, "c", child), slots=str(indices))
    return graph


def orderedCertificate(graph):
    """Attempt an exact mapping using class creation order, before VF2 search.

    This is only a fast path: unequal certificates still use general isomorphism.
    Equal certificates give an explicit bijection and compare every labeled edge.
    """
    identities = {}
    counts = Counter()
    for node, data in graph.nodes(data=True):
        if data["label"][0] == "class":
            bucket = data["label"][1]
            identities[node] = ("class", bucket, counts[bucket])
            counts[bucket] += 1
    for node, data in graph.nodes(data=True):
        if data["label"][0] != "base":
            continue
        anchors = tuple(
            sorted(
                identities[source]
                for source, _, edge in graph.in_edges(node, data=True)
                if edge["slots"] == "base"
                and identities.get(source, (None, None))[1] == "full"
            )
        )
        if not anchors:
            return None
        identities[node] = ("base", anchors)
    rows = Counter()
    for node, data in graph.nodes(data=True):
        if data["label"][0] == "expression":
            owners = tuple(
                sorted(
                    identities[source]
                    for source, _, edge in graph.in_edges(node, data=True)
                    if edge["slots"] == "member"
                )
            )
            edges = tuple(
                sorted(
                    (edge["slots"], identities[target])
                    for _, target, edge in graph.out_edges(node, data=True)
                )
            )
            rows[(data["label"], owners, edges)] += 1
        else:
            edges = tuple(
                sorted(
                    (edge["slots"], identities[target])
                    for _, target, edge in graph.out_edges(node, data=True)
                    if edge["slots"] != "member"
                )
            )
            rows[(identities[node], data["label"], edges)] += 1
    return rows


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: compare_saturation_egraphs.py REFERENCE_PREFIX CANDIDATE_PREFIX"
        )
    reference_prefix, candidate_prefix = sys.argv[1:]
    before, after = nx.DiGraph(), nx.DiGraph()
    for suffix in ["full", "partial0", "partial2"]:
        readGraph(f"{reference_prefix}-{suffix}.txt", suffix, before)
        readGraph(f"{candidate_prefix}-{suffix}.txt", suffix, after)
    print("family vertices:", len(before), len(after), flush=True)
    reference_certificate = orderedCertificate(before)
    if (
        reference_certificate is not None
        and reference_certificate == orderedCertificate(after)
    ):
        print(
            "family exact isomorphism: True (explicit class-order bijection)",
            flush=True,
        )
        return
    # Joint color refinement gives VF2 stronger, isomorphism-invariant labels.
    graphs = [before, after]
    palette = {}
    for graph in graphs:
        for node, data in graph.nodes(data=True):
            data["color"] = palette.setdefault(data["label"], len(palette))
    for _ in range(len(before) + len(after)):
        next_palette = {}
        updates = []
        for graph in graphs:
            colors = {}
            for node in graph:
                signature = (
                    graph.nodes[node]["color"],
                    tuple(
                        sorted(
                            (edge["slots"], graph.nodes[other]["color"])
                            for _, other, edge in graph.out_edges(node, data=True)
                        )
                    ),
                    tuple(
                        sorted(
                            (edge["slots"], graph.nodes[other]["color"])
                            for other, _, edge in graph.in_edges(node, data=True)
                        )
                    ),
                )
                colors[node] = next_palette.setdefault(signature, len(next_palette))
            updates.append(colors)
        for graph, colors in zip(graphs, updates):
            nx.set_node_attributes(graph, colors, "color")
        if len(next_palette) == len(palette):
            break
        palette = next_palette
    colors = [
        Counter(data["color"] for _, data in graph.nodes(data=True)) for graph in graphs
    ]
    if colors[0] != colors[1]:
        print(
            "family exact isomorphism: False (different refined partitions)", flush=True
        )
        raise SystemExit(1)
    # Encode edge labels as intermediate vertices for VF2++, which uses a
    # substantially better search order on these large directed graphs.
    expanded = []
    edge_palette = {}
    for graph in graphs:
        colored = nx.DiGraph()
        for node, data in graph.nodes(data=True):
            colored.add_node(("vertex", node), match=("vertex", data["color"]))
        for index, (source, destination, data) in enumerate(graph.edges(data=True)):
            edge = ("edge", index)
            color = edge_palette.setdefault(data["slots"], len(edge_palette))
            colored.add_node(edge, match=("edge", color))
            colored.add_edge(("vertex", source), edge)
            colored.add_edge(edge, ("vertex", destination))
        expanded.append(colored)
    match = nx.vf2pp_is_isomorphic(*expanded, node_label="match")
    print("family exact isomorphism:", match, flush=True)
    if not match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
