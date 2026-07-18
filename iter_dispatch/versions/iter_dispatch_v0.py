from dataclasses import dataclass, field


@dataclass
class Node:
    name: str = ""
    children: list[str] = field(default_factory=list)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


def get_ready(remaining: set[Node]):
    ready: list[Node] = []
    for node in remaining:
        node_ready = True
        for child in node.children:
            if child in remaining:
                node_ready = False
                break
        if node_ready:
            ready.append(node)
    if len(ready) == 0:
        raise ValueError()
    return ready


def ascend(remaining: set[Node], ordered: list[Node], selection_map: dict[int, int]):
    selection_map.pop(len(ordered), None)
    remaining.add(ordered.pop())


node_b = Node("b", [])
node_c = Node("c", [])
node_a = Node("a", ["b", "c"])


def iter_dispatch_orders(nodes: list[Node]):
    remaining = set(nodes)
    ordered = []
    selection_map = {}

    while True:
        while True:
            ready = get_ready(remaining)

            # first iter ready = [b, c]
            choice = (
                selection_map[len(ordered)] + 1 if len(ordered) in selection_map else 0
            )
            if choice < len(ready):  # descend
                selection_map[len(ordered)] = choice
                ordered.append(ready[choice])
                remaining.remove(ready[choice])
            else:  # ascend
                if len(ordered) == 0:
                    return
                ascend(remaining, ordered, selection_map)
            if not remaining:
                break

        yield ordered
        ascend(remaining, ordered, selection_map)


def get_count(graph: list[Node]):
    total_orders = 0
    for ordered in iter_dispatch_orders(graph):
        print(ordered)
        total_orders += 1

    return total_orders


graphs = {
    "b+c": [Node("b", []), Node("c", []), Node("0", ["b", "c"])],
    "a*(b+c)": [
        Node("b", []),
        Node("c", []),
        Node("a", []),
        Node("0", ["a", "1"]),
        Node("1", ["b", "c"]),
    ],
    "(a*b)+(b*c)": [
        Node("b", []),
        Node("c", []),
        Node("a", []),
        Node("0", ["1", "2"]),
        Node("1", ["a", "b"]),
        Node("2", ["b", "c"]),
    ],
}

for name, graph in graphs.items():
    print(f"# Orders ({name}): {get_count(graph)}")
