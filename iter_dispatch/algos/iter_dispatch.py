from .core import (
    Node,
    does_engine_support_mem_space,
    Op,
    cpu,
    gpu,
    storage,
    ram_cpu,
    ram_cpu_opencl,
    ram_gpu,
)


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


def validate(nodes: list[Node]):
    # validate that inputs can be accessed from the node's engine
    for node in nodes:
        assert does_engine_support_mem_space(node.engine, node.mem_space)
        for child in node.children:
            child_node = get_node(nodes, child)
            assert does_engine_support_mem_space(node.engine, child_node.mem_space)


def ascend(remaining: set[Node], ordered: list[Node], selection_map: dict[int, int]):
    selection_map.pop(len(ordered), None)
    remaining.add(ordered.pop())


def iter_dispatch_orders(nodes: list[Node]):
    validate(nodes)
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


def get_node(nodes: list[Node], name: str):
    for node in nodes:
        if name == node:
            return node
    raise ValueError()


graphs = {
    "cpu a+b": [
        Node(
            name="0",
            op=Op.ADD,
            children=["1", "2"],
            mem_space=ram_cpu,
            engine=cpu,
            size=1,
        ),
        Node(
            name="1",
            op=Op.COPYTO,
            children=["a"],
            mem_space=ram_cpu,
            engine=cpu,
            size=1,
        ),
        Node(
            name="2",
            op=Op.COPYTO,
            children=["b"],
            mem_space=ram_cpu,
            engine=cpu,
            size=1,
        ),
        Node(name="a", op=Op.INPUT, children=[], mem_space=storage, engine=cpu, size=1),
        Node(name="b", op=Op.INPUT, children=[], mem_space=storage, engine=cpu, size=1),
    ],
    "cpu,gpu (a^2 + b^2)": [
        Node(
            name="0",
            op=Op.ADD,
            children=["1", "2"],
            mem_space=ram_cpu,
            engine=cpu,
            size=1,
        ),
        Node(
            name="1", op=Op.SQRT, children=["3"], mem_space=ram_cpu, engine=cpu, size=1
        ),
        Node(
            name="3",
            op=Op.COPYTO,
            children=["a"],
            mem_space=ram_cpu,
            engine=cpu,
            size=1,
        ),
        Node(name="a", op=Op.INPUT, children=[], mem_space=storage, engine=cpu, size=1),
        Node(
            name="2",
            op=Op.COPYTO,
            children=["4"],
            mem_space=ram_cpu,
            engine=cpu,
            size=1,
        ),
        Node(
            name="4",
            op=Op.COPYTO,
            children=["5"],
            mem_space=ram_cpu_opencl,
            engine=gpu,
            size=1,
        ),
        Node(
            name="5", op=Op.SQRT, children=["6"], mem_space=ram_gpu, engine=gpu, size=1
        ),
        Node(
            name="6",
            op=Op.COPYTO,
            children=["7"],
            mem_space=ram_gpu,
            engine=gpu,
            size=1,
        ),
        Node(
            name="7",
            op=Op.COPYTO,
            children=["8"],
            mem_space=ram_cpu_opencl,
            engine=cpu,
            size=1,
        ),
        Node(
            name="8",
            op=Op.COPYTO,
            children=["b"],
            mem_space=ram_cpu,
            engine=cpu,
            size=1,
        ),
        Node(name="b", op=Op.INPUT, children=[], mem_space=storage, engine=cpu, size=1),
    ],
}

if __name__ == "__main__":
    for name, graph in graphs.items():
        print(f"({name}) # Orders: {sum(1 for _ in iter_dispatch_orders(graph))}")
