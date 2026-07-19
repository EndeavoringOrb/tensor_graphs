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


def iter_dispatch_orders(nodes: list[Node]):  # TODO: apply minimalloc to this too?
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
        Node("0", Op.ADD, ["1", "2"], ram_cpu, cpu, 1),
        Node("1", Op.COPYTO, ["a"], ram_cpu, cpu, 1),
        Node("2", Op.COPYTO, ["b"], ram_cpu, cpu, 1),
        Node("a", Op.INPUT, [], storage, cpu, 1),
        Node("b", Op.INPUT, [], storage, cpu, 1),
    ],
    "cpu,gpu (a^2 + b^2)": [
        Node("0", Op.ADD, ["1", "2"], ram_cpu, cpu, 1),
        Node("1", Op.SQRT, ["3"], ram_cpu, cpu, 1),
        Node("3", Op.COPYTO, ["a"], ram_cpu, cpu, 1),
        Node("a", Op.INPUT, [], storage, cpu, 1),
        Node("2", Op.TRANSFER, ["4"], ram_cpu, cpu, 1),
        Node("4", Op.COPYTO, ["5"], ram_cpu_opencl, gpu, 1),
        Node("5", Op.SQRT, ["6"], ram_gpu, gpu, 1),
        Node("6", Op.COPYTO, ["7"], ram_gpu, gpu, 1),
        Node("7", Op.TRANSFER, ["8"], ram_cpu_opencl, cpu, 1),
        Node("8", Op.COPYTO, ["b"], ram_cpu, cpu, 1),
        Node("b", Op.INPUT, [], storage, cpu, 1),
    ],
}

if __name__ == "__main__":
    for name, graph in graphs.items():
        print(f"({name}) # Orders: {sum(1 for _ in iter_dispatch_orders(graph))}")
