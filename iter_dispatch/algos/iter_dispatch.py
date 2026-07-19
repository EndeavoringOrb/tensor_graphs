from enum import Enum


class Op(Enum):
    INPUT = 0
    ADD = 1
    MUL = 2
    COPYTO = 3  # copy between two mem spaces with the same handle but different idxs
    TRANSFER = (
        4  # transfer between two mem spaces with the same idx but different handles
    )
    SQRT = 5


class Handle(
    Enum
):  # for determining if a kernel can be used. the thing that distinguishes a handle is the read/write api.
    STORAGE = 0  # file descriptors
    CPP = 1  # pointers
    OPENCL = 2  # cl_mem??


class EngineType(
    Enum
):  # for determining if a kernel can be used. the thing that distinguishes an engine type is if the kernel can be run on that hardware.
    CPU = 0
    QUALCOMM_IGPU = 1


class Engine:  # for cost estimation
    idx: int  # global engine idx. i.e. cpu=0, gpu0=1, gpu1=2
    engine_type: EngineType

    def __init__(self, idx: int, engine_type: EngineType) -> None:
        self.idx = idx
        self.engine_type = engine_type

    def __str__(self) -> str:
        return f"Engine(idx={self.idx},engine_type={self.engine_type})"

    def __eq__(self, other):
        if isinstance(other, Engine):
            return self.idx == other.idx and self.engine_type == other.engine_type
        return NotImplemented

    def __hash__(self):
        return hash(self.idx)


class MemSpace:  # an allocated buffer of memory
    idx: int  # where is this physically. used for determining when to insert COPYTO ops. i.e. disk=0, cpu=1, gpu0=2, gpu1=3
    handle_type: Handle  # how is this read/written. used for determining when to insert TRANSFER ops.

    def __init__(self, idx: int, handle_type: Handle) -> None:
        self.idx = idx
        self.handle_type = handle_type


# system with disk, cpu, discrete gpu
cpu = Engine(0, EngineType.CPU)
gpu = Engine(1, EngineType.QUALCOMM_IGPU)
storage = MemSpace(0, Handle.STORAGE)
ram_cpu = MemSpace(1, Handle.CPP)
ram_cpu_opencl = MemSpace(1, Handle.OPENCL)
ram_gpu = MemSpace(2, Handle.OPENCL)

engine_capabilities = {
    cpu: {storage, ram_cpu, ram_cpu_opencl},
    gpu: {ram_gpu, ram_cpu_opencl},
}

op_costs = {
    Op.INPUT: 0,
    Op.ADD: 1,
    Op.MUL: 2,
    Op.COPYTO: 10,
    Op.TRANSFER: 1,
    Op.SQRT: 10,
    (Op.SQRT, EngineType.QUALCOMM_IGPU): 1,
}


def does_engine_support_mem_space(engine: Engine, mem_space: MemSpace):
    return mem_space in engine_capabilities[engine]


class Node:
    name: str
    op: Op
    children: list[str]
    mem_space: MemSpace
    engine: Engine
    size: int

    def __init__(
        self,
        name: str,
        op: Op,
        children: list[str],
        mem_space: MemSpace,
        engine: Engine,
        size: int,
    ) -> None:
        self.name = name
        self.op = op
        self.children = children
        self.mem_space = mem_space
        self.engine = engine
        self.size = size

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def __str__(self) -> str:
        return f"Node('{self.name}',{self.op})"

    def cost(self) -> int:
        return op_costs.get((self.op, self.engine.engine_type), op_costs[self.op])


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


def get_node(nodes: list[Node], name: str):
    for node in nodes:
        if name == node:
            return node
    raise ValueError()


def get_schedule(ordered: list[Node]):
    engine_finish = {}
    schedule = []

    for node in ordered:
        children_finish = 0
        for child in node.children:
            child_engine = get_node(ordered, child).engine
            if child_engine not in engine_finish:
                engine_finish[child_engine] = 0
            child_finish = engine_finish[child_engine]
            children_finish = max(children_finish, child_finish)
        node_cost = node.cost()
        node_finish = node_cost + max(
            children_finish, engine_finish.get(node.engine, 0)
        )
        engine_finish[node.engine] = node_finish
        schedule.append(
            {
                "name": node.name,
                "op": node.op.name,
                "engine": str(node.engine),
                "start": node_finish - node_cost,
                "end": node_finish,
                "duration": node_cost,
            }
        )

    return schedule


def get_count(graph: list[Node]):
    total_orders = 0
    best_cost = float("inf")
    for ordered in iter_dispatch_orders(graph):
        schedule = get_schedule(ordered)
        final_cost = schedule[-1]["end"]
        best_cost = min(best_cost, final_cost)
        total_orders += 1

    return total_orders, best_cost


def validate(graph: list[Node]):
    for node in graph:
        assert does_engine_support_mem_space(node.engine, node.mem_space)
        for child in node.children:
            child_node = get_node(graph, child)
            assert does_engine_support_mem_space(node.engine, child_node.mem_space)


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
        Node("2", Op.COPYTO, ["4"], ram_cpu, cpu, 1),
        Node("4", Op.COPYTO, ["5"], ram_cpu_opencl, gpu, 1),
        Node("5", Op.SQRT, ["6"], ram_gpu, gpu, 1),
        Node("6", Op.COPYTO, ["7"], ram_gpu, gpu, 1),
        Node("7", Op.COPYTO, ["8"], ram_cpu_opencl, cpu, 1),
        Node("8", Op.COPYTO, ["b"], ram_cpu, cpu, 1),
        Node("b", Op.INPUT, [], storage, cpu, 1),
    ],
}
if __name__ == "__main__":
    for name, graph in graphs.items():
        validate(graph)
        total_orders, best_cost = get_count(graph)
        print(f"({name}) # Orders: {total_orders}, Best Cost: {best_cost}")
