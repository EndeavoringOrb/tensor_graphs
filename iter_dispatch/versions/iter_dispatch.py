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
):  # for determining if kernel can be used. the thing that distinguishes a handle is the read/write api.
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


op_costs = {
    Op.INPUT: 0,
    Op.ADD: 1,
    Op.MUL: 2,
    Op.COPYTO: 10,
    Op.TRANSFER: 1,
    (Op.SQRT, EngineType.CPU): 10,
    (Op.SQRT, EngineType.QUALCOMM_IGPU): 1,
}


class Node:
    name: str
    op: Op
    children: list[str]
    mem_space: MemSpace
    engine: Engine

    def __init__(
        self,
        name: str,
        op: Op,
        children: list[str],
        mem_space: MemSpace,
        engine: Engine,
    ) -> None:
        self.name = name
        self.op = op
        self.children = children
        self.mem_space = mem_space
        self.engine = engine

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

    def cost(self) -> int | None:
        return op_costs.get((self.op, self.engine.engine_type), op_costs.get(self.op))


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
        assert node_cost is not None
        node_finish = node_cost + max(
            children_finish, engine_finish.get(node.engine, 0)
        )
        engine_finish[node.engine] = node_finish
        # print(f"cost after {node}: {', '.join(f"{k}: {v}" for k, v in engine_finish.items())}")
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
    best_cost = None
    for i, ordered in enumerate(iter_dispatch_orders(graph)):
        schedule = get_schedule(ordered)
        final_cost = schedule[-1]["end"]
        best_cost = final_cost if (best_cost is None or final_cost < best_cost) else best_cost
        total_orders += 1

    return total_orders, best_cost

if __name__ == "__main__":
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
        "cpu,gpu (a^2 + b^2)": [  # for this example, assume no shared mem
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

    for name, graph in graphs.items():
        total_orders, best_cost = get_count(graph)
        print(f"({name}) # Orders: {total_orders}, Best Cost: {best_cost}")
