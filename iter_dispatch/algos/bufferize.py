from .iter_dispatch import get_node, storage
from .core import Buffer, Node


def overlaps(a: Buffer, b: Buffer):
    if b.start < a.start:
        a, b = b, a
    return b.start < a.end


def get_births(ordered: list[Node]):
    engine_finish = {}

    for node in ordered:
        children_finish = 0
        for child in node.children:
            child_engine = get_node(ordered, child).engine
            if child_engine not in engine_finish:
                engine_finish[child_engine] = 0
            child_finish = engine_finish[child_engine]
            children_finish = max(children_finish, child_finish)
        node.birth = max(children_finish, engine_finish.get(node.engine, 0))
        engine_finish[node.engine] = node.cost + node.birth


def get_deaths(ordered: list[Node]):
    for i in range(len(ordered)):
        node = ordered[i]
        node.death = node.birth + node.cost
        for j in range(i + 1, len(ordered)):
            other_node = ordered[j]
            if node in other_node.children:
                node.death = max(node.death, other_node.birth)


# simplest possible buffer assignment
def bufferize(ordered: list[Node]):
    get_births(ordered)
    get_deaths(ordered)
    buffers: list[Buffer] = []
    node_to_buffer: dict[str, int] = {}
    for i in range(len(ordered)):
        node = ordered[i]
        if node.mem_space == storage:
            continue  # we don't control storage. TODO: don't treat storage specially, just don't make any kernels that write to storage
        node_to_buffer[node.name] = len(buffers)
        buffers.append(
            Buffer(
                len(buffers),
                node.mem_space,
                node.size,
                node.birth,
                node.death,
            )
        )
    return buffers, node_to_buffer
