from .iter_dispatch import Node, MemSpace, storage


class Buffer:
    idx: int
    mem_space: MemSpace
    size: int
    start: int
    end: int
    offset: int = -1

    def __init__(
        self, idx: int, mem_space: MemSpace, size: int, start: int, end: int
    ) -> None:
        self.idx = idx
        self.mem_space = mem_space
        self.size = size
        self.start = start
        self.end = end


def overlaps(a: Buffer, b: Buffer):
    if b.start < a.start:
        a, b = b, a
    return b.start < a.end


# simplest possible buffer assignment
def bufferize(ordered: list[Node], schedule: list[dict]):
    buffers: list[Buffer] = []
    node_to_buffer: dict[str, int] = {}
    for i in range(len(ordered)):
        node = ordered[i]
        if node.mem_space == storage:
            continue  # we don't control storage
        node_schedule = schedule[i]
        node_to_buffer[node.name] = len(buffers)
        buffers.append(
            Buffer(
                len(buffers),
                node.mem_space,
                node.size,
                node_schedule["start"],
                node_schedule["end"],
            )
        )
    return buffers, node_to_buffer
