from .iter_dispatch import (
    iter_dispatch_orders,
    get_schedule,
    graphs,
)
from .bufferize import Buffer, bufferize, overlaps


def malloc(mem_cap, unallocated: list[Buffer], allocated: list[Buffer]):
    if len(unallocated) == 0:
        return allocated
    for i in range(len(unallocated)):
        offset_i = 0
        offset_max = 0
        for j in range(len(allocated)):
            if overlaps(unallocated[i], allocated[j]):
                offset_i = max(offset_i, allocated[j].offset + allocated[j].size)
            offset_max = max(offset_max, allocated[j].offset)
        if offset_i < offset_max:
            continue  # offset non-monotonic
        idx_max = 0
        for j in range(len(allocated)):
            if allocated[j].offset == offset_i:
                idx_max = max(idx_max, allocated[j].idx)
        if unallocated[i].idx < idx_max:
            continue  # index non-monotonic
        cap = mem_cap.get(unallocated[i].mem_space.idx, None)
        if (
            cap and (offset_i + unallocated[i].size) > cap
        ):  # TODO: figure out mem idx vs handle capping
            continue  # exceeds mem cap
        buf = unallocated[i]
        buf.offset = offset_i
        allocated = malloc(
            mem_cap,
            [unallocated[j] for j in range(len(unallocated)) if j != i],
            allocated + [buf],
        )
        if len(allocated) != 0:
            return allocated
    return []


if __name__ == "__main__":
    mem_cap = {
        1: 1024,
        2: 1024,
    }  # 1=cpu, 2=gpu, storage no limit because we don't write to storage
    for name, graph in graphs.items():
        for ordered in iter_dispatch_orders(graph):
            schedule = get_schedule(ordered)
            buffers, node_to_buffer = bufferize(ordered, schedule)
            allocated = malloc(mem_cap, buffers, [])
            print(f"({name}) allocated: {allocated}")
