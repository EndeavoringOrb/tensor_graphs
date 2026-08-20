from collections import defaultdict

from .bufferize import Buffer, bufferize, overlaps
from .iter_dispatch import (
    graphs,
    iter_dispatch_orders,
)


def get_min_height(unallocated: list[Buffer], allocated: list[Buffer]):
    min_height = None
    for i in range(len(unallocated)):
        offset_max = 0
        for j in range(len(allocated)):
            if overlaps(unallocated[i], allocated[j]):
                offset_max = max(offset_max, allocated[j].offset + allocated[j].size)
        height = offset_max + unallocated[i].size
        min_height = min(min_height, height) if min_height is not None else height
    assert min_height is not None
    return min_height


def malloc(mem_cap: int | None, unallocated: list[Buffer], allocated: list[Buffer]):
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

        h_min = get_min_height(unallocated, allocated)
        if offset_i >= h_min:
            continue  # dominated

        if mem_cap and (offset_i + unallocated[i].size) > mem_cap:
            continue  # exceeds mem cap

        buf = unallocated[i]
        buf.offset = offset_i
        res = malloc(
            mem_cap,
            [unallocated[j] for j in range(len(unallocated)) if j != i],
            allocated + [buf],
        )
        if len(res) != 0:
            return res
    return []


if __name__ == "__main__":
    # 1=cpu, 2=gpu, storage no limit because we don't write to storage
    mem_cap = {
        1: 1024,
        2: 1024,
    }

    for name, graph in graphs.items():
        for ordered in iter_dispatch_orders(graph):
            buffers, node_to_buffer = bufferize(ordered)
            buf_by_mem_idx = defaultdict(list)
            for buf in buffers:
                buf.idx = len(buf_by_mem_idx[buf.mem_space.idx])
                buf_by_mem_idx[buf.mem_space.idx].append(buf)
            for mem_idx, bufs in buf_by_mem_idx.items():
                allocated = malloc(mem_cap.get(mem_idx, None), bufs, [])
                print(f"({name}) ({mem_idx}) allocated: {allocated}")
