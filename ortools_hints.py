"""Translate a native compiled-plan witness into CP-SAT starting hints.

The native planner is deliberately the only component that searches for a
feasible extraction, dispatch order, and allocation.  This module only maps
that compiled-plan representation onto the variables created by
``ortools_full.OrtoolsSolver``.
"""

import math
from collections.abc import Iterable
from typing import Any


def durationForHint(enode: dict[str, Any]) -> int:
    cost = float(enode.get("cost", 0.0))
    if not math.isfinite(cost) or cost < 0.0:
        return 0
    if enode.get("is_input") or enode.get("is_cache") or enode.get("is_view"):
        return 0
    return max(1, math.ceil(cost * 1000.0))


def bufferOffsets(extraction: dict[str, Any]) -> dict[int, int]:
    return {
        int(item["id"]): max(0, int(item.get("offset", 0)))
        for item in extraction.get("buffers", [])
    }


def bucketHint(
    problem_data: dict[str, Any], bucket: dict[str, Any], extraction: dict[str, Any]
) -> dict[str, Any]:
    classes = {int(item["id"]): item for item in bucket.get("classes", [])}
    selected = {int(cid): int(eidx) for cid, eidx in extraction.get("selection_map", {}).items()}
    order = [int(cid) for cid in extraction.get("order", []) if int(cid) in selected]
    owners = {int(cid): int(buf) for cid, buf in extraction.get("eclass_to_buf", {}).items()}
    offsets = bufferOffsets(extraction)
    reserved_bases = {
        int(item["base_eclass_id"])
        for item in problem_data.get("preallocated_buffers", [])
    }

    selected_enodes = {
        cid: classes[cid]["enodes"][eidx]
        for cid, eidx in selected.items()
        if cid in classes and 0 <= eidx < len(classes[cid].get("enodes", []))
    }

    def resolveViewSource(cid: int) -> int:
        """Resolve the selected native view chain to its storage owner."""

        seen = set()
        while cid in selected_enodes and selected_enodes[cid].get("is_view"):
            if cid in seen or not selected_enodes[cid].get("children"):
                break
            seen.add(cid)
            cid = int(selected_enodes[cid]["children"][0])
        return cid

    # Native extraction buffer ids are local to the extraction.  They are not
    # the ids assigned to persistent buffers by Planner::preallocate().  Find
    # persistent owners from the selected INPUT/CACHE eclasses instead of
    # comparing unrelated numeric ids.
    native_preallocated_owners = {
        owners[cid]
        for cid, enode in selected_enodes.items()
        if cid in owners
        and (enode.get("is_input") or enode.get("is_cache"))
        and int(classes[cid]["base_eclass_id"]) in reserved_bases
    }

    def isPersistent(cid: int) -> bool:
        return (
            int(classes[cid]["base_eclass_id"]) in reserved_bases
            or owners.get(cid) in native_preallocated_owners
        )

    start: dict[int, int] = {}
    end: dict[int, int] = {}
    now = 0
    for cid in order:
        enode = classes[cid]["enodes"][selected[cid]]
        if enode.get("is_input"):
            # The CP model makes every input available at time zero, even if
            # the native topological order lists it after another input.
            start[cid] = 0
            end[cid] = 0
        else:
            start[cid] = now
            now += durationForHint(enode)
            end[cid] = now

    consumers: dict[int, list[int]] = {cid: [] for cid in order}
    inplace: dict[int, int] = {}
    views: dict[int, int] = {}
    for cid in order:
        enode = classes[cid]["enodes"][selected[cid]]
        for child in set(map(int, enode.get("children", []))):
            if child in consumers:
                consumers[child].append(cid)
        # A shared native allocation represents either a view or an in-place
        # write.  The CP model distinguishes those using the selected enode.
        if enode.get("is_view"):
            if enode.get("children"):
                child = int(enode["children"][0])
                if owners.get(child) == owners.get(cid) and child != cid:
                    views[cid] = child
        else:
            children = [int(child) for child in enode.get("children", [])]
            safe_idxs = {int(idx) for idx in enode.get("safe_inplace_idxs", [])}
            for child_idx, child in enumerate(children):
                child = int(child)
                child_source = resolveViewSource(child)
                source_enode = selected_enodes.get(child_source, {})
                if (
                    owners.get(child_source) == owners.get(cid)
                    and child != cid
                    and child_idx in safe_idxs
                    and child_source != cid
                    and owners.get(child_source) not in native_preallocated_owners
                    and not source_enode.get("is_input")
                    and not source_enode.get("is_cache")
                    and int(classes[cid]["raw_size_bytes"])
                    <= int(classes[child_source]["raw_size_bytes"])
                ):
                    inplace[cid] = child
                    break

    read_end = {cid: end[cid] for cid in order}
    for child, users in consumers.items():
        if users:
            read_end[child] = max(read_end[child], *(end[user] for user in users))

    root_id = int(bucket["root_eclass_id"])
    for cid in reversed(order):
        if cid in inplace:
            child = inplace[cid]
            # The old value must be dead before an in-place write begins.
            read_end[child] = min(read_end[child], start[cid])
        if cid in views:
            child = views[cid]
            read_end[child] = max(read_end[child], read_end[cid])
    makespan = max(end.values(), default=0)

    first_owner: dict[int, int] = {}
    for cid in order:
        first_owner.setdefault(owners.get(cid, cid), cid)

    release_horizon = {
        cid: isPersistent(cid)
        or bool(classes[cid]["enodes"][selected[cid]].get("is_input"))
        or cid == root_id
        for cid in order
    }
    releases = {cid: read_end[cid] for cid in order}
    for cid in reversed(order):
        child = inplace.get(cid, views.get(cid))
        if child is not None:
            releases[child] = max(releases[child], releases[cid])
            release_horizon[child] = release_horizon[child] or release_horizon[cid]

    # A view has the same protected status as its source.  In-place writes
    # are explicitly restricted to unprotected children by the CP model.
    protected = {
        cid: int(
            isPersistent(cid)
            or bool(classes[cid]["enodes"][selected[cid]].get("is_input"))
        )
        for cid in order
    }
    for cid in order:
        if cid in views:
            protected[cid] = protected[views[cid]]

    nodes: dict[int, dict[str, Any]] = {}
    for cid in order:
        enode = classes[cid]["enodes"][selected[cid]]
        owner = owners.get(cid, cid)
        is_view = bool(enode.get("is_view"))
        reserved = isPersistent(cid)
        fresh = (
            (
                first_owner.get(owner) == cid
                and not is_view
                and not reserved
                and cid not in inplace
            )
        )
        release = makespan if release_horizon[cid] else releases[cid]
        nodes[cid] = {
            "active": 1,
            "selection": selected[cid],
            "start": start[cid],
            "end": end[cid],
            "read_end": makespan if cid == root_id else read_end[cid],
            "release": release,
            "protected": protected[cid],
            "release_horizon": int(release_horizon[cid]),
            "fresh": 1 if fresh else 0,
            "page_offset": (
                offsets.get(owner, 0) // 4096 if fresh else 0
            ),
            "inplace": inplace.get(cid),
        }

    # The CP model contains every reachable class, including alternatives that
    # native extraction did not select.  Keep those classes in the hint so the
    # hint is complete rather than merely specifying the selected subgraph.
    for cid, cls in classes.items():
        if cid in nodes:
            continue
        nodes[cid] = {
            "active": 0,
            "selection": None,
            "start": 0,
            "end": 0,
            "read_end": 0,
            "release": 0,
            "protected": 0,
            "release_horizon": 0,
            "fresh": 0,
            "page_offset": 0,
            "inplace": None,
        }

    return {"root_id": root_id, "makespan": makespan, "nodes": nodes}


def compiledGraphsToHints(
    problem_data: dict[str, Any], compiled_graphs: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Return variable-independent hint assignments for the full CP model."""

    compiled_graphs = list(compiled_graphs)
    buckets = problem_data.get("buckets", [])
    if len(compiled_graphs) != len(buckets):
        raise ValueError(
            f"Expected one native CPU hint per bucket, got {len(compiled_graphs)}"
        )

    return {
        "cached": {
            int(item["base_eclass_id"]): 0
            for item in problem_data.get("candidates", [])
        },
        "buckets": {
            int(bucket["bucket_idx"]): bucketHint(problem_data, bucket, extraction)
            for bucket, extraction in zip(buckets, compiled_graphs)
        },
    }
