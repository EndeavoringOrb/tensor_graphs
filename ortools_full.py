"""Joint cache, extraction, dispatch, bufferization and allocation with CP-SAT.

All decisions belong to one model and one Solve call. Optional rectangles pack
buffer lifetimes against 4KB page offsets; optional kernel intervals reserve engines.
Views and in-place choices propagate allocation lifetimes back to their owners.
Only decoding (including sorting the solved schedule) happens after Solve.
"""

import math
from collections import defaultdict

from ortools.sat.python import cp_model


def memSpaceKey(mem_space):
    return f"{mem_space['type']}:{mem_space['idx']}"


class OrtoolsSolver:
    """Choose one extraction per bucket, sharing persistent cache allocations.

    An active node contributes one selected enode to the plan. A fresh node owns
    a bucket-local allocation; other active nodes alias a child or a global buffer.
    Protected buffers hold inputs or caches and cannot be overwritten in place.
    Each cache candidate has one fixed memory space. Buckets execute separately;
    the full bucket must run before any bucket reads or partially updates a cache.

    ``read_end`` bounds the last read of a tensor's current value, including views.
    ``release`` bounds the lifetime of its allocation, including later values
    written into it by in-place operations. Only the former permits overwriting.
    """

    alignment = 4096
    time_scale = 1000  # Native costs are milliseconds; schedule in microseconds.
    integer_limit = 2**60
    persistent_end = 2**32 - 1

    def __init__(self, problem_data):
        self.problem_data = problem_data
        self.model = cp_model.CpModel()
        self.buckets = problem_data["buckets"]
        self.caching_enabled = not problem_data["disable_caching"]
        # In cache-disabled mode, omit the candidate data entirely. This keeps
        # cache placement and cache-choice expressions out of the CP-SAT model,
        # rather than merely forcing them to zero after creating them.
        self.candidates = problem_data["candidates"] if self.caching_enabled else []
        self.preallocated = problem_data["preallocated_buffers"]
        self.mem_caps = problem_data["mem_caps"]
        self.next_buffer_id = (
            max((int(item["buffer_id"]) for item in self.preallocated), default=-1) + 1
        )
        self.global_buffers = []
        self.cache_choices = {}
        self.bucket_models = []
        self.objective_terms = []
        self.caps = {}
        self.preallocated_by_base_id = {}
        self.candidates_by_base_id = {}
        self.model_built = False
        self.full_bucket_idx = problem_data["full_bucket_idx"]

        # Reserve enough pages to place every allocation without reuse. Summing
        # unrounded sizes underestimates this bound for small tensors.
        preallocated_extent = 0
        for item in self.preallocated:
            size = self.alignedSize(item["size"])
            offset = int(item["offset"])
            if offset < 0:
                raise ValueError("Preallocated buffer offsets must be nonnegative")
            if item["mem_space"]["type"] != 0:
                if offset % self.alignment:
                    raise ValueError("Preallocated arena offsets must be 4096-byte aligned")
                preallocated_extent = max(preallocated_extent, offset + size)

        # A finite arena bound avoids overflowing CP-SAT when native caps are UINT64_MAX.
        self.arena_bound = (
            self.alignment
            + preallocated_extent
            + sum(self.alignedSize(item["size_bytes"]) for item in self.candidates)
            + sum(
                self.alignedSize(cls["size_bytes"])
                for bucket in self.buckets
                for cls in bucket["classes"]
            )
        )
        if self.arena_bound >= self.integer_limit:
            raise ValueError("OR-Tools full problem exceeds the integer memory range")

    def alignedSize(self, size):
        size = int(size)
        if size < 0:
            raise ValueError("Buffer sizes must be nonnegative")
        return ((size + self.alignment - 1) // self.alignment) * self.alignment

    def allocationSize(self, item):
        size = int(item["size_bytes"])
        raw_size = int(item["raw_size_bytes"])
        if raw_size < 0 or raw_size > size:
            raise ValueError("Raw tensor size must fit its allocation")
        return size if item["mem_space"]["type"] == 0 else self.alignedSize(size)

    def newBufferId(self):
        buffer_id = self.next_buffer_id
        if not 0 <= buffer_id < self.persistent_end:
            raise ValueError("Buffer id exceeds the native uint32 range")
        self.next_buffer_id += 1
        return buffer_id

    def memoryCap(self, mem_space):
        key = memSpaceKey(mem_space)
        if key not in self.caps:
            cap = self.mem_caps.get(key, self.arena_bound)
            cap = int(cap)
            if cap < 0:
                raise ValueError(f"Negative memory cap for {key}")
            self.caps[key] = min(cap, self.arena_bound)
        return self.caps[key]

    def memoryCapPages(self, mem_space):
        return self.memoryCap(mem_space) // self.alignment

    def newPageOffset(self, mem_space, name):
        if mem_space["type"] == 0:
            return None
        cap_pages = self.memoryCapPages(mem_space)
        return self.model.NewIntVar(0, cap_pages, f"{name}_page_offset")

    def duration(self, enode):
        try:
            cost = float(enode["cost"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(cost) or cost < 0:
            return None
        # buildCompiledGraph omits these metadata operations from execution.
        if any(enode.get(flag) for flag in ("is_input", "is_cache", "is_view")):
            return 0
        scaled_cost = cost * self.time_scale
        if not math.isfinite(scaled_cost) or scaled_cost >= self.integer_limit:
            raise ValueError("Kernel cost exceeds the integer time range")
        # Executable kernels need a positive interval even if their estimate is 0.
        return max(1, math.ceil(scaled_cost))

    def engineKeys(self, enode, mem_space):
        # TODO: Verify exported engines against native kernel matching.
        engines = enode.get("engines") or [
            {"type": 2, "idx": mem_space["idx"]}
            if mem_space["type"] == 3
            else {"type": 0, "idx": 0}
        ]
        return {memSpaceKey(engine) for engine in engines}

    def matchesBuffer(self, cls, buf):
        if cls["mem_space"] != buf["mem_space"]:
            return False
        buffer_size = (
            buf["size"]
            if buf["mem_space"]["type"] == 0
            else self.alignedSize(buf["size"])
        )
        if self.allocationSize(cls) != buffer_size:
            return False
        if "buffer_id" in buf:
            return True
        return int(cls["raw_size_bytes"]) == int(buf["raw_size_bytes"])

    def matchesCache(self, cls, base_eclass_id, buf):
        candidate = self.candidates_by_base_id[base_eclass_id]
        return self.matchesBuffer(cls, buf) and (
            int(cls["raw_size_bytes"]) == int(candidate["raw_size_bytes"])
        )

    def createGlobalBuffers(self):
        by_base_id = {}
        buffer_ids = set()
        for item in self.preallocated:
            base_eclass_id = item["base_eclass_id"]
            buffer_id = int(item["buffer_id"])
            if base_eclass_id in by_base_id or buffer_id in buffer_ids:
                raise ValueError("Duplicate preallocated base eclass or buffer id")
            if not 0 <= buffer_id < self.persistent_end:
                raise ValueError("Invalid preallocated buffer id")
            buffer_ids.add(buffer_id)
            raw_offset = int(item["offset"])
            raw_page_offset = raw_offset // self.alignment
            raw_size = int(item["size"])
            buf = dict(
                item,
                id=int(item["buffer_id"]),
                size=raw_size,
                raw_offset=raw_offset,
                present=self.model.NewConstant(1),
            )
            buf["page_offset"] = (
                None
                if item["mem_space"]["type"] == 0
                else self.model.NewConstant(raw_page_offset)
            )
            self.global_buffers.append(buf)
            by_base_id[item["base_eclass_id"]] = buf

        if not self.caching_enabled:
            self.preallocated_by_base_id = by_base_id
            return

        for candidate in self.candidates:
            base_eclass_id = candidate["base_eclass_id"]
            if base_eclass_id in self.candidates_by_base_id:
                raise ValueError(f"Duplicate cache candidate {base_eclass_id}")
            self.candidates_by_base_id[base_eclass_id] = candidate
            mem_space = candidate["mem_space"]

            key = memSpaceKey(mem_space)
            if mem_space["type"] == 0:
                continue
            reserved = by_base_id.get(base_eclass_id)
            if reserved is not None and not self.matchesBuffer(candidate, reserved):
                continue
            present = self.model.NewBoolVar(f"cache_{base_eclass_id}_{key}")
            if base_eclass_id in by_base_id:
                buf = by_base_id[base_eclass_id]
            else:
                page_offset = self.newPageOffset(mem_space, f"cache_{base_eclass_id}_{key}")
                buf = {
                    "id": self.newBufferId(),
                    "base_eclass_id": base_eclass_id,
                    "mem_space": mem_space,
                    "size": self.allocationSize(candidate),
                    "present": present,
                    "page_offset": page_offset,
                }
                if "raw_size_bytes" in candidate:
                    buf["raw_size_bytes"] = candidate["raw_size_bytes"]
                self.global_buffers.append(buf)
            self.cache_choices[base_eclass_id] = (present, buf)
        self.preallocated_by_base_id = by_base_id

    def addRectangle(self, rectangles, buf, start, end, horizon, name):
        if buf["mem_space"]["type"] == 0 or buf.get("page_offset") is None:
            return None
        model = self.model
        present = buf["present"]
        page_size = self.alignedSize(buf["size"]) // self.alignment
        page_offset = buf["page_offset"]

        model.Add(
            page_offset * self.alignment + buf["size"] <= self.memoryCap(buf["mem_space"])
        ).OnlyEnforceIf(present)
        # NoOverlap2D gives degenerate rectangles special semantics. Empty
        # allocations have no physical extent and should not enter the packing.
        if page_size == 0:
            return

        lifetime = model.NewIntVar(0, horizon, f"{name}_lifetime")
        time_interval = model.NewOptionalIntervalVar(
            start, lifetime, end, present, f"{name}_live"
        )
        space_interval = model.NewOptionalIntervalVar(
            page_offset, page_size, page_offset + page_size, present, f"{name}_pages"
        )
        rectangles[memSpaceKey(buf["mem_space"])].append(
            (time_interval, space_interval)
        )

    def aliasBuffer(self, node, child, present, is_view):
        model = self.model
        model.Add(node["owner"] == child["owner"]).OnlyEnforceIf(present)
        if node["page_offset"] is not None and child["page_offset"] is not None:
            model.Add(node["page_offset"] == child["page_offset"]).OnlyEnforceIf(present)
        model.Add(node["protected"] == child["protected"]).OnlyEnforceIf(present)
        model.Add(child["release"] >= node["release"]).OnlyEnforceIf(present)
        if is_view:
            model.Add(child["read_end"] >= node["read_end"]).OnlyEnforceIf(present)

    def bindGlobalBuffer(self, node, buf, present, horizon):
        model = self.model
        model.Add(node["owner"] == buf["id"]).OnlyEnforceIf(present)
        if node["page_offset"] is not None and buf["page_offset"] is not None:
            model.Add(node["page_offset"] == buf["page_offset"]).OnlyEnforceIf(present)
        model.Add(node["protected"] == 1).OnlyEnforceIf(present)
        model.Add(node["release"] == horizon).OnlyEnforceIf(present)

    def createBucket(self, bucket):
        model = self.model
        b = bucket["bucket_idx"]
        classes = {cls["id"]: cls for cls in bucket["classes"]}
        if (
            len(classes) != len(bucket["classes"])
            or bucket["root_eclass_id"] not in classes
        ):
            raise ValueError(f"Invalid eclass ids in bucket {b}")
        count = len(classes)
        durations = {
            (cid, enode["enode_idx"]): self.duration(enode)
            for cid, cls in classes.items()
            for enode in cls["enodes"]
        }
        horizon = (
            1
            + count
            + sum(
                max(
                    (
                        durations[cid, enode["enode_idx"]] or 0
                        for enode in cls["enodes"]
                    ),
                    default=0,
                )
                for cid, cls in classes.items()
            )
        )
        if horizon >= self.integer_limit:
            raise ValueError("OR-Tools full problem exceeds the integer time range")
        clean = set(bucket.get("clean_eclasses", []))
        rectangles = defaultdict(list)
        engines = defaultdict(list)
        engine_work = defaultdict(list)
        consumers = defaultdict(list)
        inplace_users = defaultdict(list)
        cache_nodes = defaultdict(list)
        global_users = defaultdict(list)
        nodes = {}
        max_owner = self.next_buffer_id + count - 1
        for cid, cls in classes.items():
            name = f"b{b}_c{cid}"
            page_offset = self.newPageOffset(cls["mem_space"], name)
            node = {
                "cls": cls,
                "id": self.newBufferId(),
                "size": self.allocationSize(cls),
                "mem_space": cls["mem_space"],
                "active": model.NewBoolVar(f"{name}_active"),
                "fresh": model.NewBoolVar(f"{name}_fresh"),
                "protected": model.NewBoolVar(f"{name}_protected"),
                "owner": model.NewIntVar(0, max_owner, f"{name}_owner"),
                "page_offset": page_offset,
                "rank": model.NewIntVar(0, count - 1, f"{name}_rank"),
                "start": model.NewIntVar(0, horizon - 1, f"{name}_start"),
                "end": model.NewIntVar(0, horizon - 1, f"{name}_end"),
                "release": model.NewIntVar(0, horizon, f"{name}_release"),
                "read_end": model.NewIntVar(0, horizon, f"{name}_read_end"),
                "selections": {},
                "inplace_choices": [],
                "cached": 0,
            }
            node["present"] = node["fresh"]
            nodes[cid] = node
            model.Add(node["fresh"] <= node["active"])
            model.Add(node["owner"] == node["id"]).OnlyEnforceIf(node["fresh"])
            model.Add(node["release"] >= node["start"] + 1).OnlyEnforceIf(
                node["active"]
            )
            model.Add(node["release"] >= node["end"])
            model.Add(node["read_end"] >= node["end"])
            model.Add(node["release"] >= node["read_end"])
            for field in ("start", "end", "rank", "read_end", "release"):
                model.Add(node[field] == 0).OnlyEnforceIf(node["active"].Not())
            self.addRectangle(
                rectangles, node, node["start"], node["release"], horizon, name
            )

        for buf in self.global_buffers:
            self.addRectangle(
                rectangles, buf, 0, horizon, horizon, f"b{b}_global{buf['id']}"
            )

        for cid, node in nodes.items():
            cls, active = node["cls"], node["active"]
            base_eclass_id = cls["base_eclass_id"]
            cache_choice = self.cache_choices.get(base_eclass_id)
            cache_matches = cache_choice is not None and self.matchesCache(
                cls, base_eclass_id, cache_choice[1]
            )
            if cache_matches:
                cached, buf = cache_choice
                node["cached"] = cached
                cache_nodes[base_eclass_id].append(node)
                model.Add(active >= cached)
                self.bindGlobalBuffer(node, buf, cached, horizon)
            cached = node["cached"]
            reserved = self.preallocated_by_base_id.get(base_eclass_id)
            if reserved is not None and not self.matchesBuffer(cls, reserved):
                reserved = None
            if reserved is not None:
                self.bindGlobalBuffer(node, reserved, active, horizon)
                global_users[reserved["id"]].append(active)
            elif cache_matches:
                global_users[cache_choice[1]["id"]].append(cached)

            choices = []
            seen_enode_indices = set()
            for enode in cls["enodes"]:
                e_idx = enode["enode_idx"]
                if e_idx in seen_enode_indices:
                    raise ValueError(f"Duplicate enode index {e_idx} in class {cid}")
                seen_enode_indices.add(e_idx)
                if not self.caching_enabled and (
                    enode.get("is_cache") or enode.get("is_scatter")
                ):
                    # Do not add cache-path variables or constraints at all.
                    # The enode remains in the input for stable diagnostics,
                    # while the model sees only executable alternatives.
                    continue
                selected = model.NewBoolVar(f"b{b}_c{cid}_e{e_idx}")
                node["selections"][e_idx] = (selected, enode)
                duration = durations[cid, e_idx]
                children = enode.get("children", [])
                if duration is None or any(child not in nodes for child in children):
                    model.Add(selected == 0)
                    continue
                model.Add(node["end"] == node["start"] + duration).OnlyEnforceIf(
                    selected
                )
                if enode.get("mem_space", cls["mem_space"]) != cls["mem_space"]:
                    model.Add(selected == 0)
                if enode.get("is_cache"):
                    candidate = self.candidates_by_base_id.get(base_eclass_id, {})
                    # TODO: Export cache versions for prior-state reads in dirty paths.
                    can_read = (
                        cid in clean
                        and b != self.full_bucket_idx
                        and (
                            "clean_buckets" not in candidate
                            or b in candidate["clean_buckets"]
                        )
                        and enode.get("base_eclass_id", base_eclass_id) == base_eclass_id
                    )
                    model.Add(selected <= cached if can_read else selected == 0)
                    if children:
                        model.Add(selected == 0)
                if enode.get("is_scatter"):
                    # A partial update needs initialized contents in untouched
                    # regions, so it cannot initialize a full-bucket cache.
                    model.Add(
                        selected <= cached
                        if b != self.full_bucket_idx
                        else selected == 0
                    )
                    if not children:
                        model.Add(selected == 0)
                for child_id in set(children):
                    child = nodes[child_id]
                    consumers[child_id].append(selected)
                    model.Add(child["active"] >= selected)
                    model.Add(child["rank"] < node["rank"]).OnlyEnforceIf(selected)
                    model.Add(child["end"] <= node["start"]).OnlyEnforceIf(selected)
                    model.Add(child["release"] >= node["end"]).OnlyEnforceIf(selected)

                if duration:
                    interval = model.NewOptionalIntervalVar(
                        node["start"],
                        duration,
                        node["end"],
                        selected,
                        f"b{b}_c{cid}_e{e_idx}_run",
                    )
                    for engine in self.engineKeys(enode, cls["mem_space"]):
                        engines[engine].append(interval)
                        engine_work[engine].append(duration * selected)

                if enode.get("is_view"):
                    if (
                        not children
                        or nodes[children[0]]["mem_space"] != cls["mem_space"]
                    ):
                        model.Add(selected == 0)
                        continue
                    child = nodes[children[0]]
                    # A broadcast view may be logically larger than its backing
                    # allocation. Native inferView supplies strides and offsets.
                    # TODO: Export view byte spans to check backing-buffer bounds.
                    self.aliasBuffer(node, child, selected, is_view=True)
                    choices.append(selected)
                    if reserved is not None:
                        model.Add(selected == 0)
                    model.Add(selected + cached <= 1)
                    for child_id in set(children):
                        model.Add(
                            nodes[child_id]["read_end"] >= node["end"]
                        ).OnlyEnforceIf(selected)
                    continue

                inplace_by_child = {}
                safe_idxs = set(enode.get("safe_inplace_idxs", []))
                if (
                    not enode.get("is_input")
                    and not enode.get("is_cache")
                    and not enode.get("is_scatter")
                    and reserved is None
                ):
                    for child_id in sorted(
                        {children[idx] for idx in safe_idxs if 0 <= idx < len(children)}
                    ):
                        child = nodes[child_id]
                        # All occurrences of a repeated operand must permit
                        # aliasing; otherwise an unsafe argument is overwritten.
                        if any(
                            idx not in safe_idxs
                            for idx, operand in enumerate(children)
                            if operand == child_id
                        ):
                            continue
                        if child["mem_space"] != cls["mem_space"] or cls[
                            "raw_size_bytes"
                        ] > child["cls"]["raw_size_bytes"]:
                            continue
                        inplace = model.NewBoolVar(
                            f"b{b}_c{cid}_e{e_idx}_inplace{child_id}"
                        )
                        model.Add(inplace <= selected)
                        model.Add(child["protected"] == 0).OnlyEnforceIf(inplace)
                        model.Add(child["read_end"] <= node["start"]).OnlyEnforceIf(
                            inplace
                        )
                        self.aliasBuffer(node, child, inplace, is_view=False)
                        node["inplace_choices"].append((inplace, child_id))
                        inplace_by_child[child_id] = inplace
                        inplace_users[child_id].append(inplace)
                        choices.append(inplace)
                        model.Add(inplace + cached <= 1)
                    model.Add(sum(inplace_by_child.values()) <= selected)
                for child_id in set(children):
                    read_conditions = [selected]
                    if child_id in inplace_by_child:
                        read_conditions.append(inplace_by_child[child_id].Not())
                    model.Add(
                        nodes[child_id]["read_end"] >= node["end"]
                    ).OnlyEnforceIf(read_conditions)
                if enode.get("is_input"):
                    if children:
                        model.Add(selected == 0)
                    model.Add(node["start"] == 0).OnlyEnforceIf(selected)
                    model.Add(node["release"] == horizon).OnlyEnforceIf(selected)
                    model.Add(node["protected"] == 1).OnlyEnforceIf(selected)
                else:
                    model.Add(node["protected"] == 0).OnlyEnforceIf(
                        [selected, node["fresh"]]
                    )

            model.Add(sum(item[0] for item in node["selections"].values()) == active)
            if reserved is not None:
                model.Add(node["fresh"] == 0)
            else:
                model.Add(node["fresh"] + sum(choices) + cached == active)
            if cid == bucket["root_eclass_id"]:
                model.Add(active == 1)
                model.Add(node["release"] == horizon)
                model.Add(node["read_end"] == horizon)

        # Two destructive consumers cannot both reuse the same tensor version,
        # even when their executions are serialized and their engines differ.
        for users in inplace_users.values():
            model.Add(sum(users) <= 1)
        for node in nodes.values():
            for present, child_id in node["inplace_choices"]:
                for selected, enode in nodes[child_id]["selections"].values():
                    if enode.get("is_view"):
                        # TODO: Export view layouts before allowing in-place writes.
                        model.Add(present + selected <= 1)

        for cid, node in nodes.items():
            if cid != bucket["root_eclass_id"]:
                model.Add(node["active"] <= sum(consumers[cid]) + node["cached"])
        for base_eclass_id, (present, _) in self.cache_choices.items():
            matches = cache_nodes[base_eclass_id]
            model.Add(present <= sum(node["active"] for node in matches))
        # Canonical native classes have distinct base identities. Do not let an
        # ambiguous export introduce independent producers of one global buffer.
        # TODO: Model versioned global bindings before allowing multiple producers.
        for users in global_users.values():
            if len(users) > 1:
                model.Add(sum(users) <= 1)

        makespan = model.NewIntVar(0, horizon, f"b{b}_makespan")
        model.AddMaxEquality(makespan, [node["end"] for node in nodes.values()])

        for engine, intervals in engines.items():
            model.AddNoOverlap(intervals)
            model.Add(makespan >= sum(engine_work[engine]))

        for items in rectangles.values():
            model.AddNoOverlap2D(
                [item[0] for item in items], [item[1] for item in items]
            )

        weight = float(bucket.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid bucket weight {weight}")
        self.objective_terms.append(weight * makespan)
        self.bucket_models.append(
            {
                "bucket": bucket,
                "nodes": nodes,
                "makespan": makespan,
                "horizon": horizon,
            }
        )

    def decodeSolution(self, solver, status):
        global_buffers = [
            {
                "id": buf["id"],
                "mem_space": buf["mem_space"],
                "size": buf["size"],
                "offset": (
                    solver.Value(buf["page_offset"]) * self.alignment
                    if buf["page_offset"] is not None
                    else buf.get("raw_offset", 0)
                ),
                "start": 0,
                "end": self.persistent_end,
            }
            for buf in self.global_buffers
            if solver.Value(buf["present"])
        ]
        cached_nodes = [
            {"base_eclass_id": base_eclass_id, "mem_space": buf["mem_space"]}
            for base_eclass_id, (present, buf) in self.cache_choices.items()
            if solver.Value(present)
        ]
        extractions = []
        for bucket_model in self.bucket_models:
            nodes = bucket_model["nodes"]
            # Native execution consumes this order and synchronizes memory
            # hazards, but does not consume the numeric schedule below.
            # TODO: Match modeled timing to native dispatch and synchronization.
            order = sorted(
                (cid for cid, node in nodes.items() if solver.Value(node["active"])),
                key=lambda cid: (
                    solver.Value(nodes[cid]["start"]),
                    solver.Value(nodes[cid]["rank"]),
                    cid,
                ),
            )
            selection, costs, owners, schedule = {}, {}, {}, {}
            buffers = {buf["id"]: dict(buf) for buf in global_buffers}
            for position, cid in enumerate(order):
                node = nodes[cid]
                selected = next(
                    enode
                    for present, enode in node["selections"].values()
                    if solver.Value(present)
                )
                selection[str(cid)] = selected["enode_idx"]
                costs[str(cid)] = float(selected["cost"])
                owners[str(cid)] = solver.Value(node["owner"])
                schedule[str(cid)] = {
                    "start": solver.Value(node["start"]),
                    "end": solver.Value(node["end"]),
                }
                if solver.Value(node["fresh"]):
                    buffers[node["id"]] = {
                        "id": node["id"],
                        "mem_space": node["mem_space"],
                        "size": node["size"],
                        "offset": (
                            solver.Value(node["page_offset"]) * self.alignment
                            if node["page_offset"] is not None
                            else 0
                        ),
                        "start": position,
                        "end": position + 1,
                    }
                    if selected.get("is_input"):
                        buffers[node["id"]].update(start=0, end=self.persistent_end)
            for position, cid in enumerate(order):
                node = nodes[cid]
                selected = node["selections"][selection[str(cid)]][1]
                for used_id in [cid] + selected.get("children", []):
                    buf = buffers[owners[str(used_id)]]
                    buf["end"] = max(buf["end"], position + 1)
            root = bucket_model["bucket"]["root_eclass_id"]
            buffers[owners[str(root)]]["end"] = max(
                buffers[owners[str(root)]]["end"], len(order) + 1
            )
            extractions.append(
                {
                    "cost": solver.Value(bucket_model["makespan"]) / self.time_scale,
                    "selection_map": selection,
                    "order": order,
                    "eclass_to_buf": owners,
                    "eclass_to_cost": costs,
                    "buffers": list(buffers.values()),
                    "schedule": schedule,
                }
            )
        return {
            "solver": "ortools_full",
            "status": solver.StatusName(status),
            "cached_nodes": cached_nodes,
            "extractions": extractions,
            "objective": solver.ObjectiveValue(),
            "best_bound": solver.BestObjectiveBound(),
        }

    def solve(self):
        if not self.buckets:
            raise ValueError("OR-Tools full requires at least one bucket")
        if len({bucket["bucket_idx"] for bucket in self.buckets}) != len(self.buckets):
            raise ValueError("Duplicate bucket indices")
        if self.full_bucket_idx not in {bucket["bucket_idx"] for bucket in self.buckets}:
            raise ValueError("Full bucket index does not identify a bucket")
        if not self.model_built:
            self.createGlobalBuffers()
            for bucket in self.buckets:
                self.createBucket(bucket)
            # Weights apply to the whole bucket, including cache maintenance.
            self.model.Minimize(sum(self.objective_terms))
            self.model_built = True

        error = self.model.Validate()
        if error:
            raise ValueError(f"Invalid OR-Tools full model: {error}")

        solver = cp_model.CpSolver()
        # Leave the solve unbounded unless the caller supplies a time limit.
        # min_compile_seconds is a native search budget, not a maximum timeout.
        max_time_seconds = self.problem_data.get("max_time_seconds")
        if max_time_seconds is not None:
            max_time_seconds = float(max_time_seconds)
            if not math.isfinite(max_time_seconds) or max_time_seconds <= 0:
                raise ValueError("max_time_seconds must be finite and positive")
            solver.parameters.max_time_in_seconds = max_time_seconds
        num_workers = int(self.problem_data.get("num_workers", 12))
        if num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        solver.parameters.num_workers = num_workers
        solver.parameters.log_search_progress = bool(
            self.problem_data.get("print_progress", True)
        )
        status = solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"OR-Tools full found no feasible joint plan: {solver.StatusName(status)}"
            )
        return self.decodeSolution(solver, status)
