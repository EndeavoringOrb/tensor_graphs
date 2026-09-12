"""Joint cache, extraction, dispatch, bufferization and allocation with CP-SAT.

All decisions belong to one model and one Solve call. Optional rectangles pack
buffer lifetimes against byte offsets; optional kernel intervals reserve engines.
Views and in-place choices propagate allocation lifetimes back to their owners.
Only decoding (including sorting the solved schedule) happens after Solve.
"""

import math
from collections import defaultdict

from ortools.sat.python import cp_model


def memSpaceKey(mem_space):
    return f"{mem_space['type']}:{mem_space['idx']}"


class OrtoolsSolver:
    alignment = 4096
    time_scale = 1000  # Native costs are milliseconds; schedule in microseconds.

    def __init__(self, problem_data):
        self.problem_data = problem_data
        self.model = cp_model.CpModel()
        self.buckets = problem_data.get("buckets", [])
        self.candidates = problem_data.get("candidates", [])
        self.preallocated = problem_data.get("preallocated_buffers", [])
        self.next_buffer_id = (
            max((int(item["buffer_id"]) for item in self.preallocated), default=-1) + 1
        )
        self.global_buffers = []
        self.cache_choices = {}
        self.bucket_models = []
        self.objective_terms = []
        self.memory_terms = []
        self.caps = {}
        # A finite arena bound avoids overflowing CP-SAT when native caps are
        # UINT64_MAX. Allocating every possible buffer separately is sufficient.
        self.arena_bound = (
            self.alignment
            + sum(int(item["offset"]) + int(item["size"]) for item in self.preallocated)
            + sum(int(item["size_bytes"]) for item in self.candidates)
            + sum(
                int(cls["size_bytes"])
                for bucket in self.buckets
                for cls in bucket["classes"]
            )
        )
        if self.arena_bound >= 2**60:
            raise ValueError("OR-Tools full problem exceeds the integer memory range")

    def newBufferId(self):
        buffer_id = self.next_buffer_id
        self.next_buffer_id += 1
        return buffer_id

    def memoryCap(self, mem_space):
        key = memSpaceKey(mem_space)
        if key not in self.caps:
            caps = self.problem_data.get("mem_caps", {})
            names = ("STORAGE", "CPP", "OPENCL", "CUDA")
            legacy_key = f"{mem_space['type']}{mem_space['idx']}"
            named_key = f"{names[mem_space['type']]}{mem_space['idx']}"
            cap = caps.get(
                key, caps.get(legacy_key, caps.get(named_key, self.arena_bound))
            )
            self.caps[key] = max(0, min(int(cap), self.arena_bound))
        return self.caps[key]

    def newOffset(self, mem_space, name):
        if mem_space["type"] == 0:
            return self.model.NewConstant(0)
        cap = self.memoryCap(mem_space)
        units = self.model.NewIntVar(0, cap // self.alignment, f"{name}_pages")
        offset = self.model.NewIntVar(0, cap, f"{name}_offset")
        self.model.Add(offset == units * self.alignment)
        return offset

    def duration(self, enode):
        cost = enode.get("cost")
        if cost is None or not math.isfinite(float(cost)) or float(cost) < 0:
            return None
        if enode.get("is_input") or enode.get("is_cache") or enode.get("is_view"):
            return 0
        return max(1, math.ceil(float(cost) * self.time_scale))

    def matchesBuffer(self, cls, buf):
        return (
            cls["mem_space"] == buf["mem_space"]
            and cls["size_bytes"] == buf["size"]
            and (
                "raw_size_bytes" not in buf
                or cls.get("raw_size_bytes") == buf["raw_size_bytes"]
            )
        )

    def createGlobalBuffers(self):
        by_logical = {}
        for item in self.preallocated:
            buf = dict(
                item, id=int(item["buffer_id"]), present=self.model.NewConstant(1)
            )
            buf["offset"] = self.model.NewConstant(int(item["offset"]))
            self.global_buffers.append(buf)
            by_logical[item["logical_id"]] = buf

        for candidate in self.candidates:
            lid = candidate["logical_id"]
            if lid in self.cache_choices:
                raise ValueError(f"Duplicate cache candidate {lid}")
            choices = []
            spaces = candidate.get("mem_spaces", [candidate["mem_space"]])
            seen = set()
            for mem_space in spaces:
                key = memSpaceKey(mem_space)
                if key in seen or mem_space["type"] == 0:
                    continue
                seen.add(key)
                if lid in by_logical and by_logical[lid]["mem_space"] != mem_space:
                    continue
                present = self.model.NewBoolVar(f"cache_{lid}_{key}")
                if lid in by_logical:
                    buf = by_logical[lid]
                else:
                    buf = {
                        "id": self.newBufferId(),
                        "logical_id": lid,
                        "mem_space": mem_space,
                        "size": int(candidate["size_bytes"]),
                        "present": present,
                        "offset": self.newOffset(mem_space, f"cache_{lid}_{key}"),
                    }
                    if "raw_size_bytes" in candidate:
                        buf["raw_size_bytes"] = candidate["raw_size_bytes"]
                    self.global_buffers.append(buf)
                choices.append((present, buf))
            self.model.Add(sum(choice[0] for choice in choices) <= 1)
            self.cache_choices[lid] = choices
        self.preallocated_by_logical = by_logical

    def addRectangle(self, rectangles, buf, start, end, horizon, name):
        if buf["mem_space"]["type"] == 0:
            return
        model = self.model
        present, size, offset = buf["present"], buf["size"], buf["offset"]
        cap = self.memoryCap(buf["mem_space"])
        model.Add(offset >= 0).OnlyEnforceIf(present)
        model.Add(offset + size <= cap).OnlyEnforceIf(present)
        lifetime = model.NewIntVar(0, horizon, f"{name}_lifetime")
        time_interval = model.NewOptionalIntervalVar(
            start, lifetime, end, present, f"{name}_live"
        )
        space_interval = model.NewOptionalIntervalVar(
            offset, size, offset + size, present, f"{name}_bytes"
        )
        rectangles[memSpaceKey(buf["mem_space"])].append(
            (time_interval, space_interval)
        )
        extent = model.NewIntVar(0, cap, f"{name}_extent")
        model.Add(extent == offset + size).OnlyEnforceIf(present)
        model.Add(extent == 0).OnlyEnforceIf(present.Not())
        self.memory_terms.append(extent)

    def aliasBuffer(self, node, child, present, is_view):
        model = self.model
        model.Add(node["owner"] == child["owner"]).OnlyEnforceIf(present)
        model.Add(node["offset"] == child["offset"]).OnlyEnforceIf(present)
        model.Add(node["protected"] == child["protected"]).OnlyEnforceIf(present)
        model.Add(child["release"] >= node["release"]).OnlyEnforceIf(present)
        if is_view:
            # All aliases of the old value must finish reading before overwrite.
            model.Add(child["read_end"] >= node["read_end"]).OnlyEnforceIf(present)

    def bindGlobalBuffer(self, node, buf, present, horizon):
        model = self.model
        model.Add(node["owner"] == buf["id"]).OnlyEnforceIf(present)
        model.Add(node["offset"] == buf["offset"]).OnlyEnforceIf(present)
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
        if horizon >= 2**60:
            raise ValueError("OR-Tools full problem exceeds the integer time range")
        logicals = {
            int(cid): int(lid)
            for cid, lid in bucket.get("eclass_to_logical", {}).items()
        }
        clean = set(bucket.get("clean_eclasses", []))
        rectangles, engines, consumers = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        nodes = {}
        max_owner = self.next_buffer_id + count
        for cid, cls in classes.items():
            name = f"b{b}_c{cid}"
            node = {
                "cls": cls,
                "id": self.newBufferId(),
                "size": int(cls["size_bytes"]),
                "mem_space": cls["mem_space"],
                "active": model.NewBoolVar(f"{name}_active"),
                "fresh": model.NewBoolVar(f"{name}_fresh"),
                "protected": model.NewBoolVar(f"{name}_protected"),
                "owner": model.NewIntVar(0, max_owner, f"{name}_owner"),
                "offset": self.newOffset(cls["mem_space"], name),
                "rank": model.NewIntVar(0, count - 1, f"{name}_rank"),
                "start": model.NewIntVar(0, horizon - 1, f"{name}_start"),
                "end": model.NewIntVar(0, horizon - 1, f"{name}_end"),
                "release": model.NewIntVar(0, horizon, f"{name}_release"),
                "read_end": model.NewIntVar(0, horizon, f"{name}_read_end"),
                "selections": {},
                "aliases": [],
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
            lid = logicals.get(cid, -1)
            # Only a full-sized class in the selected space owns a logical cache.
            cache_matches = [
                (present, buf)
                for present, buf in self.cache_choices.get(lid, [])
                if self.matchesBuffer(cls, buf)
            ]
            reserved = self.preallocated_by_logical.get(lid)
            if reserved is not None and not self.matchesBuffer(cls, reserved):
                reserved = None
            for present, buf in cache_matches:
                model.Add(active >= present)
                self.bindGlobalBuffer(node, buf, present, horizon)
            if reserved is not None:
                self.bindGlobalBuffer(node, reserved, active, horizon)
            else:
                model.Add(node["fresh"] + sum(item[0] for item in cache_matches) <= 1)

            choices = []
            for enode in cls["enodes"]:
                e_idx = enode["enode_idx"]
                if e_idx in node["selections"]:
                    raise ValueError(f"Duplicate enode index {e_idx} in class {cid}")
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
                    can_read = cid in clean and b != self.problem_data.get(
                        "full_bucket_idx", self.buckets[0]["bucket_idx"]
                    )
                    model.Add(
                        selected <= sum(item[0] for item in cache_matches)
                        if can_read
                        else selected == 0
                    )
                if enode.get("is_scatter"):
                    model.Add(selected <= sum(item[0] for item in cache_matches))
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
                    engine_list = enode.get("engines") or [
                        {"type": 2, "idx": cls["mem_space"]["idx"]}
                        if cls["mem_space"]["type"] == 3
                        else {"type": 0, "idx": 0}
                    ]
                    for engine in {memSpaceKey(engine) for engine in engine_list}:
                        engines[engine].append(interval)

                if enode.get("is_view"):
                    if (
                        not children
                        or nodes[children[0]]["mem_space"] != cls["mem_space"]
                    ):
                        model.Add(selected == 0)
                        continue
                    child = nodes[children[0]]
                    self.aliasBuffer(node, child, selected, is_view=True)
                    node["aliases"].append((selected, children[0]))
                    choices.append(selected)
                    # A view cannot initialize its own persistent allocation.
                    if reserved is not None:
                        model.Add(selected == 0)
                    for present, _ in cache_matches:
                        model.Add(selected + present <= 1)
                    for child_id in set(children):
                        model.Add(
                            nodes[child_id]["read_end"] >= node["end"]
                        ).OnlyEnforceIf(selected)
                    continue

                inplace_by_child = defaultdict(list)
                safe_idxs = enode.get("safe_inplace_idxs", [])
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
                        if child["mem_space"] != cls["mem_space"] or cls.get(
                            "raw_size_bytes", node["size"]
                        ) > child["cls"].get("raw_size_bytes", child["size"]):
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
                        node["aliases"].append((inplace, child_id))
                        inplace_by_child[child_id].append(inplace)
                        choices.append(inplace)
                        for present, _ in cache_matches:
                            model.Add(inplace + present <= 1)
                    model.Add(
                        sum(v for vals in inplace_by_child.values() for v in vals)
                        <= selected
                    )
                for child_id in set(children):
                    # In-place kernels read their overwritten input at the start;
                    # every other reader must finish before that overwrite starts.
                    model.Add(nodes[child_id]["read_end"] >= node["end"]).OnlyEnforceIf(
                        [selected] + [v.Not() for v in inplace_by_child[child_id]]
                    )
                if enode.get("is_input"):
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
                model.Add(
                    node["fresh"]
                    + sum(choices)
                    + sum(item[0] for item in cache_matches)
                    == active
                )
            if cid == bucket["root_eclass_id"]:
                model.Add(active == 1)
                model.Add(node["release"] == horizon)
                model.Add(node["read_end"] == horizon)

        # Disallow overwriting a view's base via a view input: inferView can add
        # offsets/strides, which a dense output kernel cannot inherit. This pass
        # runs after every class's alternatives have been constructed.
        for node in nodes.values():
            for present, child_id in node["aliases"]:
                if any(
                    present is sel and enode.get("is_view")
                    for sel, enode in node["selections"].values()
                ):
                    continue
                for selected, enode in nodes[child_id]["selections"].values():
                    if enode.get("is_view"):
                        model.Add(present + selected <= 1)

        for cid, node in nodes.items():
            if cid != bucket["root_eclass_id"]:
                cache_roots = [
                    present
                    for present, buf in self.cache_choices.get(
                        logicals.get(cid, -1), []
                    )
                    if self.matchesBuffer(node["cls"], buf)
                ]
                model.Add(node["active"] <= sum(consumers[cid]) + sum(cache_roots))
        for lid, choices in self.cache_choices.items():
            for present, buf in choices:
                matches = [
                    node["active"]
                    for cid, node in nodes.items()
                    if logicals.get(cid) == lid and self.matchesBuffer(node["cls"], buf)
                ]
                model.Add(present <= sum(matches))
        for intervals in engines.values():
            model.AddNoOverlap(intervals)
        for items in rectangles.values():
            model.AddNoOverlap2D(
                [item[0] for item in items], [item[1] for item in items]
            )
        makespan = model.NewIntVar(0, horizon, f"b{b}_makespan")
        model.AddMaxEquality(makespan, [node["end"] for node in nodes.values()])
        weight = float(bucket.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid bucket weight {weight}")
        self.objective_terms.append(weight * makespan)
        self.bucket_models.append(
            {"bucket": bucket, "nodes": nodes, "makespan": makespan, "horizon": horizon}
        )

    def decodeSolution(self, solver, status):
        global_buffers = [
            {
                "id": buf["id"],
                "mem_space": buf["mem_space"],
                "size": buf["size"],
                "offset": solver.Value(buf["offset"]),
                "start": 0,
                "end": 2**32 - 1,
            }
            for buf in self.global_buffers
            if solver.Value(buf["present"])
        ]
        cached_nodes = [
            {"logical_id": lid, "mem_space": buf["mem_space"]}
            for lid, choices in self.cache_choices.items()
            for present, buf in choices
            if solver.Value(present)
        ]
        extractions = []
        for bucket_model in self.bucket_models:
            nodes = bucket_model["nodes"]
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
                        "offset": solver.Value(node["offset"]),
                        "start": position,
                        "end": position + 1,
                    }
                    if selected.get("is_input"):
                        buffers[node["id"]].update(start=0, end=2**32 - 1)
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
        self.createGlobalBuffers()
        for bucket in self.buckets:
            self.createBucket(bucket)
        # Bounded secondary preference for smaller arenas and fewer caches.
        secondary_bound = max(
            1, len(self.memory_terms) * self.arena_bound + len(self.candidates)
        )
        cache_terms = [
            present for choices in self.cache_choices.values() for present, _ in choices
        ]
        self.model.Minimize(
            sum(self.objective_terms)
            + (sum(self.memory_terms) + sum(cache_terms)) * (1e-4 / secondary_bound)
        )
        error = self.model.Validate()
        if error:
            raise ValueError(f"Invalid OR-Tools full model: {error}")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(
            0.001,
            float(
                self.problem_data.get(
                    "max_time_seconds",
                    max(120.0, float(self.problem_data.get("min_compile_seconds", 0.0))),
                )
            ),
        )
        solver.parameters.num_workers = int(self.problem_data.get("num_workers", 12))
        solver.parameters.log_search_progress = bool(
            self.problem_data.get("print_progress", True)
        )
        status = solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"OR-Tools full found no feasible joint plan: {solver.StatusName(status)}"
            )
        return self.decodeSolution(solver, status)
