"""Joint cache, extraction, dispatch, bufferization and allocation with CP-SAT.

All decisions belong to one model and one Solve call. Optional rectangles pack
buffer lifetimes against 4KB page offsets; optional kernel intervals reserve engines.
Views and in-place choices propagate allocation lifetimes back to their owners.
Only decoding (including sorting the solved schedule) happens after Solve.
"""

import math
from collections import defaultdict, deque

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
        self.mem_caps = problem_data.get("mem_caps", {})
        self.next_buffer_id = (
            max((int(item["buffer_id"]) for item in self.preallocated), default=-1) + 1
        )
        self.global_buffers = []
        self.cache_choices = {}
        self.bucket_models = []
        self.objective_terms = []
        self.caps = {}

        self.preallocated_base_ids = {
            item["base_eclass_id"] for item in self.preallocated
        }
        self.preallocated_extents = {}
        for item in self.preallocated:
            ms = item["mem_space"]
            ms_key = memSpaceKey(ms)
            extent = int(item["offset"]) + int(item["size"])
            self.preallocated_extents[ms_key] = max(
                self.preallocated_extents.get(ms_key, 0), extent
            )

        # A finite arena bound avoids overflowing CP-SAT when native caps are UINT64_MAX.
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
            names = ("STORAGE", "CPP", "OPENCL", "CUDA")
            legacy_key = f"{mem_space['type']}{mem_space['idx']}"
            named_key = f"{names[mem_space['type']]}{mem_space['idx']}"
            cap = self.mem_caps.get(
                key, self.mem_caps.get(legacy_key, self.mem_caps.get(named_key, self.arena_bound))
            )
            self.caps[key] = max(0, min(int(cap), self.arena_bound))
        return self.caps[key]

    def memoryCapPages(self, mem_space):
        return self.memoryCap(mem_space) // self.alignment

    def newPageOffset(self, mem_space, name):
        if mem_space["type"] == 0:
            return None
        cap_pages = self.memoryCapPages(mem_space)
        return self.model.NewIntVar(0, cap_pages, f"{name}_page_offset")

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
        by_base_id = {}
        for item in self.preallocated:
            raw_offset = int(item["offset"])
            raw_page_offset = raw_offset // self.alignment
            raw_size = int(item["size"])
            buf = dict(
                item,
                id=int(item["buffer_id"]),
                size=raw_size,
                raw_offset=raw_offset,
                raw_page_offset=raw_page_offset,
                present=self.model.NewConstant(1),
            )
            buf["page_offset"] = (
                None
                if item["mem_space"]["type"] == 0
                else self.model.NewConstant(raw_page_offset)
            )
            buf["offset"] = buf["page_offset"]
            self.global_buffers.append(buf)
            by_base_id[item["base_eclass_id"]] = buf

        for candidate in self.candidates:
            base_eclass_id = candidate["base_eclass_id"]
            if base_eclass_id in self.cache_choices:
                raise ValueError(f"Duplicate cache candidate {base_eclass_id}")
            choices = []
            spaces = candidate.get("mem_spaces", [candidate["mem_space"]])
            seen = set()
            for mem_space in spaces:
                key = memSpaceKey(mem_space)
                if key in seen or mem_space["type"] == 0:
                    continue
                seen.add(key)
                if base_eclass_id in by_base_id and by_base_id[base_eclass_id]["mem_space"] != mem_space:
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
                        "size": int(candidate["size_bytes"]),
                        "present": present,
                        "page_offset": page_offset,
                        "offset": page_offset,
                    }
                    if "raw_size_bytes" in candidate:
                        buf["raw_size_bytes"] = candidate["raw_size_bytes"]
                    self.global_buffers.append(buf)
                choices.append((present, buf))
            self.model.Add(sum(choice[0] for choice in choices) <= 1)
            self.cache_choices[base_eclass_id] = choices
        self.preallocated_by_base_id = by_base_id

    def addRectangle(self, rectangles, buf, start, end, horizon, name):
        if buf["mem_space"]["type"] == 0 or buf.get("page_offset") is None:
            return None
        model = self.model
        present = buf["present"]
        page_size = (
            max(1, (buf["size"] + self.alignment - 1) // self.alignment)
            if buf["size"] > 0
            else 0
        )
        page_offset = buf["page_offset"]
        cap_pages = self.memoryCapPages(buf["mem_space"])

        model.Add(page_offset >= 0).OnlyEnforceIf(present)
        model.Add(page_offset + page_size <= cap_pages).OnlyEnforceIf(present)

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
        return lifetime

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
        if horizon >= 2**60:
            raise ValueError("OR-Tools full problem exceeds the integer time range")
        base_ids = {
            cls["id"]: cls["base_eclass_id"]
            for cls in classes.values()
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
            page_offset = self.newPageOffset(cls["mem_space"], name)
            node = {
                "cls": cls,
                "id": self.newBufferId(),
                "size": int(cls["size_bytes"]),
                "mem_space": cls["mem_space"],
                "active": model.NewBoolVar(f"{name}_active"),
                "fresh": model.NewBoolVar(f"{name}_fresh"),
                "protected": model.NewBoolVar(f"{name}_protected"),
                "owner": model.NewIntVar(0, max_owner, f"{name}_owner"),
                "page_offset": page_offset,
                "offset": page_offset,
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
            node["lifetime"] = self.addRectangle(
                rectangles, node, node["start"], node["release"], horizon, name
            )

        global_lifetimes = []
        for buf in self.global_buffers:
            lt = self.addRectangle(
                rectangles, buf, 0, horizon, horizon, f"b{b}_global{buf['id']}"
            )
            if lt is not None:
                global_lifetimes.append(lt)

        for cid, node in nodes.items():
            cls, active = node["cls"], node["active"]
            lid = base_ids[cid]
            cache_matches = [
                (present, buf)
                for present, buf in self.cache_choices.get(lid, [])
                if self.matchesBuffer(cls, buf)
            ]
            reserved = self.preallocated_by_base_id.get(lid)
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
                        base_ids[cid], []
                    )
                    if self.matchesBuffer(node["cls"], buf)
                ]
                model.Add(node["active"] <= sum(consumers[cid]) + sum(cache_roots))
        for lid, choices in self.cache_choices.items():
            for present, buf in choices:
                matches = [
                    node["active"]
                    for cid, node in nodes.items()
                    if base_ids[cid] == lid and self.matchesBuffer(node["cls"], buf)
                ]
                model.Add(present <= sum(matches))

        makespan = model.NewIntVar(0, horizon, f"b{b}_makespan")
        model.AddMaxEquality(makespan, [node["end"] for node in nodes.values()])

        for intervals in engines.values():
            model.AddNoOverlap(intervals)

        # Explicit lower-bound cut per engine
        for engine_key in engines.keys():
            engine_work = []
            for cid, node in nodes.items():
                for e_idx, (selected, enode) in node["selections"].items():
                    dur = durations.get((cid, e_idx))
                    if not dur or dur <= 0:
                        continue
                    engine_list = enode.get("engines") or [
                        {"type": 2, "idx": node["cls"]["mem_space"]["idx"]}
                        if node["cls"]["mem_space"]["type"] == 3
                        else {"type": 0, "idx": 0}
                    ]
                    if any(memSpaceKey(eng) == engine_key for eng in engine_list):
                        engine_work.append(dur * selected)
            if engine_work:
                model.Add(makespan >= sum(engine_work))

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
                "global_lifetimes": global_lifetimes,
            }
        )

    def runQuickExtraction(self):
        """Solves an acyclic 0-1 extraction IP in <0.2s to seed the joint model."""
        extract_model = cp_model.CpModel()
        is_cached = {}
        for cand in self.candidates:
            base_eclass_id = cand["base_eclass_id"]
            is_cached[base_eclass_id] = extract_model.NewBoolVar(f"q_cached_{base_eclass_id}")

        cand_by_ms = {}
        for cand in self.candidates:
            ms = cand["mem_space"]
            ms_key = memSpaceKey(ms)
            cand_by_ms.setdefault(ms_key, []).append(cand)

        for ms_key, cands_in_ms in cand_by_ms.items():
            sample_ms = cands_in_ms[0]["mem_space"]
            total_cap = self.memoryCap(sample_ms)
            cap = max(
                0,
                total_cap - self.preallocated_extents.get(ms_key, 0),
            )
            cache_terms = [
                cand["size_bytes"] * is_cached[cand["base_eclass_id"]]
                for cand in cands_in_ms
                if cand["base_eclass_id"] not in self.preallocated_base_ids
            ]
            if cache_terms:
                extract_model.Add(sum(cache_terms) <= cap)

        bucket_active = {}
        bucket_x = {}
        total_cost_terms = []

        full_b = self.problem_data.get("full_bucket_idx", self.buckets[0]["bucket_idx"] if self.buckets else 0)

        for b_dict in self.buckets:
            b = b_dict["bucket_idx"]
            b_weight = float(b_dict.get("weight", 1.0))
            root_id = b_dict["root_eclass_id"]
            classes_list = b_dict["classes"]
            classes_by_id = {cls["id"]: cls for cls in classes_list}
            count = len(classes_list)
            clean_eclasses = set(b_dict.get("clean_eclasses", []))
            base_ids = {
                cls["id"]: cls["base_eclass_id"]
                for cls in classes_by_id.values()
            }
            eclass_cache_lids = {}
            for cid, base_id in base_ids.items():
                lid = base_id
                if lid in is_cached:
                    eclass_cache_lids.setdefault(cid, []).append(lid)

            consumers = {}
            for cls in classes_list:
                cid = cls["id"]
                for enode in cls["enodes"]:
                    e_idx = enode["enode_idx"]
                    for ch in enode.get("children", []):
                        consumers.setdefault(ch, []).append((cid, e_idx))

            # Topological rank variables to strictly guarantee acyclicity
            rank_vars = {}
            for cls in classes_list:
                cid = cls["id"]
                rank_vars[cid] = extract_model.NewIntVar(0, count - 1, f"q_rank_{b}_{cid}")
                bucket_active[(b, cid)] = extract_model.NewBoolVar(f"q_act_{b}_{cid}")
                for enode in cls["enodes"]:
                    e_idx = enode["enode_idx"]
                    bucket_x[(b, cid, e_idx)] = extract_model.NewBoolVar(
                        f"q_x_{b}_{cid}_{e_idx}"
                    )

            for cls in classes_list:
                cid = cls["id"]
                act_var = bucket_active[(b, cid)]
                enode_vars = []
                lid_cls = base_ids[cid]
                reserved = self.preallocated_by_base_id.get(lid_cls)
                if reserved is not None and not self.matchesBuffer(cls, reserved):
                    reserved = None

                for enode in cls["enodes"]:
                    e_idx = enode["enode_idx"]
                    x_var = bucket_x[(b, cid, e_idx)]
                    enode_vars.append(x_var)

                    dur = self.duration(enode)
                    children = enode.get("children", [])
                    if (
                        dur is None
                        or any(ch not in classes_by_id for ch in children)
                        or enode.get("mem_space", cls["mem_space"]) != cls["mem_space"]
                    ):
                        extract_model.Add(x_var == 0)
                        continue

                    # Strict acyclicity constraint: every child must have smaller rank
                    for ch in children:
                        extract_model.Add(rank_vars[ch] < rank_vars[cid]).OnlyEnforceIf(x_var)

                    cost = float(enode.get("cost", 0.0))
                    is_cache = enode.get("is_cache", False)
                    is_input = enode.get("is_input", False)
                    is_scatter = enode.get("is_scatter", False)
                    is_view = enode.get("is_view", False)
                    base_eclass_id = enode.get("base_eclass_id", -1)

                    if 0 < cost < 1e8:
                        scaled_cost = int(round(cost * b_weight * 1000.0))
                        total_cost_terms.append(scaled_cost * x_var)

                    if is_view:
                        if (
                            not children
                            or classes_by_id[children[0]]["mem_space"] != cls["mem_space"]
                            or reserved is not None
                        ):
                            extract_model.Add(x_var == 0)
                            continue
                        for ch in children:
                            extract_model.Add(bucket_active[(b, ch)] >= x_var)
                    elif is_cache:
                        can_read = (cid in clean_eclasses) and (b != full_b)
                        if not can_read or base_eclass_id not in is_cached or not eclass_cache_lids.get(cid):
                            extract_model.Add(x_var == 0)
                        else:
                            extract_model.Add(x_var <= is_cached[base_eclass_id])
                    elif is_scatter:
                        for ch in children:
                            extract_model.Add(bucket_active[(b, ch)] >= x_var)
                        cache_vars = [is_cached[l] for l in eclass_cache_lids.get(cid, [])]
                        if cache_vars:
                            extract_model.Add(x_var <= sum(cache_vars))
                        else:
                            extract_model.Add(x_var == 0)
                    elif not is_input:
                        for ch in children:
                            extract_model.Add(bucket_active[(b, ch)] >= x_var)

                extract_model.Add(sum(enode_vars) == act_var)

                if cid == root_id:
                    extract_model.Add(act_var == 1)
                else:
                    p_terms = [
                        bucket_x[(b, p, e)] for p, e in consumers.get(cid, [])
                    ]
                    if p_terms:
                        extract_model.Add(act_var <= sum(p_terms))
                    else:
                        extract_model.Add(act_var == 0)

            for cand in self.candidates:
                base_eclass_id = cand["base_eclass_id"]
                for cid, mapped_base_id in base_ids.items():
                    if mapped_base_id == base_eclass_id:
                        extract_model.Add(bucket_active[(b, cid)] >= is_cached[base_eclass_id])

        # Penalize unnecessary caching when it provides no cross-bucket benefit
        for cand in self.candidates:
            total_cost_terms.append(1 * is_cached[cand["base_eclass_id"]])

        if total_cost_terms:
            extract_model.Minimize(sum(total_cost_terms))

        quick_solver = cp_model.CpSolver()
        quick_solver.parameters.max_time_in_seconds = 10.0
        quick_solver.parameters.num_workers = min(12, max(1, len(self.buckets) * 2))
        status = quick_solver.Solve(extract_model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        selected_cache = {
            cand["base_eclass_id"]
            for cand in self.candidates
            if quick_solver.Value(is_cached[cand["base_eclass_id"]]) == 1
        }
        selections_by_bucket = {}
        for b_dict in self.buckets:
            b = b_dict["bucket_idx"]
            b_sel = {}
            for cls in b_dict["classes"]:
                cid = cls["id"]
                if quick_solver.Value(bucket_active[(b, cid)]) == 1:
                    for enode in cls["enodes"]:
                        e_idx = enode["enode_idx"]
                        if quick_solver.Value(bucket_x[(b, cid, e_idx)]) == 1:
                            b_sel[cid] = e_idx
                            break
            selections_by_bucket[b] = b_sel

        return selected_cache, selections_by_bucket

    def addSolutionHint(self):
        """Builds a complete, valid initial hint using the quick extraction solve."""
        hinted_vars = set()

        def add_hint(var, val):
            if var is None:
                return
            idx = getattr(var, "index", None)
            if idx is None:
                try:
                    idx = var.Index()
                except AttributeError:
                    return
            pos_idx = idx if idx >= 0 else -idx - 1
            if pos_idx in hinted_vars:
                return
            hinted_vars.add(pos_idx)
            self.model.AddHint(var, int(val))

        quick_result = self.runQuickExtraction()
        if quick_result is None:
            return

        selected_cache, selections_by_bucket = quick_result

        # Track placed intervals in (start_time, release_time, page_offset, page_size)
        placed_intervals = defaultdict(list)
        for b_model in self.bucket_models:
            h = b_model["horizon"]
            for buf in self.global_buffers:
                if buf["mem_space"]["type"] != 0:
                    ms_key = memSpaceKey(buf["mem_space"])
                    p_off = buf.get("raw_page_offset", 0)
                    p_size = max(1, (buf["size"] + self.alignment - 1) // self.alignment)
                    placed_intervals[ms_key].append((0, h, p_off, p_size))

        # 1. Hint cache selections and global buffer page offsets
        for lid, choices in self.cache_choices.items():
            chosen_already = False
            for present, buf in choices:
                ms = buf["mem_space"]
                ms_key = memSpaceKey(ms)
                has_active_match = False
                if lid in selected_cache:
                    for b_model in self.bucket_models:
                        b = b_model["bucket"]["bucket_idx"]
                        b_active = selections_by_bucket.get(b, {})
                        for cid in b_active.keys():
                            node = b_model["nodes"][cid]
                            if node["cls"]["base_eclass_id"] == lid and self.matchesBuffer(node["cls"], buf):
                                has_active_match = True
                                break
                        if has_active_match:
                            break

                val = 1 if (has_active_match and not chosen_already) else 0
                if val:
                    chosen_already = True
                add_hint(present, val)

                if buf.get("page_offset") is not None and "raw_page_offset" not in buf:
                    p_size = max(1, (buf["size"] + self.alignment - 1) // self.alignment)
                    overlapping = [p for p in placed_intervals[ms_key]]
                    forbidden = sorted([(p[2], p[2] + p[3]) for p in overlapping], key=lambda x: x[0])
                    c_off = 0
                    for f_start, f_end in forbidden:
                        if c_off + p_size <= f_start:
                            break
                        if c_off < f_end:
                            c_off = f_end
                    buf["raw_page_offset"] = c_off
                    placed_intervals[ms_key].append((0, 2**40, c_off, p_size))
                    add_hint(buf["page_offset"], c_off)
                elif buf.get("page_offset") is not None:
                    add_hint(buf["page_offset"], buf.get("raw_page_offset", 0))

        # 2. Hint per-bucket schedule, lifetimes, and non-overlapping 2D layout
        for bucket_model in self.bucket_models:
            bucket = bucket_model["bucket"]
            b = bucket["bucket_idx"]
            nodes = bucket_model["nodes"]
            horizon = bucket_model["horizon"]
            makespan = bucket_model["makespan"]
            classes = {cls["id"]: cls for cls in bucket["classes"]}
            root_id = bucket["root_eclass_id"]
            base_ids = {
                cls["id"]: cls["base_eclass_id"]
                for cls in classes.values()
            }

            best_enode = selections_by_bucket.get(b, {})
            if root_id not in best_enode:
                continue

            active_cids = set(best_enode.keys())

            in_degree = {cid: 0 for cid in active_cids}
            parents_map = defaultdict(list)
            for cid in active_cids:
                e_idx = best_enode[cid]
                enode = next(e for e in classes[cid]["enodes"] if e["enode_idx"] == e_idx)
                for ch in enode.get("children", []):
                    if ch in active_cids:
                        parents_map[ch].append(cid)
                        in_degree[cid] += 1

            q = deque([cid for cid, deg in in_degree.items() if deg == 0])
            order = []
            while q:
                curr = q.popleft()
                order.append(curr)
                for p in parents_map[curr]:
                    in_degree[p] -= 1
                    if in_degree[p] == 0:
                        q.append(p)

            for cid in active_cids:
                if cid not in order:
                    order.append(cid)

            ranks = {cid: idx for idx, cid in enumerate(order)}

            curr_engine_time = defaultdict(lambda: 1)
            start_time = {}
            end_time = {}

            for cid in order:
                e_idx = best_enode[cid]
                enode = next(e for e in classes[cid]["enodes"] if e["enode_idx"] == e_idx)
                children = [ch for ch in enode.get("children", []) if ch in active_cids]
                parent_end = max([0] + [end_time.get(ch, 0) for ch in children])

                if enode.get("is_input"):
                    start_time[cid] = 0
                    end_time[cid] = 0
                elif enode.get("is_view"):
                    start_time[cid] = parent_end
                    end_time[cid] = parent_end
                else:
                    dur = self.duration(enode)
                    if dur is None or dur <= 0:
                        dur = 1
                    engine_list = enode.get("engines") or [
                        {"type": 2, "idx": classes[cid]["mem_space"]["idx"]}
                        if classes[cid]["mem_space"]["type"] == 3
                        else {"type": 0, "idx": 0}
                    ]
                    engine_keys = [memSpaceKey(eng) for eng in engine_list]
                    engine_ready = max([curr_engine_time[ek] for ek in engine_keys], default=1)

                    st = max(parent_end, engine_ready)
                    et = st + dur
                    start_time[cid] = st
                    end_time[cid] = et
                    for ek in engine_keys:
                        curr_engine_time[ek] = et

            release_time = {}
            read_end_time = {}
            for cid in order:
                e_idx = best_enode[cid]
                enode = next(e for e in classes[cid]["enodes"] if e["enode_idx"] == e_idx)
                lid = base_ids[cid]
                reserved = self.preallocated_by_base_id.get(lid)
                is_reserved = (
                    reserved is not None and self.matchesBuffer(classes[cid], reserved)
                )
                is_cached_node = (
                    lid in selected_cache
                    and any(self.matchesBuffer(classes[cid], c_buf) for _, c_buf in self.cache_choices.get(lid, []))
                )

                if cid == root_id or enode.get("is_input") or is_reserved or is_cached_node:
                    rel = horizon
                else:
                    consumer_ends = [end_time[p] for p in parents_map.get(cid, [])]
                    rel = max([end_time[cid], start_time[cid] + 1] + consumer_ends)

                release_time[cid] = rel
                read_end_time[cid] = horizon if cid == root_id else rel

            for cid in reversed(order):
                e_idx = best_enode.get(cid)
                if e_idx is not None:
                    enode = next(
                        e for e in classes[cid]["enodes"] if e["enode_idx"] == e_idx
                    )
                    if enode.get("is_view"):
                        children = [
                            ch for ch in enode.get("children", []) if ch in active_cids
                        ]
                        if children:
                            base_cid = children[0]
                            release_time[base_cid] = max(
                                release_time[base_cid], release_time[cid]
                            )
                            read_end_time[base_cid] = max(
                                read_end_time[base_cid], read_end_time[cid]
                            )

            page_offsets = {}
            is_fresh = {}
            owners = {}
            is_prot = {}

            # Preallocated buffers
            for cid in order:
                node = nodes[cid]
                cls = node["cls"]
                lid = base_ids[cid]
                reserved = self.preallocated_by_base_id.get(lid)
                if reserved is not None and self.matchesBuffer(cls, reserved):
                    is_fresh[cid] = 0
                    owners[cid] = reserved["id"]
                    is_prot[cid] = 1
                    page_offsets[cid] = reserved.get("raw_page_offset", 0)

            # Cached candidates
            for cid in order:
                if cid in page_offsets:
                    continue
                node = nodes[cid]
                cls = node["cls"]
                lid = base_ids[cid]
                if lid in selected_cache:
                    for _, c_buf in self.cache_choices.get(lid, []):
                        if self.matchesBuffer(cls, c_buf):
                            is_fresh[cid] = 0
                            owners[cid] = c_buf["id"]
                            is_prot[cid] = 1
                            page_offsets[cid] = c_buf.get("raw_page_offset", 0)
                            break

            # Views inherit owner, page_offset, and protected from base
            for cid in order:
                if cid in page_offsets:
                    continue
                node = nodes[cid]
                e_idx = best_enode[cid]
                enode = next(e for e in classes[cid]["enodes"] if e["enode_idx"] == e_idx)
                children = [ch for ch in enode.get("children", []) if ch in active_cids]

                if enode.get("is_view") and children and children[0] in page_offsets:
                    base_cid = children[0]
                    is_fresh[cid] = 0
                    owners[cid] = owners[base_cid]
                    is_prot[cid] = is_prot.get(base_cid, 0)
                    page_offsets[cid] = page_offsets[base_cid]

            # Dynamic fresh buffers: first-fit time-interval coloring
            for cid in order:
                if cid in page_offsets:
                    continue
                node = nodes[cid]
                cls = node["cls"]
                e_idx = best_enode[cid]
                enode = next(e for e in classes[cid]["enodes"] if e["enode_idx"] == e_idx)

                if cls["mem_space"]["type"] == 0:
                    page_offsets[cid] = 0
                    is_fresh[cid] = 1
                    owners[cid] = node["id"]
                    is_prot[cid] = 1 if enode.get("is_input") else 0
                else:
                    ms_key = memSpaceKey(cls["mem_space"])
                    p_size = max(1, (node["size"] + self.alignment - 1) // self.alignment)
                    st = start_time[cid]
                    rel = release_time[cid]

                    overlapping = [
                        p for p in placed_intervals[ms_key]
                        if max(p[0], st) < min(p[1], rel)
                    ]
                    forbidden = sorted([(p[2], p[2] + p[3]) for p in overlapping], key=lambda x: x[0])
                    cand_off = 0
                    for f_start, f_end in forbidden:
                        if cand_off + p_size <= f_start:
                            break
                        if cand_off < f_end:
                            cand_off = f_end

                    page_offsets[cid] = cand_off
                    placed_intervals[ms_key].append((st, rel, cand_off, p_size))
                    is_fresh[cid] = 1
                    owners[cid] = node["id"]
                    is_prot[cid] = 1 if enode.get("is_input") else 0

            # Hint active variables
            for cid in active_cids:
                node = nodes[cid]
                e_idx = best_enode[cid]

                add_hint(node["active"], 1)
                add_hint(node["fresh"], is_fresh[cid])
                add_hint(node["owner"], owners[cid])
                if node["page_offset"] is not None:
                    add_hint(node["page_offset"], page_offsets[cid])
                add_hint(node["protected"], is_prot.get(cid, 0))
                add_hint(node["rank"], ranks[cid])
                add_hint(node["start"], start_time[cid])
                add_hint(node["end"], end_time[cid])
                add_hint(node["release"], release_time[cid])
                add_hint(node["read_end"], read_end_time[cid])
                if node.get("lifetime") is not None:
                    add_hint(node["lifetime"], release_time[cid] - start_time[cid])

                selection_vars = set()
                for s_idx, (sel_var, _) in node["selections"].items():
                    selection_vars.add(sel_var)
                    add_hint(sel_var, 1 if s_idx == e_idx else 0)

                for present_var, _ in node["aliases"]:
                    if present_var not in selection_vars:
                        add_hint(present_var, 0)

            # Hint inactive variables so the hint is complete
            for cid, node in nodes.items():
                if cid not in active_cids:
                    add_hint(node["active"], 0)
                    add_hint(node["fresh"], 0)
                    add_hint(node["protected"], 0)
                    add_hint(node["owner"], node["id"])
                    if node["page_offset"] is not None:
                        add_hint(node["page_offset"], 0)
                    add_hint(node["rank"], 0)
                    add_hint(node["start"], 0)
                    add_hint(node["end"], 0)
                    add_hint(node["release"], 0)
                    add_hint(node["read_end"], 0)
                    if node.get("lifetime") is not None:
                        add_hint(node["lifetime"], 0)
                    for sel_var, _ in node["selections"].values():
                        add_hint(sel_var, 0)
                    for present_var, _ in node["aliases"]:
                        add_hint(present_var, 0)

            max_end = max([end_time.get(cid, 0) for cid in active_cids], default=0)
            add_hint(makespan, max_end)

            for lt in bucket_model.get("global_lifetimes", []):
                add_hint(lt, horizon)

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
                "end": 2**32 - 1,
            }
            for buf in self.global_buffers
            if solver.Value(buf["present"])
        ]
        cached_nodes = [
            {"base_eclass_id": lid, "mem_space": buf["mem_space"]}
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
                        "offset": (
                            solver.Value(node["page_offset"]) * self.alignment
                            if node["page_offset"] is not None
                            else 0
                        ),
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

        # Primary optimization: minimize total execution latency
        self.model.Minimize(sum(self.objective_terms))

        # Seed CP-SAT with an optimal extraction + topological schedule & placement hint
        self.addSolutionHint()

        error = self.model.Validate()
        if error:
            raise ValueError(f"Invalid OR-Tools full model: {error}")

        solver = cp_model.CpSolver()
        # solver.parameters.max_time_in_seconds = max(
        #     0.001,
        #     float(
        #         self.problem_data.get(
        #             "max_time_seconds",
        #             max(
        #                 120.0,
        #                 float(self.problem_data.get("min_compile_seconds", 0.0)),
        #             ),
        #         )
        #     ),
        # )
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
