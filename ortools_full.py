import json
import heapq
import math
import sys
from collections import defaultdict

from ortools.sat.python import cp_model

from ortools_hints import compiledGraphsToHints


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

    def __init__(
        self,
        problem_data,
        primary_fixings=None,
        must_change_groups=None,
        incumbent_assignments=None,
    ):
        self.problem_data = problem_data
        self.model = cp_model.CpModel()
        self.buckets = problem_data["buckets"]
        self.caching_enabled = not problem_data["disable_caching"]
        self.candidates = problem_data.get("candidates", []) if self.caching_enabled else []
        self.preallocated = problem_data.get("preallocated_buffers", [])
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
        self.inplace_nodes = {}
        self.external_hints = compiledGraphsToHints(
            problem_data, problem_data.get("cpu_hints", [])
        ) if problem_data.get("cpu_hints") else {"cached": {}, "buckets": {}}
        self.model_built = False
        self.full_bucket_idx = problem_data["full_bucket_idx"]
        self.primary_fixings = dict(primary_fixings or {})
        self.must_change_groups = set(must_change_groups or ())
        self.incumbent_assignments = dict(incumbent_assignments or {})
        self.primary_metadata = None

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

    def addHint(self, var, val):
        """Apply one hint supplied by the native compiled-plan witness."""
        if var is not None and hasattr(var, "Index"):
            self.model.AddHint(var, int(val))

    @staticmethod
    def selectionKey(bucket_idx, eclass_id):
        return f"selection:{int(bucket_idx)}:{int(eclass_id)}"

    @staticmethod
    def inplaceKey(bucket_idx, eclass_id):
        return f"inplace:{int(bucket_idx)}:{int(eclass_id)}"

    @staticmethod
    def cacheKey(base_eclass_id):
        return f"cache:{int(base_eclass_id)}"

    def addInplaceSourceConstraints(self, node, child_id, inplace):
        """Make an in-place choice follow a selected view chain.

        Native bufferization resolves a safe in-place operand to the underlying
        non-view eclass before checking size, memory space, and last use.  The
        CP model must apply those checks to the same terminal source while
        retaining the direct child for source reconstruction during decoding.
        """

        def visit(source_id, conditions, seen):
            source = self.inplace_nodes.get(source_id)
            if source is None or source_id in seen:
                self.model.Add(inplace == 0).OnlyEnforceIf(conditions)
                return

            next_seen = seen | {source_id}
            for selected, enode in source["selections"].values():
                path = conditions + [selected]
                children = enode.get("children", [])
                if enode.get("is_view"):
                    if not children:
                        self.model.Add(inplace == 0).OnlyEnforceIf(path)
                    else:
                        visit(int(children[0]), path, next_seen)
                    continue

                compatible = (
                    not enode.get("is_input")
                    and not enode.get("is_cache")
                    and enode.get("mem_space", source["mem_space"])
                    == node["mem_space"]
                    and int(node["cls"]["raw_size_bytes"])
                    <= int(source["cls"]["raw_size_bytes"])
                )
                if not compatible:
                    self.model.Add(inplace == 0).OnlyEnforceIf(path)
                else:
                    # The terminal source, rather than a view's temporary
                    # eclass, must be dead before the overwrite begins.
                    self.model.Add(source["read_end"] <= node["start"]).OnlyEnforceIf(path)

        visit(int(child_id), [inplace], set())

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
            self.caps[key] = cap
        return self.caps[key]

    def memoryCapPages(self, mem_space):
        return self.memoryCap(mem_space) // self.alignment

    def newPageOffset(self, mem_space, size, name):
        if mem_space["type"] == 0:
            return None
        cap = self.memoryCap(mem_space)
        if cap < size:
            return self.model.NewIntVar(0, 0, f"{name}_page_offset")
        max_pages = (cap - size) // self.alignment
        return self.model.NewIntVar(0, max_pages, f"{name}_page_offset")

    def duration(self, enode):
        try:
            cost = float(enode["cost"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(cost) or cost < 0:
            return None
        if any(enode.get(flag) for flag in ("is_input", "is_cache", "is_view")):
            return 0
        scaled_cost = cost * self.time_scale
        if not math.isfinite(scaled_cost) or scaled_cost >= self.integer_limit:
            raise ValueError("Kernel cost exceeds the integer time range")
        return max(1, math.ceil(scaled_cost))

    def engineKeys(self, enode, mem_space):
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
                size = self.allocationSize(candidate)
                page_offset = self.newPageOffset(mem_space, size, f"cache_{base_eclass_id}_{key}")
                buf = {
                    "id": self.newBufferId(),
                    "base_eclass_id": base_eclass_id,
                    "mem_space": mem_space,
                    "size": size,
                    "present": present,
                    "page_offset": page_offset,
                }
                if "raw_size_bytes" in candidate:
                    buf["raw_size_bytes"] = candidate["raw_size_bytes"]
                
                if page_offset is not None and type(page_offset) is not int:
                    self.model.Add(page_offset == 0).OnlyEnforceIf(present.Not())
                self.global_buffers.append(buf)
            self.cache_choices[base_eclass_id] = (present, buf)
            cached_value = self.external_hints.get("cached", {}).get(base_eclass_id)
            if cached_value is not None:
                self.addHint(present, cached_value)
                if (
                    not cached_value
                    and base_eclass_id not in by_base_id
                    and buf.get("page_offset") is not None
                ):
                    self.addHint(buf["page_offset"], 0)
            
        self.preallocated_by_base_id = by_base_id

    def addRectangle(self, rectangles, buf, start, end, horizon, name):
        if buf["mem_space"]["type"] == 0 or buf.get("page_offset") is None:
            return
        model = self.model
        present = buf["present"]
        page_size = self.alignedSize(buf["size"]) // self.alignment
        page_offset = buf["page_offset"]

        if self.memoryCap(buf["mem_space"]) < buf["size"]:
            model.Add(present == 0)
            return

        if page_size == 0:
            return

        if type(start) is int and type(end) is int:
            lifetime = end - start
        else:
            lifetime = model.NewIntVar(0, horizon, f"{name}_lifetime")
            model.Add(lifetime == end - start)
            buf["lifetime"] = lifetime

        time_interval = model.NewOptionalIntervalVar(
            start, lifetime, end, present, f"{name}_live"
        )
        space_interval = model.NewOptionalIntervalVar(
            page_offset, page_size, page_offset + page_size, present, f"{name}_pages"
        )
        rectangles[memSpaceKey(buf["mem_space"])].append(
            (time_interval, space_interval, page_size)
        )

    def aliasBuffer(self, node, child, present, is_view):
        model = self.model
        model.Add(node["protected"] == child["protected"]).OnlyEnforceIf(present)
        model.Add(child["release"] >= node["release"]).OnlyEnforceIf(present)
        if is_view:
            model.Add(child["read_end"] >= node["read_end"]).OnlyEnforceIf(present)

    def bindGlobalBuffer(self, node, buf, present, horizon):
        model = self.model
        model.Add(node["protected"] == 1).OnlyEnforceIf(present)
        model.Add(node["release"] == horizon).OnlyEnforceIf(present)

    def _get_reachable_classes(self, bucket):
        """Prunes dead and ungrounded classes before model generation to eliminate wasted presolve cycles."""
        classes = {cls["id"]: cls for cls in bucket["classes"]}
        root_id = bucket["root_eclass_id"]

        grounded = set()
        parent_enodes = defaultdict(list)
        remaining = {}

        for cid, cls in classes.items():
            for enode in cls["enodes"]:
                key = (cid, enode["enode_idx"])
                children = [c for c in enode.get("children", []) if c in classes]
                u_children = set(children)
                remaining[key] = len(u_children)
                if len(u_children) == 0:
                    grounded.add(cid)
                for child in u_children:
                    parent_enodes[child].append((cid, key))

        queue = list(grounded)
        while queue:
            curr = queue.pop()
            for p_cid, p_key in parent_enodes[curr]:
                if remaining[p_key] > 0:
                    remaining[p_key] -= 1
                    if remaining[p_key] == 0:
                        if p_cid not in grounded:
                            grounded.add(p_cid)
                            queue.append(p_cid)

        reachable = set()
        roots = [root_id]
        if self.caching_enabled:
            for cand in self.candidates:
                base_id = cand["base_eclass_id"]
                for cid, cls in classes.items():
                    if cls.get("base_eclass_id") == base_id and cid in grounded:
                        roots.append(cid)

        q = [r for r in roots if r in grounded]
        while q:
            curr = q.pop()
            if curr in reachable:
                continue
            reachable.add(curr)
            cls = classes[curr]
            for enode in cls["enodes"]:
                children = [c for c in enode.get("children", []) if c in classes and c in grounded]
                if len(set(children)) == len(set(enode.get("children", []))):
                    for child in children:
                        if child not in reachable:
                            q.append(child)

        return reachable

    def applyExternalHints(self, plan, nodes, horizon, makespan):
        """Hint the CP model with the native extraction choices.

        The native witness and this model use the same selected eclasses,
        alias sources, lifetimes, and arena offsets.  Every variable belonging
        to this bucket is assigned so CP-SAT can validate the native witness
        before searching for an improvement.
        """
        hinted_nodes = plan.get("nodes", {})
        self.addHint(makespan, int(plan.get("makespan", 0)))

        for cid, node in nodes.items():
            hint = hinted_nodes.get(cid)
            active = hint is not None and int(hint.get("active", 0)) != 0

            self.addHint(node["active"], int(active))
            self.addHint(node["fresh"], int(hint.get("fresh", 0)) if active else 0)
            self.addHint(node["protected"], int(hint.get("protected", 0)) if active else 0)
            self.addHint(node["start"], int(hint.get("start", 0)) if active else 0)
            self.addHint(node["end"], int(hint.get("end", 0)) if active else 0)
            read_end = int(hint.get("read_end", 0)) if active else 0
            if active and int(hint.get("release_horizon", 0)):
                read_end = horizon if int(cid) == int(plan.get("root_id", -1)) else read_end
            release = int(hint.get("release", 0)) if active else 0
            if active and int(hint.get("release_horizon", 0)):
                release = horizon
            self.addHint(node["read_end"], read_end)
            self.addHint(node["release"], release)
            if node["page_offset"] is not None:
                self.addHint(node["page_offset"], int(hint.get("page_offset", 0)) if active else 0)
            if "lifetime" in node:
                lifetime = (
                    release - (int(hint.get("start", 0)) if active else 0)
                    if active
                    else 0
                )
                self.addHint(node["lifetime"], lifetime)

            selection = hint.get("selection") if active else None
            for e_idx, (selection_var, _) in node["selections"].items():
                self.addHint(selection_var, int(selection == e_idx))

            inplace_child = hint.get("inplace") if active else None
            for inplace_var, child_id, enode_idx in node["inplace_choices"]:
                self.addHint(
                    inplace_var,
                    int(active and selection == enode_idx and inplace_child == child_id),
                )


    def createBucket(self, bucket):
        model = self.model
        b = bucket["bucket_idx"]

        # 1. Prune dead classes prior to model construction
        reachable = self._get_reachable_classes(bucket)
        classes = {cls["id"]: cls for cls in bucket["classes"] if cls["id"] in reachable}
        if (
            len(classes) == 0
            or bucket["root_eclass_id"] not in classes
        ):
            raise ValueError(f"Invalid eclass ids in bucket {b}")
        count = len(classes)
        durations = {
            (cid, enode["enode_idx"]): self.duration(enode)
            for cid, cls in classes.items()
            for enode in cls["enodes"]
        }

        # 2. Reserve enough integer time for the native CPU witness.  The
        # witness itself is produced by the C++ iterative search and is only
        # a starting point for this model.
        plan = self.external_hints.get("buckets", {}).get(b)

        # 3. Enforce mathematical overflow guard against CP-SAT int64_t::max() limits
        total_pages = sum(
            self.alignedSize(cls["size_bytes"]) // self.alignment
            for cls in classes.values()
        ) + sum(
            self.alignedSize(buf["size"]) // self.alignment
            for buf in self.global_buffers
            if buf["mem_space"]["type"] != 0
        )
        max_safe_horizon = (2**62 - 1) // max(1, total_pages)

        desired_horizon = (
            1
            + count
            + sum(
                max(
                    (
                        durations.get((cid, enode["enode_idx"])) or 0
                        for enode in cls["enodes"]
                    ),
                    default=0,
                )
                for cid, cls in classes.items()
            )
        )
        if plan is not None:
            desired_horizon = max(desired_horizon, int(plan["makespan"]) + 1)

        horizon = min(desired_horizon, max_safe_horizon)
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

        for cid, cls in classes.items():
            name = f"b{b}_c{cid}"
            size = self.allocationSize(cls)
            page_offset = self.newPageOffset(cls["mem_space"], size, name)
            node = {
                "cls": cls,
                "id": self.newBufferId(),
                "size": size,
                "mem_space": cls["mem_space"],
                "active": model.NewBoolVar(f"{name}_active"),
                "fresh": model.NewBoolVar(f"{name}_fresh"),
                "protected": model.NewBoolVar(f"{name}_protected"),
                "page_offset": page_offset,
                "start": model.NewIntVar(0, horizon - 1, f"{name}_start"),
                "end": model.NewIntVar(0, horizon - 1, f"{name}_end"),
                "release": model.NewIntVar(0, horizon, f"{name}_release"),
                "read_end": model.NewIntVar(0, horizon, f"{name}_read_end"),
                "selections": {},
                "inplace_choices": [],
                "cached": 0,
                "sources": [],
            }
            node["present"] = node["fresh"]
            nodes[cid] = node
            
            node["sources"].append((node["fresh"], node["id"], False))

            if page_offset is not None and type(page_offset) is not int:
                model.Add(page_offset == 0).OnlyEnforceIf(node["fresh"].Not())

            model.Add(node["release"] >= node["start"] + 1).OnlyEnforceIf(
                node["active"]
            )
            model.Add(node["read_end"] >= node["end"]).OnlyEnforceIf(
                node["active"]
            )
            model.Add(node["release"] >= node["read_end"]).OnlyEnforceIf(
                node["active"]
            )
            
            for field in ("start", "end", "read_end", "release", "protected"):
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
                model.AddImplication(cached, active)
                self.bindGlobalBuffer(node, buf, cached, horizon)
            cached = node["cached"]
            reserved = self.preallocated_by_base_id.get(base_eclass_id)
            if reserved is not None and not self.matchesBuffer(cls, reserved):
                reserved = None
                
            if reserved is not None:
                self.bindGlobalBuffer(node, reserved, active, horizon)
                global_users[reserved["id"]].append(active)
                node["sources"].append((active, reserved["id"], False))
            elif cache_matches:
                global_users[cache_choice[1]["id"]].append(cached)
                node["sources"].append((cached, cache_choice[1]["id"], False))

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
                    continue
                selected = model.NewBoolVar(f"b{b}_c{cid}_e{e_idx}")
                node["selections"][e_idx] = (selected, enode)
                duration = durations.get((cid, e_idx))
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
                    can_read = (
                        cid in clean
                        and b != self.full_bucket_idx
                        and (
                            "clean_buckets" not in candidate
                            or b in candidate["clean_buckets"]
                        )
                        and enode.get("base_eclass_id", base_eclass_id) == base_eclass_id
                    )
                    if can_read and type(cached) is not int:
                        model.AddImplication(selected, cached)
                    else:
                        model.Add(selected == 0)
                    if children:
                        model.Add(selected == 0)
                if enode.get("is_scatter"):
                    if b != self.full_bucket_idx and type(cached) is not int:
                        model.AddImplication(selected, cached)
                    else:
                        model.Add(selected == 0)
                    if not children:
                        model.Add(selected == 0)
                for child_id in set(children):
                    child = nodes[child_id]
                    consumers[child_id].append(selected)
                    model.AddImplication(selected, child["active"])
                    model.Add(child["end"] <= node["start"]).OnlyEnforceIf(selected)

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
                    self.aliasBuffer(node, child, selected, is_view=True)
                    choices.append(selected)
                    node["sources"].append((selected, children[0], True))
                    
                    if reserved is not None:
                        model.Add(selected == 0)
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
                        if any(
                            idx not in safe_idxs
                            for idx, operand in enumerate(children)
                            if operand == child_id
                        ):
                            continue
                        if child["mem_space"] != cls["mem_space"]:
                            continue
                        inplace = model.NewBoolVar(
                            f"b{b}_c{cid}_e{e_idx}_inplace{child_id}"
                        )
                        model.AddImplication(inplace, selected)
                        model.Add(child["protected"] == 0).OnlyEnforceIf(inplace)
                        model.Add(child["read_end"] <= node["start"]).OnlyEnforceIf(
                            inplace
                        )
                        self.aliasBuffer(node, child, inplace, is_view=False)
                        node["inplace_choices"].append((inplace, child_id, e_idx))
                        node["sources"].append((inplace, child_id, True))
                        
                        inplace_by_child[child_id] = inplace
                        inplace_users[child_id].append(inplace)
                        choices.append(inplace)
                            
                    if inplace_by_child:
                        if len(inplace_by_child) > 1:
                            model.AddAtMostOne(list(inplace_by_child.values()))
                        for inplace in inplace_by_child.values():
                            model.AddImplication(inplace, selected)
                            
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

            model.AddExactlyOne([item[0] for item in node["selections"].values()] + [node["active"].Not()])
            
            if reserved is not None:
                model.Add(node["fresh"] == 0)
            else:
                terms = [node["fresh"], node["active"].Not()] + choices
                if type(cached) is not int:
                    terms.append(cached)
                model.AddExactlyOne(terms)
                
            if cid == bucket["root_eclass_id"]:
                model.Add(active == 1)
                model.Add(node["release"] == horizon)
                model.Add(node["read_end"] == horizon)

        for users in inplace_users.values():
            if len(users) > 1:
                model.AddAtMostOne(users)
                
        # Apply native-compatible terminal-source checks after every class has
        # had its selection variables created.  This allows an in-place choice
        # against a direct view child while still enforcing the underlying
        # source's size, protection, and last-read constraints.
        self.inplace_nodes = nodes
        for node in nodes.values():
            for present, child_id, _ in node["inplace_choices"]:
                self.addInplaceSourceConstraints(node, child_id, present)

        for cid, node in nodes.items():
            if cid != bucket["root_eclass_id"]:
                or_terms = consumers[cid][:]
                if type(node["cached"]) is not int:
                    or_terms.append(node["cached"])
                if not or_terms:
                    model.Add(node["active"] == 0)
                else:
                    model.AddBoolOr(or_terms).OnlyEnforceIf(node["active"])
                    
        for base_eclass_id, (present, _) in self.cache_choices.items():
            matches = cache_nodes[base_eclass_id]
            if not matches:
                model.Add(present == 0)
            else:
                model.AddBoolOr([node["active"] for node in matches]).OnlyEnforceIf(present)
                
        for users in global_users.values():
            if len(users) > 1:
                model.AddAtMostOne(users)

        makespan = model.NewIntVar(0, horizon, f"b{b}_makespan")
        
        for node in nodes.values():
            model.Add(makespan >= node["end"])

        for engine, intervals in engines.items():
            model.AddNoOverlap(intervals)
            model.Add(makespan >= sum(engine_work[engine]))

        for ms_key, items in rectangles.items():
            model.AddNoOverlap2D(
                [item[0] for item in items], [item[1] for item in items]
            )
            
            # Redundant Cumulative constraint for global memory capacity reasoning
            cap_pages = self.caps[ms_key] // self.alignment
            model.AddCumulative(
                [item[0] for item in items], [item[2] for item in items], cap_pages
            )

        # The native planner already found a valid full-bucket witness. Keep
        # the full bucket in the model for cache initialization and ordering,
        # but exclude its makespan from the optimization objective. Keep the
        # legacy default for hand-built single-bucket test problems that omit
        # the weight field; exported sessions always include the full bucket's
        # explicit zero weight.
        include_in_objective = b != self.full_bucket_idx or (
            len(self.buckets) == 1 and "weight" not in bucket
        )
        if include_in_objective:
            weight = float(bucket.get("weight", 1.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"Invalid bucket weight {weight}")
            self.objective_terms.append(weight * makespan)
        
        if plan is not None:
            self.applyExternalHints(plan, nodes, horizon, makespan)
        
        self.bucket_models.append(
            {
                "bucket": bucket,
                "nodes": nodes,
                "makespan": makespan,
                "horizon": horizon,
            }
        )

    def _addSelectionFixing(self, node, value):
        selected_vars = list(node["selections"].items())
        for enode_idx, (var, _) in selected_vars:
            var_value = value is not None and int(enode_idx) == int(value)
            self.model.Add(var == int(var_value))

    @staticmethod
    def _inplaceAssignmentMatches(value, child_id, enode_idx):
        """Match an in-place assignment without losing its selected enode."""
        if value is None:
            return False
        if not isinstance(value, dict):
            raise ValueError(
                "In-place assignments must be None or an object with "
                "enode_idx and child_id"
            )
        return (
            int(value["child_id"]) == int(child_id)
            and int(value["enode_idx"]) == int(enode_idx)
        )

    def _addInplaceFixing(self, node, value):
        for inplace_var, child_id, enode_idx in node["inplace_choices"]:
            var_value = self._inplaceAssignmentMatches(
                value, child_id, enode_idx
            )
            self.model.Add(inplace_var == int(var_value))

    def _addPrimaryFixings(self):
        if not self.primary_fixings and not self.must_change_groups:
            return

        group_literals = []
        for bucket_model in self.bucket_models:
            bucket = bucket_model["bucket"]
            bucket_idx = int(bucket["bucket_idx"])
            for eclass_id, node in bucket_model["nodes"].items():
                selection_key = self.selectionKey(bucket_idx, eclass_id)
                if selection_key in self.primary_fixings:
                    self._addSelectionFixing(
                        node, self.primary_fixings[selection_key]
                    )
                if selection_key in self.must_change_groups:
                    incumbent = self.incumbent_assignments.get(selection_key)
                    for enode_idx, (var, _) in node["selections"].items():
                        if incumbent is None:
                            group_literals.append(var)
                        else:
                            group_literals.append(
                                var.Not()
                                if int(enode_idx) == int(incumbent)
                                else var
                            )

                inplace_key = self.inplaceKey(bucket_idx, eclass_id)
                if inplace_key in self.primary_fixings:
                    self._addInplaceFixing(
                        node,
                        self.primary_fixings[inplace_key],
                    )
                if inplace_key in self.must_change_groups:
                    incumbent = self.incumbent_assignments.get(inplace_key)
                    for inplace_var, child_id, enode_idx in node["inplace_choices"]:
                        if incumbent is None:
                            group_literals.append(inplace_var)
                        else:
                            group_literals.append(
                                inplace_var.Not()
                                if self._inplaceAssignmentMatches(
                                    incumbent,
                                    child_id,
                                    enode_idx,
                                )
                                else inplace_var
                            )

        for base_eclass_id, (present, _) in self.cache_choices.items():
            cache_key = self.cacheKey(base_eclass_id)
            if cache_key in self.primary_fixings:
                self.model.Add(
                    present == int(bool(self.primary_fixings[cache_key]))
                )
            if cache_key in self.must_change_groups:
                incumbent = bool(self.incumbent_assignments.get(cache_key, 0))
                group_literals.append(present.Not() if incumbent else present)

        if self.must_change_groups:
            if not group_literals:
                self.model.AddBoolOr([])
            else:
                self.model.AddBoolOr(group_literals)

    def getNeighborhoodMetadata(self):
        """Return selector-facing metadata for the complete CP model.

        The metadata intentionally describes primary decision groups rather than
        individual timing/allocation variables.  Derived variables remain free
        during an LNS repair solve and are propagated by the exact model.
        """
        groups = {}
        adjacency = defaultdict(set)

        for bucket_model in self.bucket_models:
            bucket = bucket_model["bucket"]
            bucket_idx = int(bucket["bucket_idx"])
            nodes = bucket_model["nodes"]
            for eclass_id, node in nodes.items():
                cls = node["cls"]
                key = self.selectionKey(bucket_idx, eclass_id)
                enodes = list(cls.get("enodes", []))
                costs = []
                for enode in enodes:
                    try:
                        cost = float(enode.get("cost", 0.0) or 0.0)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if math.isfinite(cost):
                        costs.append(cost)
                groups[key] = {
                    "key": key,
                    "kind": "selection",
                    "bucket_idx": bucket_idx,
                    "eclass_id": int(eclass_id),
                    "base_eclass_id": int(cls.get("base_eclass_id", eclass_id)),
                    "choice_count": len(node["selections"]),
                    "selectable": len(node["selections"]) > 1,
                    "features": [
                        float(len(node["selections"])),
                        float(min(costs, default=0.0)),
                        float(max(costs, default=0.0)),
                        float(cls.get("size_bytes", 0)),
                        float(cls.get("raw_size_bytes", 0)),
                        float(cls.get("mem_space", {}).get("type", 0)),
                        float(cls.get("mem_space", {}).get("idx", 0)),
                    ],
                }
                adjacency.setdefault(key, set())

                if node["inplace_choices"]:
                    inplace_key = self.inplaceKey(bucket_idx, eclass_id)
                    groups[inplace_key] = {
                        "key": inplace_key,
                        "kind": "inplace",
                        "bucket_idx": bucket_idx,
                        "eclass_id": int(eclass_id),
                        "base_eclass_id": int(
                            cls.get("base_eclass_id", eclass_id)
                        ),
                        "choice_count": len(node["inplace_choices"]),
                        "selectable": True,
                        "features": [
                            1.0,
                            float(len(node["inplace_choices"])),
                            float(cls.get("size_bytes", 0)),
                            float(cls.get("raw_size_bytes", 0)),
                            float(cls.get("mem_space", {}).get("type", 0)),
                            float(cls.get("mem_space", {}).get("idx", 0)),
                            0.0,
                        ],
                    }
                    adjacency.setdefault(inplace_key, set()).add(key)
                    adjacency[key].add(inplace_key)

                for enode in enodes:
                    for child_id in enode.get("children", []):
                        child_key = self.selectionKey(bucket_idx, child_id)
                        if child_key in groups:
                            adjacency[key].add(child_key)
                            adjacency[child_key].add(key)

        # Add dataflow edges in a second pass so class ordering in the input
        # cannot hide an edge to a class that was encountered later.
        for bucket_model in self.bucket_models:
            bucket_idx = int(bucket_model["bucket"]["bucket_idx"])
            for eclass_id, node in bucket_model["nodes"].items():
                key = self.selectionKey(bucket_idx, eclass_id)
                for enode in node["cls"].get("enodes", []):
                    for child_id in enode.get("children", []):
                        child_key = self.selectionKey(bucket_idx, child_id)
                        if child_key in groups:
                            adjacency[key].add(child_key)
                            adjacency[child_key].add(key)

        for base_eclass_id in self.cache_choices:
            cache_key = self.cacheKey(base_eclass_id)
            groups[cache_key] = {
                "key": cache_key,
                "kind": "cache",
                "base_eclass_id": int(base_eclass_id),
                "choice_count": 2,
                "selectable": True,
                "features": [2.0, float(base_eclass_id), 0.0, 0.0, 0.0, 0.0, 1.0],
            }
            adjacency.setdefault(cache_key, set())
            for key, group in groups.items():
                if group.get("base_eclass_id") == int(base_eclass_id):
                    adjacency[cache_key].add(key)
                    adjacency[key].add(cache_key)

        return {
            "groups": groups,
            "adjacency": {
                key: sorted(neighbors) for key, neighbors in adjacency.items()
            },
            "feature_dim": max(
                (len(group["features"]) for group in groups.values()),
                default=0,
            ),
        }

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
        include_primary_assignments = bool(
            self.problem_data.get("include_primary_assignments", False)
        )
        primary_assignments = {
            self.cacheKey(base_eclass_id): int(solver.Value(present))
            for base_eclass_id, (present, _) in self.cache_choices.items()
        }
        extractions = []
        
        for bucket_model in self.bucket_models:
            nodes = bucket_model["nodes"]
            bucket_idx = int(bucket_model["bucket"]["bucket_idx"])
            active_cids = {
                cid for cid, node in nodes.items() if solver.Value(node["active"])
            }
            
            selections = {}
            for cid in active_cids:
                node = nodes[cid]
                for present, enode in node["selections"].values():
                    if solver.Value(present):
                        selections[cid] = enode
                        break

            if include_primary_assignments:
                for cid, node in nodes.items():
                    selected_idx = None
                    for enode_idx, (present, _) in node["selections"].items():
                        if solver.Value(present):
                            selected_idx = int(enode_idx)
                            break
                    primary_assignments[self.selectionKey(bucket_idx, cid)] = selected_idx

                    inplace_assignment = None
                    for inplace_var, child_id, enode_idx in node["inplace_choices"]:
                        if solver.Value(inplace_var):
                            inplace_assignment = {
                                "enode_idx": int(enode_idx),
                                "child_id": int(child_id),
                            }
                            break
                    primary_assignments[
                        self.inplaceKey(bucket_idx, cid)
                    ] = inplace_assignment

            # Reconstruct topological sequence based exactly on solver timestamps
            adj = {cid: [] for cid in active_cids}
            in_degree = {cid: 0 for cid in active_cids}
            for cid in active_cids:
                for child_id in set(selections[cid].get("children", [])):
                    if child_id in active_cids:
                        adj[child_id].append(cid)
                        in_degree[cid] += 1
            
            queue = []
            for cid in active_cids:
                if in_degree[cid] == 0:
                    heapq.heappush(queue, (solver.Value(nodes[cid]["start"]), cid))
            
            order = []
            while queue:
                _, cid = heapq.heappop(queue)
                order.append(cid)
                for consumer in adj[cid]:
                    in_degree[consumer] -= 1
                    if in_degree[consumer] == 0:
                        heapq.heappush(queue, (solver.Value(nodes[consumer]["start"]), consumer))

            # Lazily unroll equivalent memory owners
            owners = {}
            for cid in order:
                node = nodes[cid]
                owner_val = None
                for present, src, is_node in node["sources"]:
                    if (type(present) is int and present) or (type(present) is not int and solver.Value(present)):
                        owner_val = owners[str(src)] if is_node else src
                        break
                if owner_val is None:
                    owner_val = node["id"]
                owners[str(cid)] = owner_val

            selection, costs, schedule = {}, {}, {}
            buffers = {buf["id"]: dict(buf) for buf in global_buffers}
            for position, cid in enumerate(order):
                node = nodes[cid]
                selected = selections[cid]
                selection[str(cid)] = selected["enode_idx"]
                costs[str(cid)] = float(selected["cost"])
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
                selected = selections[cid]
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
            
        result = {
            "solver": "ortools_full",
            "status": solver.StatusName(status),
            "cached_nodes": cached_nodes,
            "extractions": extractions,
            "objective": solver.ObjectiveValue(),
            "best_bound": solver.BestObjectiveBound(),
        }
        if include_primary_assignments:
            result["primary_assignments"] = primary_assignments
        return result

    def buildModel(self):
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
            self.model.Minimize(sum(self.objective_terms))
            self._addPrimaryFixings()
            self.model_built = True

        return self

    def solve(self):
        self.buildModel()

        error = self.model.Validate()
        if error:
            raise ValueError(f"Invalid OR-Tools full model: {error}")

        solver = cp_model.CpSolver()
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
            self.problem_data.get(
                "log_search_progress", self.problem_data.get("print_progress", True)
            )
        )
        solver.parameters.stop_after_first_solution = bool(
            self.problem_data.get("stop_after_first_solution", False)
        )
        # solver.parameters.linearization_level = 0
        solver.parameters.use_lns_only = True
        status = solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"OR-Tools full found no feasible joint plan: {solver.StatusName(status)}"
            )
        return self.decodeSolution(solver, status)


def solveOrtools(problem_data):
    """Solve one exported problem for the native subprocess bridge."""
    return OrtoolsSolver(problem_data).solve()


def main():
    if len(sys.argv) < 3:
        print("Usage: python ortools_full.py <problem.json> <solution.json>", file=sys.stderr)
        return 2

    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        problem_data = json.load(handle)
    solution = solveOrtools(problem_data)
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump(solution, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
