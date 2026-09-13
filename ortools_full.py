import math
import heapq
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
        self.hints = {}
        self.model_built = False
        self.full_bucket_idx = problem_data["full_bucket_idx"]

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
        """Safely record unique hints by variable index to prevent CP-SAT validation errors."""
        if var is not None and hasattr(var, "Index"):
            self.hints[var.Index()] = (var, int(val))

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
                    self.addHint(page_offset, 0)
                
                self.global_buffers.append(buf)
            self.cache_choices[base_eclass_id] = (present, buf)
            self.addHint(present, 0)
            
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

    def _extract_greedy_plan(self, bucket, classes):
        """Non-recursive Knuth Dijkstra to calculate optimal grounding and sequential plan."""
        root_id = bucket["root_eclass_id"]
        
        reserved = {}
        for cid, cls in classes.items():
            base_id = cls["base_eclass_id"]
            res = self.preallocated_by_base_id.get(base_id)
            if res is not None and not self.matchesBuffer(cls, res):
                res = None
            reserved[cid] = res

        parent_enodes = defaultdict(list)
        remaining_children = {}
        valid_enodes = {}

        for cid, cls in classes.items():
            has_res = reserved[cid] is not None
            if cls["mem_space"]["type"] != 0 and self.allocationSize(cls) > self.memoryCap(cls["mem_space"]):
                continue

            for enode in cls["enodes"]:
                e_idx = enode["enode_idx"]
                if not self.caching_enabled and (
                    enode.get("is_cache") or enode.get("is_scatter")
                ):
                    continue
                dur = self.duration(enode)
                if dur is None:
                    continue
                children = enode.get("children", [])
                if any(c not in classes for c in children):
                    continue
                if enode.get("mem_space", cls["mem_space"]) != cls["mem_space"]:
                    continue
                if enode.get("is_cache") or enode.get("is_scatter"):
                    continue
                if enode.get("is_view"):
                    if not children or classes[children[0]]["mem_space"] != cls["mem_space"]:
                        continue
                    if has_res:
                        continue
                if enode.get("is_input") and children:
                    continue

                key = (cid, e_idx)
                valid_enodes[key] = enode
                u_children = set(children)
                remaining_children[key] = len(u_children)
                for child in u_children:
                    parent_enodes[child].append(key)

        best_cost = {}
        best_selection = {}
        pq = []

        for (cid, e_idx), enode in valid_enodes.items():
            if remaining_children[(cid, e_idx)] == 0:
                cost = self.duration(enode) or 0
                if cid not in best_cost or cost < best_cost[cid]:
                    best_cost[cid] = cost
                    best_selection[cid] = e_idx
                    heapq.heappush(pq, (cost, cid))

        settled = set()
        while pq:
            curr_cost, cid = heapq.heappop(pq)
            if cid in settled:
                continue
            settled.add(cid)
            if cid == root_id:
                break

            for p_cid, p_e_idx in parent_enodes[cid]:
                if remaining_children[(p_cid, p_e_idx)] > 0:
                    remaining_children[(p_cid, p_e_idx)] -= 1
                    if remaining_children[(p_cid, p_e_idx)] == 0:
                        p_enode = valid_enodes[(p_cid, p_e_idx)]
                        p_dur = self.duration(p_enode) or 0
                        p_cost = p_dur + sum(
                            best_cost[c] for c in set(p_enode.get("children", []))
                        )
                        if p_cid not in best_cost or p_cost < best_cost[p_cid]:
                            best_cost[p_cid] = p_cost
                            best_selection[p_cid] = p_e_idx
                            heapq.heappush(pq, (p_cost, p_cid))

        if root_id not in best_selection:
            return None

        active_cids = set()
        queue = [root_id]
        while queue:
            curr = queue.pop()
            if curr in active_cids:
                continue
            active_cids.add(curr)
            e_idx = best_selection[curr]
            enode = valid_enodes[(curr, e_idx)]
            for child in set(enode.get("children", [])):
                if child in classes and child in best_selection:
                    queue.append(child)

        adj = {c: [] for c in active_cids}
        in_degree = {c: 0 for c in active_cids}
        for c in active_cids:
            e_idx = best_selection[c]
            enode = valid_enodes[(c, e_idx)]
            for child in set(enode.get("children", [])):
                if child in active_cids:
                    adj[child].append(c)
                    in_degree[c] += 1

        # LIFO (Depth-First) ready queue minimizes peak concurrent activation memory
        queue = [c for c in active_cids if in_degree[c] == 0]
        topo = []
        while queue:
            curr = queue.pop()
            topo.append(curr)
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        t = 0
        start_t, end_t = {}, {}
        for c in topo:
            start_t[c] = t
            dur = self.duration(valid_enodes[(c, best_selection[c])]) or 0
            end_t[c] = t + dur
            t = end_t[c]

        return {
            "root_id": root_id,
            "best_selection": best_selection,
            "active_cids": active_cids,
            "topo": topo,
            "adj": adj,
            "start_t": start_t,
            "end_t": end_t,
            "hint_makespan": t,
            "valid_enodes": valid_enodes,
            "reserved": reserved,
        }

    def _apply_hints(self, plan, nodes, horizon, makespan_var):
        root_id = plan["root_id"]
        active_cids = plan["active_cids"]
        best_selection = plan["best_selection"]
        topo = plan["topo"]
        adj = plan["adj"]
        start_t = plan["start_t"]
        end_t = plan["end_t"]
        valid_enodes = plan["valid_enodes"]
        reserved = plan["reserved"]

        # 1. Greedy In-place Identification
        topo_pos = {c: i for i, c in enumerate(topo)}
        reused_children = set()
        inplace_chosen = {}

        for c in topo:
            enode = valid_enodes[(c, best_selection[c])]
            has_res = reserved.get(c) is not None
            if has_res or enode.get("is_input") or enode.get("is_view"):
                continue

            safe_idxs = set(enode.get("safe_inplace_idxs", []))
            children = enode.get("children", [])
            for idx in sorted(safe_idxs):
                if not (0 <= idx < len(children)):
                    continue
                child_id = children[idx]
                if child_id not in active_cids or child_id in reused_children:
                    continue
                if reserved.get(child_id) is not None:
                    continue
                child_enode = valid_enodes[(child_id, best_selection[child_id])]
                if child_enode.get("is_input") or child_enode.get("is_view"):
                    continue
                if nodes[child_id]["mem_space"] != nodes[c]["mem_space"]:
                    continue
                if nodes[c]["cls"]["raw_size_bytes"] > nodes[child_id]["cls"]["raw_size_bytes"]:
                    continue

                consumers = adj[child_id]
                if not consumers:
                    continue
                last_consumer = max(consumers, key=lambda cons: topo_pos[cons])
                if last_consumer != c:
                    continue

                inplace_chosen[c] = child_id
                reused_children.add(child_id)
                break

        # 2. Synchronize Tensor Lifetimes
        read_end_t, release_t, protected_val = {}, {}, {}
        for c in topo:
            enode = valid_enodes[(c, best_selection[c])]
            has_res = reserved.get(c) is not None
            consumers = adj[c]

            if c in reused_children:
                cons_inplace = [cons for cons in consumers if inplace_chosen.get(cons) == c][0]
                other_cons = [cons for cons in consumers if cons != cons_inplace]
                read_end_t[c] = max([end_t[cons] for cons in other_cons] + [end_t[c]])
                read_end_t[c] = min(read_end_t[c], start_t[cons_inplace])
            else:
                read_end_t[c] = max([end_t[cons] for cons in consumers] + [end_t[c]])

            if c == root_id or enode.get("is_input") or has_res:
                release_t[c] = horizon
                protected_val[c] = 1
            else:
                release_t[c] = read_end_t[c]
                protected_val[c] = 0

        for c in reversed(topo):
            if c in inplace_chosen:
                child_id = inplace_chosen[c]
                release_t[child_id] = max(release_t[child_id], release_t[c])
            enode = valid_enodes[(c, best_selection[c])]
            if enode.get("is_view"):
                child = enode["children"][0]
                read_end_t[child] = max(read_end_t.get(child, 0), read_end_t[c])
                release_t[child] = max(release_t.get(child, 0), release_t[c])
                protected_val[c] = protected_val[child]
            if c == root_id:
                read_end_t[c] = horizon

        # 3. Fast 1D Allocator with In-Place Reuse
        preallocated_intervals = defaultdict(list)
        for res in self.preallocated:
            if res["mem_space"]["type"] != 0:
                ms = memSpaceKey(res["mem_space"])
                pages = self.alignedSize(res["size"]) // self.alignment
                start_page = res["offset"] // self.alignment
                preallocated_intervals[ms].append((start_page, start_page + pages))

        offsets = {}
        active_allocations = defaultdict(list)

        for c in topo:
            node = nodes[c]
            enode = valid_enodes[(c, best_selection[c])]
            has_res = reserved.get(c) is not None
            is_view = enode.get("is_view")

            if c in inplace_chosen or is_view:
                continue

            fresh = not has_res
            if fresh and node["page_offset"] is not None:
                ms = memSpaceKey(node["mem_space"])
                pages = self.alignedSize(node["size"]) // self.alignment
                if pages == 0:
                    offsets[c] = 0
                    continue

                cur_t = start_t[c]
                active_allocations[ms] = [
                    alloc for alloc in active_allocations[ms] if alloc[0] > cur_t
                ]

                live_intervals = [
                    (sp, ep) for (_, sp, ep) in active_allocations[ms]
                ] + preallocated_intervals[ms]
                live_intervals.sort()

                curr_offset = 0
                for sp, ep in live_intervals:
                    if curr_offset + pages <= sp:
                        break
                    curr_offset = max(curr_offset, ep)

                if curr_offset + pages <= self.memoryCapPages(node["mem_space"]):
                    offsets[c] = curr_offset
                    active_allocations[ms].append((release_t[c], curr_offset, curr_offset + pages))

        # 4. Inject Complete Hints
        self.addHint(makespan_var, max(end_t.values()) if end_t else 0)

        for cid, node in nodes.items():
            if cid in active_cids:
                self.addHint(node["active"], 1)
                self.addHint(node["start"], start_t[cid])
                self.addHint(node["end"], end_t[cid])
                self.addHint(node["read_end"], read_end_t[cid])
                self.addHint(node["release"], release_t[cid])
                self.addHint(node["protected"], protected_val[cid])
                if "lifetime" in node:
                    self.addHint(node["lifetime"], release_t[cid] - start_t[cid])

                enode = valid_enodes[(cid, best_selection[cid])]
                has_res = reserved.get(cid) is not None
                is_view = enode.get("is_view")
                is_inplace = (cid in inplace_chosen)
                fresh = 1 if (not has_res and not is_view and not is_inplace) else 0
                self.addHint(node["fresh"], fresh)

                if node["page_offset"] is not None and type(node["page_offset"]) is not int:
                    if fresh:
                        if cid in offsets:
                            self.addHint(node["page_offset"], offsets[cid])
                        else:
                            print(f"[Warning] Fresh tensor {cid} in {node['mem_space']} exceeded cap!")
                    else:
                        self.addHint(node["page_offset"], 0)


                for e_idx, (sel_var, _) in node["selections"].items():
                    self.addHint(sel_var, 1 if e_idx == best_selection[cid] else 0)

                for inplace_var, child_id in node["inplace_choices"]:
                    is_in = (cid in inplace_chosen and inplace_chosen[cid] == child_id)
                    self.addHint(inplace_var, 1 if is_in else 0)
            else:
                self.addHint(node["active"], 0)
                self.addHint(node["fresh"], 0)
                self.addHint(node["protected"], 0)
                self.addHint(node["start"], 0)
                self.addHint(node["end"], 0)
                self.addHint(node["read_end"], 0)
                self.addHint(node["release"], 0)
                if "lifetime" in node:
                    self.addHint(node["lifetime"], 0)

                self.addHint(node.get("page_offset"), 0)
                for sel_var, _ in node["selections"].values():
                    self.addHint(sel_var, 0)

                for inplace_var, _ in node["inplace_choices"]:
                    self.addHint(inplace_var, 0)

            if type(node["cached"]) is not int:
                self.addHint(node["cached"], 0)

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

        # 2. Extract Greedy Plan for Tight Horizon
        plan = self._extract_greedy_plan(bucket, classes)

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

        if plan is not None:
            desired_horizon = max(count + 1, int(plan["hint_makespan"] * 2))
        else:
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
                        if child["mem_space"] != cls["mem_space"] or cls[
                            "raw_size_bytes"
                        ] > child["cls"]["raw_size_bytes"]:
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
                        node["inplace_choices"].append((inplace, child_id))
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
                
        for node in nodes.values():
            for present, child_id in node["inplace_choices"]:
                for selected, enode in nodes[child_id]["selections"].values():
                    if enode.get("is_view"):
                        model.AddAtMostOne([present, selected])

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

        weight = float(bucket.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid bucket weight {weight}")
        self.objective_terms.append(weight * makespan)
        
        if plan is not None:
            self._apply_hints(plan, nodes, horizon, makespan)
        
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
            active_cids = set(cid for cid, node in nodes.items() if solver.Value(node["active"]))
            
            selections = {}
            for cid in active_cids:
                node = nodes[cid]
                for present, enode in node["selections"].values():
                    if solver.Value(present):
                        selections[cid] = enode
                        break

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
            self.model.Minimize(sum(self.objective_terms))
            
            # Commit unique, deduplicated hints to CP-SAT
            for var, val in self.hints.values():
                self.model.AddHint(var, val)
                
            self.model_built = True

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
            self.problem_data.get("print_progress", True)
        )
        # solver.parameters.linearization_level = 0
        # solver.parameters.stop_after_first_solution = True
        status = solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"OR-Tools full found no feasible joint plan: {solver.StatusName(status)}"
            )
        return self.decodeSolution(solver, status)