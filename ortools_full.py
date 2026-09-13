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
                
                # Zero-bind unused spatial variables globally
                if page_offset is not None and type(page_offset) is not int:
                    self.model.Add(page_offset == 0).OnlyEnforceIf(present.Not())
                
                self.global_buffers.append(buf)
            self.cache_choices[base_eclass_id] = (present, buf)
            
            # Global hint to suppress cache utilization initially
            self.model.AddHint(present, 0)
            if buf.get("page_offset") is not None and type(buf["page_offset"]) is not int:
                self.model.AddHint(buf["page_offset"], 0)
            
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
            (time_interval, space_interval)
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

    def _compute_hints(self, bucket, nodes, horizon, makespan_var):
        """Topological DP greedy hint generation providing a 100% complete model initialization."""
        root_id = bucket["root_eclass_id"]
        memo = {}
        visiting = set()
        
        def get_best(cid):
            if cid in memo: return memo[cid]
            if cid in visiting: return (float('inf'), {})
            visiting.add(cid)
            
            node = nodes[cid]
            cls = node["cls"]
            has_res = self.preallocated_by_base_id.get(cls["base_eclass_id"]) is not None
            best_cost = float('inf')
            best_e = None
            best_deps = {}
            for e_idx, (sel_var, enode) in node["selections"].items():
                dur = self.duration(enode)
                if dur is None: continue
                children = enode.get("children", [])
                if any(c not in nodes for c in children): continue
                if enode.get("mem_space", cls["mem_space"]) != cls["mem_space"]: continue
                if enode.get("is_cache") or enode.get("is_scatter"): continue
                if enode.get("is_view"):
                    if not children or nodes[children[0]]["mem_space"] != cls["mem_space"]: continue
                    if has_res: continue
                if enode.get("is_input") and children: continue
                
                c_cost = 0
                valid = True
                deps = {}
                for c in set(children):
                    cc, cdeps = get_best(c)
                    if cc == float('inf'):
                        valid = False
                        break
                    c_cost += cc
                    deps.update(cdeps)
                
                if not valid: continue
                t_cost = dur + c_cost
                if t_cost < best_cost:
                    best_cost = t_cost
                    best_e = e_idx
                    best_deps = deps
                    
            if best_e is not None:
                res = {cid: best_e}
                res.update(best_deps)
                memo[cid] = (best_cost, res)
            else:
                memo[cid] = (float('inf'), {})
                
            visiting.remove(cid)
            return memo[cid]
            
        cost, selections = get_best(root_id)
        if cost == float('inf'): return
        
        active_cids = set(selections.keys())
        
        # Topological Sort Active Subgraph
        adj = {c: [] for c in active_cids}
        in_degree = {c: 0 for c in active_cids}
        for c in active_cids:
            enode = nodes[c]["selections"][selections[c]][1]
            for child in set(enode.get("children", [])):
                if child in active_cids:
                    adj[child].append(c)
                    in_degree[c] += 1
                    
        queue = [c for c in active_cids if in_degree[c] == 0]
        topo = []
        while queue:
            curr = queue.pop(0)
            topo.append(curr)
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
                    
        # Simulate execution times natively as a purely sequential schedule
        t = 0
        start_t, end_t = {}, {}
        for c in topo:
            start_t[c] = t
            dur = self.duration(nodes[c]["selections"][selections[c]][1])
            end_t[c] = t + (dur if dur else 0)
            t = end_t[c]
            
        # Simulate tensor lifetimes
        read_end_t, release_t, protected_val = {}, {}, {}
        for c in topo:
            enode = nodes[c]["selections"][selections[c]][1]
            has_res = self.preallocated_by_base_id.get(nodes[c]["cls"]["base_eclass_id"]) is not None
            consumers = adj[c]
            
            read_end_t[c] = max([end_t[cons] for cons in consumers] + [end_t[c]])
            
            if c == root_id or enode.get("is_input") or has_res:
                release_t[c] = horizon
                protected_val[c] = 1
            else:
                release_t[c] = read_end_t[c]
                protected_val[c] = 0

        # View Lifetimes propagate to underlying parent
        for c in reversed(topo):
            enode = nodes[c]["selections"][selections[c]][1]
            if enode.get("is_view"):
                child = enode["children"][0]
                read_end_t[child] = max(read_end_t.get(child, 0), read_end_t[c])
                release_t[child] = max(release_t.get(child, 0), release_t[c])
                protected_val[c] = protected_val[child]
            if c == root_id:
                read_end_t[c] = horizon

        # Bump Allocate Safe 2D Offsets (Simple Greedy 1D Interval Allocator)
        offsets = {}
        for c in topo:
            node = nodes[c]
            enode = node["selections"][selections[c]][1]
            has_res = self.preallocated_by_base_id.get(node["cls"]["base_eclass_id"]) is not None
            is_view = enode.get("is_view")
            fresh = not has_res and not is_view
            
            if fresh and node["page_offset"] is not None:
                ms = memSpaceKey(node["mem_space"])
                pages = self.alignedSize(node["size"]) // self.alignment
                if pages == 0:
                    offsets[c] = 0
                    continue
                
                live_intervals = []
                for past_c in topo:
                    if past_c == c: break
                    if past_c in offsets and release_t[past_c] > start_t[c]:
                        past_ms = memSpaceKey(nodes[past_c]["mem_space"])
                        if past_ms == ms:
                            past_pages = self.alignedSize(nodes[past_c]["size"]) // self.alignment
                            live_intervals.append((offsets[past_c], offsets[past_c] + past_pages))
                            
                for res in self.preallocated:
                    if res["mem_space"]["type"] != 0 and memSpaceKey(res["mem_space"]) == ms:
                        p_start = res["offset"] // self.alignment
                        p_pages = self.alignedSize(res["size"]) // self.alignment
                        live_intervals.append((p_start, p_start + p_pages))
                        
                live_intervals.sort()
                
                curr_offset = 0
                assigned = False
                for start_p, end_p in live_intervals:
                    if curr_offset + pages <= start_p:
                        offsets[c] = curr_offset
                        assigned = True
                        break
                    curr_offset = max(curr_offset, end_p)
                    
                if not assigned:
                    if curr_offset + pages <= self.memoryCapPages(node["mem_space"]):
                        offsets[c] = curr_offset
                    else:
                        offsets[c] = 0

        # Inject Hints natively to CP-SAT
        self.model.AddHint(makespan_var, max(end_t.values()) if end_t else 0)

        for cid, node in nodes.items():
            if cid in active_cids:
                self.model.AddHint(node["active"], 1)
                self.model.AddHint(node["start"], start_t[cid])
                self.model.AddHint(node["end"], end_t[cid])
                self.model.AddHint(node["read_end"], read_end_t[cid])
                self.model.AddHint(node["release"], release_t[cid])
                self.model.AddHint(node["protected"], protected_val[cid])
                if "lifetime" in node:
                    self.model.AddHint(node["lifetime"], release_t[cid] - start_t[cid])
                
                enode = node["selections"][selections[cid]][1]
                has_res = self.preallocated_by_base_id.get(node["cls"]["base_eclass_id"]) is not None
                fresh = 1 if (not has_res and not enode.get("is_view")) else 0
                self.model.AddHint(node["fresh"], fresh)
                
                if node["page_offset"] is not None and type(node["page_offset"]) is not int:
                    self.model.AddHint(node["page_offset"], offsets.get(cid, 0))
                    
                for e_idx, (sel_var, _) in node["selections"].items():
                    self.model.AddHint(sel_var, 1 if e_idx == selections[cid] else 0)
            else:
                self.model.AddHint(node["active"], 0)
                self.model.AddHint(node["fresh"], 0)
                self.model.AddHint(node["protected"], 0)
                self.model.AddHint(node["start"], 0)
                self.model.AddHint(node["end"], 0)
                self.model.AddHint(node["read_end"], 0)
                self.model.AddHint(node["release"], 0)
                if "lifetime" in node:
                    self.model.AddHint(node["lifetime"], 0)
                    
                if node["page_offset"] is not None and type(node["page_offset"]) is not int:
                    self.model.AddHint(node["page_offset"], 0)
                for sel_var, _ in node["selections"].values():
                    self.model.AddHint(sel_var, 0)
                    
            for inplace_var, _ in node["inplace_choices"]:
                self.model.AddHint(inplace_var, 0)
            if type(node["cached"]) is not int:
                self.model.AddHint(node["cached"], 0)


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

        for items in rectangles.values():
            model.AddNoOverlap2D(
                [item[0] for item in items], [item[1] for item in items]
            )

        weight = float(bucket.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid bucket weight {weight}")
        self.objective_terms.append(weight * makespan)
        
        self._compute_hints(bucket, nodes, horizon, makespan)
        
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
        status = solver.Solve(self.model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"OR-Tools full found no feasible joint plan: {solver.StatusName(status)}"
            )
        return self.decodeSolution(solver, status)