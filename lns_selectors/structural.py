"""Training-free structural neighborhood selection for exact LNS."""

import math
import random
from collections import defaultdict, deque

from .protocol import NeighborhoodSelector


class StructuralNeighborhoodSelector(NeighborhoodSelector):
    """Select a critical, structurally closed set of primary decision groups.

    The selector deliberately only returns primary group keys.  Timing,
    reachability, allocation, and cache constraints remain owned by the exact
    OR-Tools model during the repair solve.
    """

    def __init__(self, target_size=15, max_neighborhood_size=64, seed=None):
        self.max_neighborhood_size = max(1, int(max_neighborhood_size))
        self.target_size = min(max(1, int(target_size)), self.max_neighborhood_size)
        self.random = random.Random(seed)
        self._bucket_classes = {}
        self._bucket_extractions = {}
        self._bucket_specs = {}
        self._context = {}

    @staticmethod
    def _asInt(value, default=None):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    @staticmethod
    def _lookup(mapping, key, default=None):
        if not isinstance(mapping, dict):
            return default
        if key in mapping:
            return mapping[key]
        string_key = str(key)
        if string_key in mapping:
            return mapping[string_key]
        integer_key = StructuralNeighborhoodSelector._asInt(key)
        if integer_key is not None and integer_key in mapping:
            return mapping[integer_key]
        return default

    @staticmethod
    def _selectionKey(bucket_idx, eclass_id):
        return f"selection:{int(bucket_idx)}:{int(eclass_id)}"

    @staticmethod
    def _inplaceKey(bucket_idx, eclass_id):
        return f"inplace:{int(bucket_idx)}:{int(eclass_id)}"

    @staticmethod
    def _cacheKey(base_eclass_id):
        return f"cache:{int(base_eclass_id)}"

    @staticmethod
    def _parseSelectionKey(key):
        if not isinstance(key, str):
            return None
        parts = key.split(":")
        if len(parts) != 3 or parts[0] != "selection":
            return None
        bucket_idx = StructuralNeighborhoodSelector._asInt(parts[1])
        eclass_id = StructuralNeighborhoodSelector._asInt(parts[2])
        if bucket_idx is None or eclass_id is None:
            return None
        return bucket_idx, eclass_id

    def _prepareContext(self, context):
        self._context = context or {}
        incumbent = self._context.get("incumbent", {}) or {}
        extractions = incumbent.get("extractions", []) or []
        model = self._context.get("model")
        problem_data = (
            model.get("problem_data", model)
            if isinstance(model, dict)
            else getattr(model, "problem_data", None)
        )
        if not isinstance(problem_data, dict):
            problem_data = self._context.get("problem_data", {}) or {}

        buckets = problem_data.get("buckets", [])
        if not buckets and isinstance(model, dict):
            buckets = model.get("buckets", []) or []
        if not buckets and model is not None and not isinstance(model, dict):
            buckets = getattr(model, "buckets", []) or []
        if not buckets and getattr(model, "bucket_models", None):
            buckets = [item.get("bucket", {}) for item in model.bucket_models]

        self._bucket_classes = {}
        self._bucket_extractions = {}
        self._bucket_specs = {}
        for position, bucket in enumerate(buckets):
            if not isinstance(bucket, dict):
                continue
            bucket_idx = self._asInt(bucket.get("bucket_idx"), position)
            classes = {
                self._asInt(cls.get("id"), cls.get("id")): cls
                for cls in bucket.get("classes", [])
                if isinstance(cls, dict) and cls.get("id") is not None
            }
            self._bucket_classes[bucket_idx] = classes
            self._bucket_specs[bucket_idx] = bucket
            extraction = extractions[position] if position < len(extractions) else {}
            if not isinstance(extraction, dict):
                extraction = {}
            self._bucket_extractions[bucket_idx] = extraction

        # A few callers construct a light-weight model with bucket_models but
        # no problem_data.  Preserve enough raw class information for tests and
        # custom integrations in that case.
        if not self._bucket_classes and model is not None:
            for item in getattr(model, "bucket_models", []) or []:
                bucket = item.get("bucket", {})
                bucket_idx = self._asInt(bucket.get("bucket_idx"), 0)
                nodes = item.get("nodes", {})
                self._bucket_classes[bucket_idx] = {
                    self._asInt(cid, cid): node.get("cls", {})
                    for cid, node in nodes.items()
                    if isinstance(node, dict) and isinstance(node.get("cls"), dict)
                }

    def _classesForBucket(self, bucket_idx):
        return self._bucket_classes.get(int(bucket_idx), {})

    def _extractionForBucket(self, bucket_idx):
        return self._bucket_extractions.get(int(bucket_idx), {})

    def _group(self, metadata, key):
        return (metadata.get("groups", {}) or {}).get(key, {})

    def _activeIds(self, bucket_idx, extraction):
        selection_map = extraction.get("selection_map", {}) or {}
        return {
            self._asInt(cid, cid)
            for cid in selection_map
            if self._asInt(cid, cid) is not None
        }

    def _selectedEnode(self, bucket_idx, eclass_id, extraction=None):
        extraction = extraction or self._extractionForBucket(bucket_idx)
        selection_map = extraction.get("selection_map", {}) or {}
        selected_idx = self._lookup(selection_map, eclass_id)
        selected_idx = self._asInt(selected_idx)
        cls = self._classesForBucket(bucket_idx).get(eclass_id, {})
        for enode in cls.get("enodes", []) if isinstance(cls, dict) else []:
            if self._asInt(enode.get("enode_idx")) == selected_idx:
                return enode
        if selected_idx is not None and isinstance(cls, dict):
            enodes = cls.get("enodes", [])
            if 0 <= selected_idx < len(enodes):
                return enodes[selected_idx]
        return None

    def _selectedCost(self, bucket_idx, eclass_id, extraction, schedule_entry=None):
        costs = extraction.get("eclass_to_cost", {}) or {}
        cost = self._lookup(costs, eclass_id)
        try:
            if cost is not None and math.isfinite(float(cost)):
                return max(0.0, float(cost))
        except (TypeError, ValueError, OverflowError):
            pass
        enode = self._selectedEnode(bucket_idx, eclass_id, extraction)
        try:
            if enode is not None and math.isfinite(float(enode.get("cost", 0.0))):
                return max(0.0, float(enode.get("cost", 0.0)))
        except (TypeError, ValueError, OverflowError):
            pass
        if schedule_entry:
            try:
                return max(
                    0.0,
                    float(schedule_entry.get("end", 0))
                    - float(schedule_entry.get("start", 0)),
                )
            except (TypeError, ValueError, OverflowError):
                pass
        return 0.0

    @staticmethod
    def _engineValues(enode, schedule_entry, cls=None):
        values = []
        if isinstance(schedule_entry, dict):
            for field in ("engine", "engine_key", "engine_idx"):
                if field in schedule_entry:
                    values.append(str(schedule_entry[field]))
            scheduled_engines = schedule_entry.get("engines")
            if scheduled_engines:
                values.extend(str(item) for item in scheduled_engines)
        if isinstance(enode, dict):
            engines = enode.get("engines") or []
            if not isinstance(engines, (list, tuple, set)):
                engines = [engines]
            for engine in engines:
                if isinstance(engine, dict):
                    values.append(
                        f"{engine.get('type', '')}:{engine.get('idx', '')}"
                    )
                else:
                    values.append(str(engine))
        if not values and isinstance(cls, dict):
            mem_space = cls.get("mem_space")
            if isinstance(mem_space, dict):
                values.append(
                    f"{mem_space.get('type', '')}:{mem_space.get('idx', '')}"
                )
        return set(values or ["default"])

    def _taskRecords(self):
        records = {}
        for bucket_idx, extraction in self._bucket_extractions.items():
            schedule = extraction.get("schedule", {}) or {}
            for raw_cid, entry in schedule.items():
                cid = self._asInt(raw_cid, raw_cid)
                if not isinstance(entry, dict):
                    continue
                enode = self._selectedEnode(bucket_idx, cid, extraction)
                cls = self._classesForBucket(bucket_idx).get(cid, {})
                records[(bucket_idx, cid)] = {
                    "bucket_idx": bucket_idx,
                    "eclass_id": cid,
                    "schedule": entry,
                    "start": self._asInt(entry.get("start"), 0),
                    "end": self._asInt(entry.get("end"), 0),
                    "enode": enode or {},
                    "engines": self._engineValues(enode, entry, cls),
                }
        return records

    def _criticalGraphSlack(self, records):
        """Approximate total float from selected data and engine successors."""
        successors = defaultdict(set)
        for record_key, record in records.items():
            bucket_idx = record["bucket_idx"]
            for child_id in record["enode"].get("children", []):
                child_key = (bucket_idx, self._asInt(child_id, child_id))
                if child_key in records:
                    successors[child_key].add(record_key)

        by_engine = defaultdict(list)
        for record_key, record in records.items():
            for engine in record["engines"]:
                by_engine[(record["bucket_idx"], engine)].append(record_key)
        for tasks in by_engine.values():
            tasks.sort(key=lambda key: (records[key]["start"], records[key]["end"], key[1]))
            for previous, current in zip(tasks, tasks[1:]):
                if records[previous]["end"] <= records[current]["start"]:
                    successors[previous].add(current)

        horizon = max((record["end"] for record in records.values()), default=0)
        latest_start = {key: horizon for key in records}
        for key in sorted(
            records, key=lambda item: (records[item]["start"], records[item]["end"]), reverse=True
        ):
            if successors.get(key):
                latest_start[key] = min(latest_start[item] for item in successors[key])
        return {
            key: max(0.0, float(latest_start[key] - records[key]["start"]))
            for key in records
        }

    def _find_critical_path_seeds(self, incumbent, metadata):
        """Return selectable keys on the current critical or near-critical path."""
        records = self._taskRecords()
        if not records:
            return []

        latest_key = max(
            records,
            key=lambda key: (records[key]["end"], records[key]["start"], key[0], key[1]),
        )
        engine_predecessors = defaultdict(set)
        for record_key, record in records.items():
            for engine in record["engines"]:
                engine_predecessors[
                    (record["bucket_idx"], engine, record["end"])
                ].add(record_key)
        stack = [latest_key]
        visited = set()
        critical = []
        while stack:
            current_key = stack.pop()
            if current_key in visited:
                continue
            visited.add(current_key)
            current = records[current_key]
            current_start = current["start"]
            bucket_idx = current["bucket_idx"]
            selection_key = self._selectionKey(bucket_idx, current["eclass_id"])
            group = self._group(metadata, selection_key)
            if int(group.get("choice_count", 0)) > 1 and group.get("selectable", True):
                critical.append(selection_key)

            for child_id in current["enode"].get("children", []):
                child_key = (bucket_idx, self._asInt(child_id, child_id))
                child = records.get(child_key)
                if child is not None and child["end"] == current_start:
                    stack.append(child_key)

            for engine in current["engines"]:
                for candidate_key in engine_predecessors.get(
                    (bucket_idx, engine, current_start), ()
                ):
                    if candidate_key != current_key:
                        stack.append(candidate_key)

        if critical:
            return sorted(
                set(critical), key=lambda key: self._seedImpact(key, records, metadata), reverse=True
            )

        # No selectable zero-slack node: use the smallest total float among
        # active selectable groups, which is the useful near-critical fallback.
        slack = self._criticalGraphSlack(records)
        candidates = []
        for record_key, record in records.items():
            key = self._selectionKey(record["bucket_idx"], record["eclass_id"])
            group = self._group(metadata, key)
            if int(group.get("choice_count", 0)) > 1 and group.get("selectable", True):
                candidates.append((slack.get(record_key, float("inf")), key))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [key for _, key in candidates]

    def _seedImpact(self, key, records, metadata):
        parsed = self._parseSelectionKey(key)
        if parsed is None:
            return (0.0, 0, key)
        bucket_idx, eclass_id = parsed
        record = records.get((bucket_idx, eclass_id), {})
        group = self._group(metadata, key)
        duration = self._selectedCost(
            bucket_idx,
            eclass_id,
            self._extractionForBucket(bucket_idx),
            record.get("schedule"),
        )
        choices = int(group.get("choice_count", 0))
        return (duration * choices, duration, choices, key)

    def _pickHighestImpactSeed(self, seeds, metadata):
        if not seeds:
            return None
        records = self._taskRecords()
        ranked = sorted(
            seeds, key=lambda key: self._seedImpact(key, records, metadata), reverse=True
        )
        best_impact = self._seedImpact(ranked[0], records, metadata)
        ties = [
            key
            for key in ranked
            if self._seedImpact(key, records, metadata) == best_impact
        ]
        return self.random.choice(ties)

    def _pick_highest_impact_seed(self, seeds, metadata):
        """Compatibility spelling for the engineering-plan helper name."""
        return self._pickHighestImpactSeed(seeds, metadata)

    def _fastestEnode(self, cls):
        enodes = list(cls.get("enodes", [])) if isinstance(cls, dict) else []
        if not enodes:
            return None

        def rank(enode):
            try:
                cost = float(enode.get("cost", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                cost = float("inf")
            return (cost, len(enode.get("children", [])), self._asInt(enode.get("enode_idx"), 0))

        return min(enodes, key=rank)

    def _addAlternateSupport(self, bucket_idx, eclass_id, nodes, metadata, budget, seen):
        key = self._selectionKey(bucket_idx, eclass_id)
        groups = metadata.get("groups", {}) or {}
        if key not in groups or key in seen:
            return
        if len(nodes) >= budget:
            return
        nodes.add(key)
        seen.add(key)
        cls = self._classesForBucket(bucket_idx).get(eclass_id, {})
        if not cls:
            return
        extraction = self._extractionForBucket(bucket_idx)
        active_ids = self._activeIds(bucket_idx, extraction)
        fastest = self._fastestEnode(cls)
        if fastest is None:
            return
        # An input/constant is a valid grounding leaf.  Its selection key is
        # still included when it is represented as an inactive decision group.
        if fastest.get("is_input") or fastest.get("is_constant"):
            return
        for child_id in fastest.get("children", []):
            child_id = self._asInt(child_id, child_id)
            if child_id in active_ids:
                continue
            if len(nodes) >= budget:
                return
            self._addAlternateSupport(
                bucket_idx, child_id, nodes, metadata, budget, seen
            )

    def _expand_alternate_support_nodes(
        self, nodes, incumbent, metadata, budget
    ):
        """Ground inactive children of alternative enodes before closure."""
        del incumbent  # The prepared extraction is the incumbent snapshot.
        result = set()
        seen = set(nodes)
        remaining = max(0, int(budget) - len(nodes))
        if not remaining:
            return result
        for key in list(nodes):
            parsed = self._parseSelectionKey(key)
            if parsed is None:
                continue
            bucket_idx, eclass_id = parsed
            extraction = self._extractionForBucket(bucket_idx)
            selected_idx = self._asInt(
                self._lookup(extraction.get("selection_map", {}), eclass_id)
            )
            cls = self._classesForBucket(bucket_idx).get(eclass_id, {})
            for enode in cls.get("enodes", []) if isinstance(cls, dict) else []:
                if self._asInt(enode.get("enode_idx")) == selected_idx:
                    continue
                for child_id in enode.get("children", []):
                    child_id = self._asInt(child_id, child_id)
                    if child_id in self._activeIds(bucket_idx, extraction):
                        continue
                    before = set(result)
                    self._addAlternateSupport(
                        bucket_idx, child_id, result, metadata, remaining, seen
                    )
                    seen.update(result)
                    if result != before and len(result) >= remaining:
                        return result
        return result

    def _activeConsumers(self):
        consumers = defaultdict(set)
        for bucket_idx, extraction in self._bucket_extractions.items():
            active_ids = self._activeIds(bucket_idx, extraction)
            for eclass_id in active_ids:
                enode = self._selectedEnode(bucket_idx, eclass_id, extraction)
                if not enode:
                    continue
                parent_key = self._selectionKey(bucket_idx, eclass_id)
                for child_id in enode.get("children", []):
                    child_id = self._asInt(child_id, child_id)
                    if child_id in active_ids:
                        consumers[(bucket_idx, child_id)].add(parent_key)
        return consumers

    def _alternateConsumers(self, nodes):
        consumers = defaultdict(set)
        for key in nodes:
            parsed = self._parseSelectionKey(key)
            if parsed is None:
                continue
            bucket_idx, eclass_id = parsed
            cls = self._classesForBucket(bucket_idx).get(eclass_id, {})
            extraction = self._extractionForBucket(bucket_idx)
            selected_idx = self._asInt(
                self._lookup(extraction.get("selection_map", {}), eclass_id)
            )
            for enode in cls.get("enodes", []) if isinstance(cls, dict) else []:
                if self._asInt(enode.get("enode_idx")) == selected_idx:
                    continue
                for child_id in enode.get("children", []):
                    child_id = self._asInt(child_id, child_id)
                    consumers[(bucket_idx, child_id)].add(key)
        return consumers

    def _expand_incumbent_support_nodes(
        self, nodes, incumbent, metadata, budget
    ):
        """Close the incumbent support cone at the unfrozen-consumer cut."""
        del incumbent
        nodes = set(nodes)
        active_consumers = self._activeConsumers()
        alternate_consumers = self._alternateConsumers(nodes)
        queue = deque(key for key in nodes if self._parseSelectionKey(key))
        while queue and len(nodes) < budget:
            parent_key = queue.popleft()
            parsed = self._parseSelectionKey(parent_key)
            if parsed is None:
                continue
            bucket_idx, eclass_id = parsed
            extraction = self._extractionForBucket(bucket_idx)
            if eclass_id not in self._activeIds(bucket_idx, extraction):
                continue
            enode = self._selectedEnode(bucket_idx, eclass_id, extraction)
            if not enode:
                continue
            for child_id in enode.get("children", []):
                child_id = self._asInt(child_id, child_id)
                child_key = self._selectionKey(bucket_idx, child_id)
                if child_key in nodes or child_key not in (metadata.get("groups", {}) or {}):
                    continue
                child_ref = (bucket_idx, child_id)
                consumers = set(active_consumers.get(child_ref, ()))
                consumers.update(alternate_consumers.get(child_ref, ()))
                if consumers and consumers.issubset(nodes) and len(nodes) < budget:
                    nodes.add(child_key)
                    for support_key, support_consumers in self._alternateConsumers(
                        {child_key}
                    ).items():
                        alternate_consumers[support_key].update(support_consumers)
                    queue.append(child_key)
        return nodes

    def _problemData(self):
        model = self._context.get("model")
        problem_data = (
            model.get("problem_data", model)
            if isinstance(model, dict)
            else getattr(model, "problem_data", None)
        )
        if isinstance(problem_data, dict):
            return problem_data
        return self._context.get("problem_data", {}) or {}

    def _memoryIsTight(self, bucket_idx, eclass_id):
        extraction = self._extractionForBucket(bucket_idx)
        schedule = self._lookup(extraction.get("schedule", {}), eclass_id, {}) or {}
        cls = self._classesForBucket(bucket_idx).get(eclass_id, {})
        mem_space = cls.get("mem_space", {}) if isinstance(cls, dict) else {}
        data = self._problemData()
        caps = data.get("mem_caps", {}) or {}
        cap_key = f"{mem_space.get('type', '')}:{mem_space.get('idx', '')}"
        cap = caps.get(cap_key, caps.get(str(mem_space.get("type", ""))))
        if cap is None:
            cap = self._bucket_specs.get(bucket_idx, {}).get("mem_cap")
        try:
            cap = float(cap)
        except (TypeError, ValueError):
            return False
        if cap <= 0:
            return False

        try:
            window_start = float(schedule.get("start", 0))
            window_end = float(schedule.get("end", window_start))
        except (TypeError, ValueError):
            return False
        peak = 0.0
        buffers = extraction.get("buffers", []) or []
        overlapping = []
        for buffer in buffers:
            if isinstance(buffer, dict):
                buffer_start = buffer.get("start", 0)
                buffer_end = buffer.get("end", buffer_start)
                size = buffer.get("size", 0)
                buffer_space = buffer.get("mem_space", mem_space)
            elif isinstance(buffer, (list, tuple)) and len(buffer) >= 4:
                buffer_start, buffer_end, _, size = buffer[:4]
                buffer_space = mem_space
            else:
                continue
            if isinstance(buffer_space, dict) and buffer_space != mem_space:
                continue
            try:
                if float(buffer_start) <= window_end and float(buffer_end) >= window_start:
                    overlapping.append(float(size))
            except (TypeError, ValueError):
                continue
        if overlapping:
            peak = sum(overlapping)
        else:
            # Native extraction buffer lifetimes are topological positions,
            # while schedule timestamps are engine time.  If units differ,
            # use the conservative bucket peak for this coupling decision.
            for buffer in buffers:
                if isinstance(buffer, dict):
                    if buffer.get("mem_space", mem_space) != mem_space:
                        continue
                    size = buffer.get("size", 0)
                elif isinstance(buffer, (list, tuple)) and len(buffer) >= 4:
                    size = buffer[3]
                else:
                    continue
                try:
                    peak += float(size)
                except (TypeError, ValueError):
                    pass
        return peak >= 0.9 * cap

    def _get_coupled_inplace_keys(self, selection_keys, metadata):
        groups = metadata.get("groups", {}) or {}
        result = set()
        for key in selection_keys:
            parsed = self._parseSelectionKey(key)
            if parsed is None:
                continue
            bucket_idx, eclass_id = parsed
            inplace_key = self._inplaceKey(bucket_idx, eclass_id)
            if inplace_key in groups:
                result.add(inplace_key)

            if not self._memoryIsTight(bucket_idx, eclass_id):
                continue
            extraction = self._extractionForBucket(bucket_idx)
            enode = self._selectedEnode(bucket_idx, eclass_id, extraction)
            if not enode:
                continue
            for child_id in enode.get("children", []):
                child_id = self._asInt(child_id, child_id)
                child_selection = self._selectionKey(bucket_idx, child_id)
                child_inplace = self._inplaceKey(bucket_idx, child_id)
                if child_selection in groups and child_inplace in groups:
                    result.update((child_selection, child_inplace))
        return result

    def _consumerSelectionKeys(self, bucket_idx, child_ids, groups):
        result = set()
        classes = self._classesForBucket(bucket_idx)
        wanted = set(child_ids)
        for parent_id, cls in classes.items():
            if any(
                self._asInt(child_id, child_id) in wanted
                for enode in cls.get("enodes", [])
                for child_id in enode.get("children", [])
            ):
                key = self._selectionKey(bucket_idx, parent_id)
                if key in groups:
                    result.add(key)
        return result

    def _get_coupled_cache_clusters(self, selection_keys, metadata):
        groups = metadata.get("groups", {}) or {}
        result = set()
        base_ids = set()
        for key in selection_keys:
            group = groups.get(key, {})
            if group.get("kind") == "selection":
                base_id = self._asInt(group.get("base_eclass_id"))
                if base_id is not None:
                    base_ids.add(base_id)

        problem_data = self._problemData()
        full_bucket_idx = self._asInt(problem_data.get("full_bucket_idx"), 0)
        for base_id in base_ids:
            cache_key = self._cacheKey(base_id)
            if cache_key not in groups:
                continue
            result.add(cache_key)
            for bucket_idx, classes in self._bucket_classes.items():
                matching = [
                    cid
                    for cid, cls in classes.items()
                    if self._asInt(cls.get("base_eclass_id"), cid) == base_id
                ]
                for cid in matching:
                    selection_key = self._selectionKey(bucket_idx, cid)
                    if selection_key not in groups:
                        continue
                    # The producer in the full bucket and all readers in
                    # partial/decode buckets are part of one cache repair.
                    result.add(selection_key)
                    if bucket_idx != full_bucket_idx:
                        result.update(
                            self._consumerSelectionKeys(bucket_idx, [cid], groups)
                        )
        return result

    def selectNeighborhood(self, context):
        self._prepareContext(context)
        metadata = context.get("metadata", {}) or {}
        groups = metadata.get("groups", {}) or {}
        if not groups:
            return set()

        critical_seeds = self._find_critical_path_seeds(
            context.get("incumbent", {}) or {}, metadata
        )
        seed = self._pick_highest_impact_seed(critical_seeds, metadata)
        if seed is None:
            candidates = [
                key
                for key, group in groups.items()
                if group.get("kind") == "selection"
                and group.get("selectable", False)
            ]
            seed = self.random.choice(candidates) if candidates else next(iter(groups), None)
        if seed is None:
            return set()

        neighborhood = {seed}
        neighborhood.update(
            self._expand_alternate_support_nodes(
                neighborhood,
                context.get("incumbent", {}) or {},
                metadata,
                self.target_size,
            )
        )
        neighborhood = self._expand_incumbent_support_nodes(
            neighborhood,
            context.get("incumbent", {}) or {},
            metadata,
            self.max_neighborhood_size,
        )
        neighborhood.update(self._get_coupled_inplace_keys(neighborhood, metadata))
        neighborhood.update(self._get_coupled_cache_clusters(neighborhood, metadata))
        return {key for key in neighborhood if key in groups}
