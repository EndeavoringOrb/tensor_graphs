"""Cache-first, dependency-aware neighborhoods for exact LNS."""

import math
from dataclasses import dataclass

from .structural import StructuralNeighborhoodSelector


@dataclass(frozen=True)
class _CacheCandidate:
    """One cache state to try during one selector invocation."""

    priority: tuple
    seed_keys: frozenset
    target_fixings: tuple
    cache_key: str = ""


class CacheNeighborhoodSelector(StructuralNeighborhoodSelector):
    """Try one ranked cache neighborhood at a time.

    Cache candidates are consumed from the end of the rank list.  Consequently
    candidates that enable a cache (``cache=1``) are placed at the end and are
    tried before cache-disabling candidates.  The support closure is computed
    only for the candidate selected during the current call.
    """

    def __init__(self, target_size=15, seed=None):
        # The inherited structural helpers require this argument internally,
        # but CacheNeighborhoodSelector deliberately does not impose a
        # neighborhood-size limit.
        super().__init__(target_size, max(1, int(target_size)), seed)
        self._candidate_signature = None
        self._candidates = []
        self._candidate_targets = {}
        self._classes_by_base = {}
        self._candidates_by_base = {}
        self._selected_parents = {}
        self._candidate_phase = None

    def _prepareContext(self, context):
        super()._prepareContext(context)

        self._classes_by_base = {}
        for bucket_idx, classes in self._bucket_classes.items():
            for eclass_id, cls in classes.items():
                base_eclass_id = self._asInt(cls.get("base_eclass_id"), eclass_id)
                self._classes_by_base.setdefault(base_eclass_id, []).append(
                    (bucket_idx, eclass_id, cls)
                )

        problem_data = self._problemData()
        self._candidates_by_base = {
            self._asInt(item.get("base_eclass_id")): item
            for item in problem_data.get("candidates", [])
            if isinstance(item, dict) and item.get("base_eclass_id") is not None
        }

        self._selected_parents = {}
        for bucket_idx, extraction in self._bucket_extractions.items():
            for eclass_id in self._activeIds(bucket_idx, extraction):
                enode = self._selectedEnode(bucket_idx, eclass_id, extraction)
                if not enode:
                    continue
                for child_id in enode.get("children", []):
                    child_id = self._asInt(child_id, child_id)
                    self._selected_parents.setdefault(
                        (bucket_idx, child_id), set()
                    ).add(self._selectionKeyFromParts(bucket_idx, eclass_id))

    @staticmethod
    def _shapeVolume(shape):
        if not isinstance(shape, (list, tuple)):
            return 0.0
        volume = 1
        try:
            for dimension in shape:
                dimension = int(dimension)
                if dimension < 0:
                    return 0.0
                volume *= dimension
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return float(volume)

    @staticmethod
    def _selectionKeyFromParts(bucket_idx, eclass_id):
        return f"selection:{int(bucket_idx)}:{int(eclass_id)}"

    def getCandidateFixings(self):
        """Return target primary assignments for the selected candidate."""
        return dict(self._candidate_targets)

    def _incumbentAssignments(self, context):
        return (context.get("incumbent", {}) or {}).get("primary_assignments", {}) or {}

    def _assignment(self, assignments, key):
        return self._lookup(assignments, key)

    def _classesForBase(self, base_eclass_id):
        return self._classes_by_base.get(base_eclass_id, ())

    def _candidateForBase(self, base_eclass_id, cache_key, groups, assignments):
        candidate = self._candidates_by_base.get(base_eclass_id)
        if candidate is None:
            return None

        current_cache = int(bool(self._assignment(assignments, cache_key)))
        target_cache = 1 - current_cache
        matching = []
        scatter_rank = 0
        best_ratio = 0.0
        estimated_savings = 0.0
        full_bucket_idx = self._asInt(
            self._problemData().get("full_bucket_idx"), 0
        )
        for bucket_idx, eclass_id, cls in self._classesForBase(base_eclass_id):
            selection_key = self._selectionKeyFromParts(bucket_idx, eclass_id)
            group = groups.get(selection_key, {})
            if not group.get("selectable"):
                continue
            if not self._matchesCacheClass(cls, candidate):
                continue
            matching.append(selection_key)

            if bucket_idx != full_bucket_idx:
                cache_enodes = [
                    enode
                    for enode in self._validEnodes(
                        bucket_idx,
                        eclass_id,
                        {cache_key: target_cache},
                    )
                    if enode.get("is_cache")
                ]
                if cache_enodes:
                    selected = self._selectedEnode(bucket_idx, eclass_id)
                    try:
                        selected_cost = float(selected.get("cost", 0.0))
                    except (AttributeError, TypeError, ValueError, OverflowError):
                        selected_cost = 0.0
                    try:
                        cache_cost = min(
                            float(enode.get("cost", 0.0))
                            for enode in cache_enodes
                        )
                    except (TypeError, ValueError, OverflowError):
                        cache_cost = selected_cost
                    if math.isfinite(selected_cost) and math.isfinite(cache_cost):
                        estimated_savings += max(
                            0.0,
                            selected_cost - cache_cost,
                        )

            scatter_volumes = []
            for enode in cls.get("enodes", []):
                if not enode.get("is_scatter"):
                    continue
                children = enode.get("children", [])
                if not children:
                    continue
                child = self._classesForBucket(bucket_idx).get(
                    self._asInt(children[0], children[0]), {}
                )
                update_volume = self._shapeVolume(child.get("shape"))
                if update_volume > 0:
                    scatter_volumes.append(update_volume)
            if scatter_volumes:
                scatter_rank = 1
                current_volume = self._shapeVolume(cls.get("shape"))
                if current_volume <= 0:
                    current_volume = float(cls.get("raw_size_bytes", 0) or 0)
                best_ratio = max(
                    best_ratio,
                    current_volume / min(scatter_volumes),
                )

        if not matching:
            return None

        # The tuple is consumed from the end.  Cache enabling therefore wins,
        # followed by estimated execution savings, scatter-capable classes,
        # and then the largest update ratio.
        priority = (
            2 if target_cache else 1,
            estimated_savings,
            scatter_rank,
            best_ratio,
            -base_eclass_id,
        )
        return _CacheCandidate(
            priority=priority,
            seed_keys=frozenset({cache_key, *matching}),
            target_fixings=((cache_key, target_cache),),
            cache_key=cache_key,
        )

    def _matchesCacheClass(self, cls, candidate):
        if cls.get("mem_space") != candidate.get("mem_space"):
            return False
        try:
            if int(cls.get("raw_size_bytes", -1)) != int(
                candidate.get("raw_size_bytes", -2)
            ):
                return False
            class_size = int(cls.get("size_bytes", -1))
            candidate_size = int(candidate.get("size_bytes", -2))
            if cls.get("mem_space", {}).get("type") != 0:
                alignment = 4096
                candidate_size = (
                    (candidate_size + alignment - 1) // alignment
                ) * alignment
            return class_size == candidate_size
        except (TypeError, ValueError, OverflowError):
            return False

    def _validEnodes(self, bucket_idx, eclass_id, cache_targets):
        cls = self._classesForBucket(bucket_idx).get(eclass_id, {})
        classes = self._classesForBucket(bucket_idx)
        problem_data = self._problemData()
        caching_enabled = not problem_data.get("disable_caching", False)
        full_bucket_idx = self._asInt(problem_data.get("full_bucket_idx"), 0)
        base_eclass_id = self._asInt(cls.get("base_eclass_id"), eclass_id)
        cache_key = self._cacheKey(base_eclass_id)
        cache_enabled = cache_targets.get(cache_key)
        if cache_enabled is None:
            cache_enabled = bool(
                self._assignment(
                    self._incumbentAssignments(self._context), cache_key
                )
            )
        candidate = self._candidates_by_base.get(base_eclass_id, {})
        clean = {
            self._asInt(value, value)
            for value in self._bucket_specs.get(bucket_idx, {}).get(
                "clean_eclasses", []
            )
        }

        result = []
        for enode in cls.get("enodes", []):
            children = [self._asInt(child, child) for child in enode.get("children", [])]
            if any(child not in classes for child in children):
                continue
            if enode.get("mem_space", cls.get("mem_space")) != cls.get("mem_space"):
                continue
            try:
                cost = float(enode.get("cost"))
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(cost) or cost < 0:
                continue
            if not caching_enabled and (enode.get("is_cache") or enode.get("is_scatter")):
                continue
            if enode.get("is_cache"):
                clean_buckets = candidate.get("clean_buckets")
                clean_buckets = (
                    {
                        self._asInt(value, value) for value in clean_buckets
                    }
                    if clean_buckets is not None
                    else None
                )
                if (
                    not cache_enabled
                    or bucket_idx == full_bucket_idx
                    or eclass_id not in clean
                    or (clean_buckets is not None and bucket_idx not in clean_buckets)
                    or children
                ):
                    continue
            if enode.get("is_scatter") and (not cache_enabled or bucket_idx == full_bucket_idx):
                continue
            if enode.get("is_view") and (
                not children
                or classes[children[0]].get("mem_space") != cls.get("mem_space")
            ):
                continue
            result.append(enode)
        return result

    def _currentActive(self, bucket_idx, eclass_id):
        extraction = self._extractionForBucket(bucket_idx)
        return eclass_id in self._activeIds(bucket_idx, extraction)

    def _currentChildren(self, bucket_idx, eclass_id):
        enode = self._selectedEnode(bucket_idx, eclass_id)
        if not enode:
            return set()
        return {
            self._asInt(child, child) for child in enode.get("children", [])
        }

    def _hasFrozenSupport(self, bucket_idx, eclass_id, nodes, cache_targets):
        bucket = self._bucket_specs.get(bucket_idx, {})
        if self._asInt(bucket.get("root_eclass_id")) == eclass_id:
            return True

        cls = self._classesForBucket(bucket_idx).get(eclass_id, {})
        base_eclass_id = self._asInt(cls.get("base_eclass_id"), eclass_id)
        cache_key = self._cacheKey(base_eclass_id)
        cache_value = cache_targets.get(cache_key)
        if cache_value is None:
            cache_value = bool(
                self._assignment(self._incumbentAssignments(self._context), cache_key)
            )
        if cache_value and self._cacheKeyMatchesClass(cache_key, bucket_idx, eclass_id):
            return True

        if any(
            parent_key not in nodes
            for parent_key in self._selected_parents.get(
                (bucket_idx, eclass_id), ()
            )
        ):
            return True
        return False

    def _cacheKeyMatchesClass(self, cache_key, bucket_idx, eclass_id):
        base_eclass_id = self._asInt(cache_key.split(":")[-1])
        candidate = self._candidates_by_base.get(base_eclass_id)
        cls = self._classesForBucket(bucket_idx).get(eclass_id)
        return candidate is not None and cls is not None and self._matchesCacheClass(cls, candidate)

    def _supportClosure(self, candidate, metadata):
        groups = metadata.get("groups", {}) or {}
        nodes = set(candidate.seed_keys)
        cache_targets = dict(candidate.target_fixings)
        changed = True
        while changed:
            changed = False
            for key in list(nodes):
                parsed = self._parseSelectionKey(key)
                if parsed is None:
                    continue
                bucket_idx, eclass_id = parsed
                domains = self._validEnodes(bucket_idx, eclass_id, cache_targets)
                if not domains:
                    continue
                current_children = self._currentChildren(bucket_idx, eclass_id)
                domain_children = [
                    {
                        self._asInt(child, child)
                        for child in enode.get("children", [])
                    }
                    for enode in domains
                ]
                union_children = set().union(*domain_children)
                common_children = set.intersection(*domain_children)
                affected_children = (
                    current_children | union_children
                ) - common_children
                for child_id in affected_children:
                    child_key = self._selectionKeyFromParts(bucket_idx, child_id)
                    if child_key not in groups:
                        continue
                    if not self._currentActive(bucket_idx, child_id):
                        needs_selection = True
                    else:
                        needs_selection = not self._hasFrozenSupport(
                            bucket_idx, child_id, nodes, cache_targets
                        )
                    if not needs_selection or child_key in nodes:
                        continue
                    nodes.add(child_key)
                    changed = True

            for key in list(nodes):
                parsed = self._parseSelectionKey(key)
                if parsed is None:
                    continue
                inplace_key = self._inplaceKey(*parsed)
                if inplace_key in groups and inplace_key not in nodes:
                    nodes.add(inplace_key)
                    changed = True

        return {
            key for key in nodes if key in groups
        }

    def _candidateSignature(self, context):
        assignments = self._incumbentAssignments(context)
        return tuple(sorted((str(key), repr(value)) for key, value in assignments.items()))

    def _fallbackCandidates(self, metadata, assignments):
        candidates = []
        groups = metadata.get("groups", {}) or {}
        for key, group in groups.items():
            if group.get("kind") != "selection" or not group.get("selectable"):
                continue
            parsed = self._parseSelectionKey(key)
            if parsed is None:
                continue
            bucket_idx, eclass_id = parsed
            current = self._asInt(self._assignment(assignments, key))
            alternatives = [
                enode
                for enode in self._validEnodes(bucket_idx, eclass_id, {})
                if self._asInt(enode.get("enode_idx")) != current
            ]
            if not alternatives:
                continue
            target = min(
                alternatives,
                key=lambda enode: (
                    float(enode.get("cost", 0.0) or 0.0),
                    self._asInt(enode.get("enode_idx"), 0),
                ),
            )
            candidates.append(
                _CacheCandidate(
                    priority=(0, 0, 0.0, -self._asInt(eclass_id, 0)),
                    seed_keys=frozenset({key}),
                    target_fixings=((key, int(target["enode_idx"])),),
                )
            )
        return candidates

    def _buildCandidates(self, context, metadata):
        assignments = self._incumbentAssignments(context)
        groups = metadata.get("groups", {}) or {}
        candidates = []
        for key, group in groups.items():
            if group.get("kind") != "cache":
                continue
            base_eclass_id = self._asInt(group.get("base_eclass_id"))
            if base_eclass_id is None:
                continue
            candidate = self._candidateForBase(
                base_eclass_id, key, groups, assignments
            )
            if candidate is not None:
                candidates.append(candidate)
        return sorted(candidates, key=lambda candidate: candidate.priority)

    def selectNeighborhood(self, context):
        self._prepareContext(context)
        metadata = context.get("metadata", {}) or {}
        groups = metadata.get("groups", {}) or {}
        if not groups:
            self._candidate_targets = {}
            return set()

        assignments = self._incumbentAssignments(context)
        signature = self._candidateSignature(context)
        if signature != self._candidate_signature:
            self._candidate_signature = signature
            self._candidates = self._buildCandidates(context, metadata)
            self._candidate_phase = "cache"
            if not self._candidates:
                self._candidates = self._fallbackCandidates(metadata, assignments)
                self._candidate_phase = "fallback"

        elif not self._candidates and self._candidate_phase == "cache":
            self._candidates = self._fallbackCandidates(metadata, assignments)
            self._candidate_phase = "fallback"

        if not self._candidates:
            self._candidate_targets = {}
            return set()

        candidate = self._candidates.pop()
        self._candidate_targets = dict(candidate.target_fixings)
        final_set = self._supportClosure(candidate, metadata)
        print(f"selected neighborhood size: {len(final_set)}")
        return final_set
