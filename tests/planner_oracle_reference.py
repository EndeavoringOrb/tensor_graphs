"""Independent finite planner specification; no tensor_graphs or solver imports.

Supported scope is explicit: one CPP memory space, no preallocation or safe
in-place kernels, finite domains, views, CPU engines, caches, weighted buckets.
Unknown fields are errors so adversarial fixtures cannot silently lose meaning.
"""

from fractions import Fraction
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import tempfile


ORACLE_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
cache_directory = None
cache_hits = 0
cache_misses = 0


def configureCache(directory):
    global cache_directory, cache_hits, cache_misses
    cache_directory = None if directory is None else Path(directory)
    cache_hits = cache_misses = 0
    if cache_directory is not None:
        cache_directory.mkdir(parents=True, exist_ok=True)


def cacheStats():
    return {"hits": cache_hits, "misses": cache_misses}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checkKeys(value, allowed, label):
    require(isinstance(value, dict), f"{label} must be an object")
    require(not set(value) - set(allowed), f"Unknown fields in {label}: {set(value) - set(allowed)}")


def finiteNumber(value, label, maximum=256):
    require(type(value) in (int, float), f"{label} must be a number")
    number = Fraction(str(value))
    require(0 <= number <= maximum, f"{label} must be between 0 and {maximum}")
    require(number.denominator <= 16 and number.denominator & (number.denominator - 1) == 0,
            f"{label} must be a multiple of 1/16 (e.g. 0.5, not 0.1)")
    return value


def domainValues(values, label, maximum=255):
    require(isinstance(values, list) and values, f"{label} must be a nonempty list")
    require(all(type(v) is int and 0 <= v <= maximum for v in values),
            f"{label} values must be integers in [0,{maximum}]")
    require(len(set(values)) == len(values), f"{label} contains duplicates")
    values = sorted(values)
    require(values[-1] - values[0] < 32 or len(values) == values[-1] - values[0] + 1,
            f"{label}: sparse domains must fit in a 32-value window")
    return values


def normalizeCase(case):
    checkKeys(case, {"name", "description", "capacity_pages", "caches", "buckets",
                     "nodes", "root", "expected_optimum", "cutoffs", "probes"}, "case")
    require(isinstance(case.get("name"), str) and case["name"], "Case needs a name")
    require(("buckets" in case) != ("nodes" in case), "Use either buckets or top-level nodes/root")
    buckets = case.get("buckets", [{"nodes": case.get("nodes"), "root": case.get("root"), "weight": 1}])
    require(isinstance(buckets, list) and 1 <= len(buckets) <= 4, "Use 1..4 buckets")
    caches = case.get("caches", {})
    require(isinstance(caches, dict), "caches must be an object")
    base_names = list(caches)
    normalized_buckets = []
    largest_extent = 1
    for b, bucket in enumerate(buckets):
        checkKeys(bucket, {"nodes", "root", "weight"}, f"bucket {b}")
        nodes = bucket.get("nodes")
        require(isinstance(nodes, dict) and 1 <= len(nodes) <= 6, "Use 1..6 named nodes per bucket")
        node_names = list(nodes)
        require(bucket.get("root") in nodes, f"bucket {b}: root must name a node")
        root = node_names.index(bucket["root"])
        normalized_nodes = []
        for c, (node_name, node) in enumerate(nodes.items()):
            checkKeys(node, {"alternatives", "base", "size_pages", "selected", "start", "offset"}, node_name)
            base = node.get("base", node_name)
            require(isinstance(base, str) and base, "base must be a nonempty name")
            if base not in base_names:
                base_names.append(base)
            size = node.get("size_pages", 1)
            require(type(size) is int and 1 <= size <= 32, "size_pages must be in [1,32]")
            alternatives = node.get("alternatives", [{}])
            require(isinstance(alternatives, list) and 1 <= len(alternatives) <= 30, "Use 1..30 alternatives")
            normalized_alternatives = []
            for alternative in alternatives:
                checkKeys(alternative, {"children", "cost", "engines", "op", "view"}, "alternative")
                children = alternative.get("children", [])
                require(isinstance(children, list) and all(ch in nodes for ch in children), "Unknown child node")
                op = alternative.get("op", "compute")
                require(op in ("compute", "input", "cache", "scatter"), f"Unsupported op: {op}")
                require(op not in ("cache", "scatter") or base in caches, "Cache/scatter needs a declared cache base")
                require(op not in ("input", "cache") or not children, "Input/cache nodes cannot have children")
                view = alternative.get("view", False)
                require(type(view) is bool, "view must be boolean")
                require(not view or (op == "compute" and len(children) == 1), "Views need exactly one child and op compute")
                engines = alternative.get("engines", [0])
                require(isinstance(engines, list) and all(type(e) is int and 0 <= e < 4 for e in engines), "Use engine IDs 0..3")
                require(len(set(engines)) == len(engines), "Repeated engine IDs")
                normalized_alternatives.append({
                    "children": [node_names.index(ch) for ch in children], "op": op, "view": view,
                    "cost": finiteNumber(alternative.get("cost", 1), "cost"), "engines": engines,
                })
            selected = domainValues(node.get("selected", list(range(0 if c != root else 1, len(alternatives) + 1))),
                                    "selected", len(alternatives))
            require(c != root or 0 not in selected, "Root must be mandatory")
            starts = domainValues(node.get("start", list(range(len(nodes)))), "start")
            offsets = domainValues(node.get("offset", [c]), "offset")
            largest_extent = max(largest_extent, max(offsets) + size)
            normalized_nodes.append({"name": node_name, "base_id": base_names.index(base) + 1,
                                     "size_pages": size, "alternatives": normalized_alternatives,
                                     "selected": selected, "start": starts, "offset": offsets})
        # Distinct classes in a bucket must not accidentally alias via base ID.
        require(len({n["base_id"] for n in normalized_nodes}) == len(normalized_nodes), "Duplicate base in one bucket; use a view for aliases")
        normalized_buckets.append({"nodes": normalized_nodes, "root": root,
                                   "weight": finiteNumber(bucket.get("weight", 1), "weight", 8)})
    normalized_caches = []
    for base, cache in caches.items():
        checkKeys(cache, {"domain", "size_pages"}, f"cache {base}")
        size = cache.get("size_pages", 1)
        require(type(size) is int and 1 <= size <= 32, "cache size_pages must be in [1,32]")
        copies = [n for b in normalized_buckets for n in b["nodes"] if n["base_id"] == base_names.index(base) + 1]
        require(copies and size >= max(n["size_pages"] for n in copies), "Cache size must cover every bucket copy")
        normalized_caches.append({"base_id": base_names.index(base) + 1, "size_pages": size,
                                  "domain": domainValues(cache.get("domain", [0, 1]), "cache domain", 1)})
    capacity = case.get("capacity_pages", largest_extent)
    require(type(capacity) is int and 1 <= capacity <= 256, "capacity_pages must be in [1,256]")
    is_dag = True
    for bucket in normalized_buckets:
        def visit(c, stack, done):
            if c in stack:
                return False
            if c in done:
                return True
            stack.add(c)
            for a in bucket["nodes"][c]["alternatives"]:
                for ch in a["children"]:
                    if not visit(ch, stack, done):
                        return False
            stack.remove(c)
            done.add(c)
            return True
        done = set()
        if not all(visit(c, set(), done) for c in range(len(bucket["nodes"]))):
            is_dag = False
    return {"capacity_pages": capacity, "caches": normalized_caches,
            "buckets": normalized_buckets, "is_dag": is_dag}


class EnumerationBudget:
    def __init__(self, limit):
        self.limit = limit
        self.work = 0

    def spend(self):
        self.work += 1
        require(self.work <= self.limit, f"Oracle enumeration exceeded {self.limit} steps; shrink the case")


def bucketFeasible(bucket, selected, starts, offsets, cached, capacity):
    """Check a complete local assignment directly; never call a propagator."""
    nodes = bucket["nodes"]
    active = {i for i, value in enumerate(selected) if value > 0}
    if bucket["root"] not in active:
        return False
    reachable = set()
    frontier = [bucket["root"]]
    while frontier:
        c = frontier.pop()
        if c in reachable:
            continue
        if c not in active:
            return False
        reachable.add(c)
        frontier.extend(nodes[c]["alternatives"][selected[c] - 1]["children"])
    if reachable != active:
        return False
    if len({starts[c] for c in active}) != len(active):
        return False
    owners = {}
    for c in sorted(active, key=lambda i: starts[i]):
        node = nodes[c]
        alternative = node["alternatives"][selected[c] - 1]
        if any(ch not in active or starts[ch] >= starts[c] for ch in alternative["children"]):
            return False
        if alternative["op"] in ("cache", "scatter") and not cached.get(node["base_id"], 0):
            return False
        if alternative["view"]:
            ch = alternative["children"][0]
            if offsets[c] != offsets[ch]:
                return False
            owners[c] = owners[ch]
        else:
            owners[c] = c
        if offsets[c] + node["size_pages"] > capacity:
            return False
    for c, node in enumerate(nodes):
        if cached.get(node["base_id"], 0) and c not in active:
            return False
    # Reads through view chains extend the owner's lifetime. Allocation sharing
    # is legal only after every read; no kernel in this schema is safe in-place.
    last_reads = {c: starts[c] for c in active}
    for reader in active:
        for ch in nodes[reader]["alternatives"][selected[reader] - 1]["children"]:
            last_reads[owners[ch]] = max(last_reads[owners[ch]], starts[reader])
    for c in active:
        for d in active:
            if c >= d or owners[c] == owners[d]:
                continue
            if max(offsets[c], offsets[d]) >= min(offsets[c] + nodes[c]["size_pages"], offsets[d] + nodes[d]["size_pages"]):
                continue
            def persistent(i):
                alt = nodes[i]["alternatives"][selected[i] - 1]
                return i == bucket["root"] or alt["op"] in ("input", "cache") or cached.get(nodes[i]["base_id"], 0)
            if persistent(c) or persistent(d):
                return False
            owner_c, owner_d = owners[c], owners[d]
            if not (last_reads[owner_c] < starts[owner_d] or last_reads[owner_d] < starts[owner_c]):
                return False
    return True


def bucketCost(bucket, selected, starts):
    finishes, engine_free = {}, {}
    for c in sorted((i for i, value in enumerate(selected) if value), key=lambda i: starts[i]):
        a = bucket["nodes"][c]["alternatives"][selected[c] - 1]
        if a["op"] in ("input", "cache"):
            finishes[c] = Fraction(0)
            continue
        dependency_finish = max((finishes[ch] for ch in a["children"]), default=Fraction(0))
        if a["view"]:
            finishes[c] = dependency_finish
            continue
        engines = a["engines"] or [0]
        begin = max(dependency_finish, max((engine_free.get(e, Fraction(0)) for e in engines), default=Fraction(0)))
        finish = begin + Fraction(str(a["cost"]))
        finishes[c] = finish
        for engine in engines:
            engine_free[engine] = finish
    return max(engine_free.values(), default=Fraction(0))


def bucketPlans(bucket, cached, capacity, budget):
    nodes = bucket["nodes"]
    for selected in product(*(n["selected"] for n in nodes)):
        budget.spend()
        active = [i for i, v in enumerate(selected) if v]
        # Cheap structural rejection before schedule/allocation enumeration.
        if bucket["root"] not in active:
            continue
        if any(any(not selected[ch] for ch in nodes[c]["alternatives"][selected[c] - 1]["children"]) for c in active):
            continue
        for ranks in product(*(nodes[c]["start"] for c in active)):
            budget.spend()
            if len(set(ranks)) != len(ranks):
                continue
            starts = [0] * len(nodes)
            for c, rank in zip(active, ranks):
                starts[c] = rank
            if any(any(starts[ch] >= starts[c] for ch in nodes[c]["alternatives"][selected[c] - 1]["children"]) for c in active):
                continue
            for addresses in product(*(nodes[c]["offset"] for c in active)):
                budget.spend()
                offsets = [0] * len(nodes)
                for c, address in zip(active, addresses):
                    offsets[c] = address
                if bucketFeasible(bucket, selected, starts, offsets, cached, capacity):
                    yield {"selected": list(selected), "start": starts, "offset": offsets,
                           "cost": bucketCost(bucket, selected, starts)}


def globalFeasible(problem, cached, plans):
    if sum(c["size_pages"] for c in problem["caches"] if cached[c["base_id"]]) > problem["capacity_pages"]:
        return False
    for cache in problem["caches"]:
        base = cache["base_id"]
        if not cached[base]:
            continue
        addresses = {plan["offset"][c] for bucket, plan in zip(problem["buckets"], plans)
                     for c, node in enumerate(bucket["nodes"]) if node["base_id"] == base}
        if len(addresses) != 1:
            return False
        address = next(iter(addresses))
        if address + cache["size_pages"] > problem["capacity_pages"]:
            return False
        # A cache reserves its maximum declared footprint in every bucket,
        # including buckets in which its class does not occur.
        for bucket, plan in zip(problem["buckets"], plans):
            for c, node in enumerate(bucket["nodes"]):
                if not plan["selected"][c] or node["base_id"] == base:
                    continue
                a = node["alternatives"][plan["selected"][c] - 1]
                # Views of this cache legitimately occupy its range.
                owner = c
                while bucket["nodes"][owner]["alternatives"][plan["selected"][owner] - 1]["view"]:
                    owner = bucket["nodes"][owner]["alternatives"][plan["selected"][owner] - 1]["children"][0]
                if bucket["nodes"][owner]["base_id"] == base:
                    continue
                if max(address, plan["offset"][c]) < min(address + cache["size_pages"], plan["offset"][c] + node["size_pages"]):
                    return False
    return True


def validateWitness(problem, witness):
    """Validate a returned plan directly, without enumerating the whole case.

    An extracted dispatch order omits absolute ranks. Greedily pick the earliest
    allowed rank after the previous rank; this finds a realization whenever one
    exists for that total order, even with sparse domains.
    """
    checkKeys(witness, {"cached", "buckets"}, "native witness")
    values = witness["cached"]
    require(isinstance(values, list) and len(values) == len(problem["caches"]), "Wrong witness cache count")
    require(all(type(v) is int and v in c["domain"] for c, v in zip(problem["caches"], values)), "Witness cache choice outside domain")
    cached = {c["base_id"]: v for c, v in zip(problem["caches"], values)}
    require(isinstance(witness["buckets"], list) and len(witness["buckets"]) == len(problem["buckets"]), "Wrong witness bucket count")
    plans = []
    for bucket, claimed in zip(problem["buckets"], witness["buckets"]):
        checkKeys(claimed, {"selected", "order", "offset"}, "native bucket witness")
        nodes = bucket["nodes"]
        for field in ("selected", "order", "offset"):
            require(isinstance(claimed[field], list) and len(claimed[field]) == len(nodes), f"Wrong witness {field} count")
            require(all(type(value) is int for value in claimed[field]), f"Noninteger witness {field}")
        selected, offsets = claimed["selected"], claimed["offset"]
        require(all(v in n["selected"] for v, n in zip(selected, nodes)), "Witness selection outside domain")
        active = [c for c, value in enumerate(selected) if value]
        require(all(offsets[c] in nodes[c]["offset"] for c in active), "Witness address outside domain")
        order = claimed["order"]
        require(all(order[c] >= 0 for c in active) and len({order[c] for c in active}) == len(active), "Invalid witness dispatch order")
        starts = [0] * len(nodes)
        previous_rank = -1
        for c in sorted(active, key=lambda i: order[i]):
            possible = [rank for rank in nodes[c]["start"] if rank > previous_rank]
            require(possible, "Witness order has no realization within start domains")
            starts[c] = previous_rank = min(possible)
        require(bucketFeasible(bucket, selected, starts, offsets, cached, problem["capacity_pages"]), "Witness violates local feasibility")
        plans.append({"selected": selected, "start": starts, "offset": offsets,
                      "cost": bucketCost(bucket, selected, starts)})
    require(globalFeasible(problem, cached, plans), "Witness violates shared cache constraints")
    return sum((Fraction(str(b["weight"])) * p["cost"] for b, p in zip(problem["buckets"], plans)), Fraction(0))


def enumerateUncached(problem, limit=250_000, witness_filter=None):
    budget = EnumerationBudget(limit)
    best_cost, witness = None, None
    feasible_count = 0
    for cache_values in product(*(c["domain"] for c in problem["caches"])):
        cached = {c["base_id"]: value for c, value in zip(problem["caches"], cache_values)}
        local_plans = [list(bucketPlans(b, cached, problem["capacity_pages"], budget)) for b in problem["buckets"]]
        for plans in product(*local_plans):
            budget.spend()
            if not globalFeasible(problem, cached, plans):
                continue
            if witness_filter is not None and not witness_filter(cache_values, plans):
                continue
            feasible_count += 1
            cost = sum((Fraction(str(b["weight"])) * p["cost"] for b, p in zip(problem["buckets"], plans)), Fraction(0))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                witness = {"cached": list(cache_values), "buckets": [
                    {key: value for key, value in p.items() if key != "cost"} for p in plans]}
    return {"cost": best_cost, "witness": witness, "feasible_count": feasible_count, "steps": budget.work}


def enumerateOptimum(problem, limit=250_000, witness_filter=None):
    """Cache reference truth only, never native pass/fail or filtered witnesses."""
    global cache_hits, cache_misses
    if cache_directory is None or witness_filter is not None:
        return enumerateUncached(problem, limit, witness_filter)
    problem_json = json.dumps(problem, sort_keys=True, separators=(",", ":"), allow_nan=False)
    key = hashlib.sha256((ORACLE_SOURCE_HASH + "\n" + problem_json).encode("utf-8")).hexdigest()
    path = cache_directory / f"{key}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        require(cached["key"] == key, f"Oracle cache key mismatch: {path}")
        payload = cached["result"]
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        require(digest == cached["result_hash"], f"Oracle cache checksum mismatch: {path}; rerun with --no-cache")
        require(payload["steps"] <= limit, f"Cached enumeration requires {payload['steps']} steps, above limit {limit}")
        result = dict(payload)
        result["cost"] = None if payload["cost"] is None else Fraction(payload["cost"])
        cache_hits += 1
        return result
    cache_misses += 1
    result = enumerateUncached(problem, limit)
    payload = dict(result)
    payload["cost"] = None if result["cost"] is None else str(result["cost"])
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    record = {"key": key, "oracle_source_hash": ORACLE_SOURCE_HASH, "result_hash": digest, "result": payload}
    # Atomic replacement allows parallel reviewers to run the same case safely.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_directory,
                                     prefix=f".{key}.", suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return result
