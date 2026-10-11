"""Adversarial exhaustive correctness gate for small declarative planner cases.

Run: .venv/Scripts/python.exe -m tests.test_planner_oracle
Add cases: drop JSON files into tests/planner_oracle_cases (no registration).
"""

import argparse
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tests.planner_oracle_reference import cacheStats, configureCache, enumerateOptimum, normalizeCase, require, validateWitness


CASE_DIR = Path(__file__).resolve().parent / "planner_oracle_cases"
RESULT_PREFIX = "PLANNER_ORACLE_RESULT="


def displayCost(cost):
    return None if cost is None else str(cost)


def nativeCall(problem, mode, cutoff=None):
    import tensor_graphs
    require(hasattr(tensor_graphs, "_run_planner_oracle_case"),
            "Rebuild bindings: .venv/Scripts/python.exe build.py --targets bindings --no-lint --log-level INFO")
    args = [json.dumps(problem), mode]
    if cutoff is not None:
        args.append(float(cutoff))
    return json.loads(tensor_graphs._run_planner_oracle_case(*args))


def applyProbe(problem, fixes):
    out = deepcopy(problem)
    require(isinstance(fixes, list), "probe fixes must be a list")
    for fix in fixes:
        require(isinstance(fix, dict), "probe fix must be an object")
        if "cache" in fix:
            require(set(fix) == {"cache", "domain"}, "cache fix needs cache index and domain")
            target = out["caches"][fix["cache"]]
            field = "domain"
        else:
            require(set(fix) == {"bucket", "node", "field", "domain"}, "node fix needs bucket, node, field, domain")
            bucket = out["buckets"][fix["bucket"]]
            targets = [node for node in bucket["nodes"] if node["name"] == fix["node"]]
            require(len(targets) == 1, f"Unknown probe node {fix['node']}")
            target = targets[0]
            field = fix["field"]
            require(field in ("selected", "start", "offset"), "Unknown probe field")
        domain = fix["domain"]
        require(isinstance(domain, list) and domain and all(type(v) is int for v in domain), "Invalid probe domain")
        require(len(set(domain)) == len(domain) and set(domain) <= set(target[field]), "Probe must narrow an existing domain")
        target[field] = sorted(domain)
    return out


def checkPropagation(problem, reference, limit, cutoff=None):
    result = nativeCall(problem, "propagate", cutoff)
    optimum = reference["cost"]
    improving_exists = optimum is not None and (cutoff is None or optimum < cutoff)
    if improving_exists:
        require(result["consistent"], f"Propagation removed every optimum: {result['conflict']}; witness={reference['witness']}")
    if not result["consistent"]:
        return
    if optimum is not None:
        bound = result["lower_bound"]
        require(type(bound) in (float, int) and math.isfinite(bound), "Invalid native lower bound")
        require(bound <= float(optimum) + 1e-5,
                f"Invalid lower bound {bound} exceeds independent optimum {optimum}; witness={reference['witness']}")
    survivors = enumerateOptimum(result["problem"], limit)
    if improving_exists:
        require(survivors["cost"] == optimum,
                f"Propagation lost all optimal representatives: before={optimum}, after={survivors['cost']}; witness={reference['witness']}")


def checkSearch(problem, reference, limit):
    result = nativeCall(problem, "solve")
    optimum = reference["cost"]
    require(result["exhausted"], "Small native search did not exhaust its frontier")
    if optimum is None:
        require(result["cost"] is None and not result["solved"], "Native planner returned a plan for an infeasible case")
        return
    require(result["solved"] and result["cost"] is not None,
            f"Native search lost every feasible plan; optimum={optimum}, witness={reference['witness']}")
    require(result["cost"] == float(optimum),
            f"Wrong native optimum: expected={optimum}, got={result['cost']}; witness={reference['witness']}")
    require("witness" in result, "Native search returned no checkable incumbent")
    try:
        validated_cost = validateWitness(problem, result["witness"])
    except ValueError as error:
        raise ValueError(f"Native incumbent is infeasible: {error}; witness={result['witness']}") from error
    require(float(validated_cost) == result["cost"],
            f"Native incumbent has an incorrect cost: {result['witness']}")


def runCase(path, limit, reference_only):
    start_time = time.perf_counter()
    case = json.loads(path.read_text(encoding="utf-8-sig"))
    problem = normalizeCase(case)
    reference = enumerateOptimum(problem, limit)
    if "expected_optimum" in case:
        expected = case["expected_optimum"]
        require(reference["cost"] == (None if expected is None else Fraction(str(expected))),
                f"Case's expected optimum {expected} disagrees with independent enumeration {reference['cost']}")
    variants = [("root", problem, reference)]
    for probe in case.get("probes", []):
        require(isinstance(probe, dict) and set(probe) == {"name", "fixes"}, "probe needs name and fixes")
        narrowed = applyProbe(problem, probe["fixes"])
        variants.append((probe["name"], narrowed, enumerateOptimum(narrowed, limit)))
    checks = 0
    if not reference_only:
        for name, variant, oracle in variants:
            try:
                checkPropagation(variant, oracle, limit)
                checks += 1
                # Above, exactly at, and below the independently known optimum.
                # Cutoff-only checks never supply an unvalidated incumbent to solve.
                cutoffs = case.get("cutoffs")
                if cutoffs is None:
                    cutoffs = [] if oracle["cost"] is None else [oracle["cost"] + 1, oracle["cost"], max(0, oracle["cost"] - 1)]
                for cutoff in cutoffs:
                    checkPropagation(variant, oracle, limit, Fraction(str(cutoff)))
                    checks += 1
                checkSearch(variant, oracle, limit)
                checks += 1
            except Exception as error:
                raise AssertionError(f"{case['name']} / {name}: {error}") from error
    return {"name": case["name"], "passed": True, "optimum": displayCost(reference["cost"]),
            "feasible_assignments": reference["feasible_count"], "oracle_steps": reference["steps"],
            "native_checks": checks, "seconds": time.perf_counter() - start_time}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--case", help="Run one case by name, stem, or JSON path")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument("--timeout", type=float, default=20, help="Hard wall-clock limit per case, including native search")
    parser.add_argument("--max-steps", type=int, default=250_000, help="Maximum enumeration work per oracle call")
    parser.add_argument("--reference-only", action="store_true", help="Enumerate fixture semantics without loading the native solver")
    parser.add_argument("--json-report", type=Path, help="Write machine-readable pass/fail results")
    parser.add_argument("--cache-dir", type=Path, default=CASE_DIR.parents[1] / ".cache" / "planner_oracle",
                        help="Content-addressed reference enumeration cache")
    parser.add_argument("--no-cache", action="store_true", help="Recompute all reference results without reading or writing the cache")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.timeout > 0 and args.max_steps > 0, "Timeout and step limit must be positive")
    if args.worker:
        configureCache(None if args.no_cache else args.cache_dir)
        try:
            result = runCase(args.worker, args.max_steps, args.reference_only)
        except Exception:
            result = {"name": args.worker.stem, "passed": False, "error": traceback.format_exc()}
        result["cache"] = cacheStats()
        print(RESULT_PREFIX + json.dumps(result))
        return 0 if result["passed"] else 1
    paths = sorted(args.cases_dir.rglob("*.json"))
    if args.case:
        explicit = Path(args.case)
        if explicit.is_file():
            paths = [explicit.resolve()]
        else:
            paths = [p for p in paths if p.stem == args.case or json.loads(p.read_text(encoding="utf-8-sig")).get("name") == args.case]
    require(paths, "No oracle cases found (empty suites must not pass)")
    if args.list:
        for path in paths:
            print(f"{path.stem}: {json.loads(path.read_text(encoding='utf-8-sig'))['name']}")
        return 0
    results = []
    for path in paths:
        command = [sys.executable, "-m", "tests.test_planner_oracle", "--worker", str(path.resolve()),
                   "--max-steps", str(args.max_steps)]
        if args.reference_only:
            command.append("--reference-only")
        if args.no_cache:
            command.append("--no-cache")
        else:
            command.extend(["--cache-dir", str(args.cache_dir.resolve())])
        try:
            process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                     text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=args.timeout)
            records = [line[len(RESULT_PREFIX):] for line in process.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
            require(len(records) == 1, f"Worker crashed or returned invalid output (exit {process.returncode}):\n{process.stdout[-4000:]}\n{process.stderr[-4000:]}")
            result = json.loads(records[0])
            if process.returncode != 0:
                result["passed"] = False
        except Exception as error:
            result = {"name": path.stem, "passed": False, "error": str(error)}
        result["file"] = str(path)
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['name']}"
              + (f" optimum={result['optimum']} ({result['seconds']:.3f}s) cache={result['cache']}" if result["passed"] else f"\n{result.get('error', 'Worker failed')}"))
    passed = all(result["passed"] for result in results)
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps({"passed": passed, "cases": results}, indent=2) + "\n", encoding="utf-8")
    print(f"{sum(r['passed'] for r in results)}/{len(results)} cases passed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
