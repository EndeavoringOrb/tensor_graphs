"""Goal validation and PR promotion on the same worker as kernel evaluation.

Only operators configure executable recipes. Candidate submissions cannot supply
commands, paths, environment variables, or GitHub credentials.
"""
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import uuid
from pathlib import Path

from flask import jsonify, render_template, request
from psycopg.types.json import Jsonb

from .storage import fingerprint


def promotionConfig(config_path=None):
    config_path = config_path or os.environ.get("KERNEL_BENCH_PROMOTION_CONFIG")
    if not config_path:
        return None
    config = json.loads(Path(config_path).read_text())
    if not isinstance(config.get("command"), list) or not config["command"] or not all(
            isinstance(arg, str) and arg for arg in config["command"]):
        raise ValueError("Promotion command must be an argv array")
    if not any("{report}" in arg for arg in config["command"]) or not any("{cache}" in arg for arg in config["command"]):
        raise ValueError("Promotion command must include {report} and {cache}")
    for key in ("kernel_speedup", "goal_speedup"):
        value = config.get(key, 1.02 if key == "kernel_speedup" else 1.01)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 1:
            raise ValueError(f"{key} must be finite and greater than one")
        config[key] = value
    timeout = config.get("timeout_seconds", 7200)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 86400:
        raise ValueError("timeout_seconds must be an integer in 1..86400")
    config["timeout_seconds"] = timeout
    if "planning_command" in config and (not isinstance(config["planning_command"], list) or
            not config["planning_command"] or not all(isinstance(arg, str) and arg for arg in config["planning_command"]) or
            not all(any("{" + field + "}" in arg for arg in config["planning_command"]) for field in ("report", "cache"))):
        raise ValueError("planning_command must be an argv array with {report} and {cache}")
    if config.get("github_repo") is not None and (not isinstance(config["github_repo"], str) or
            not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", config["github_repo"])):
        raise ValueError("github_repo must be an owner/repository name")
    return config


def promotionEligibility(parent, config):
    if config is None:
        return {"eligible": False, "reason": "promotion_unconfigured"}
    if parent["mode"] != "evaluate" or parent["status"] != "completed":
        return {"eligible": False, "reason": "completed_evaluation_required"}
    result = parent["result"]
    comparisons = result.get("comparisons", [])
    if any(row.get("speedup") is not None and row["speedup"] >= config["kernel_speedup"] for row in comparisons):
        return {"eligible": True, "reason": "kernel_speedup"}
    tested = set(result.get("tested_configuration_ids", []))
    coverage = [row["configuration_id"] for row in comparisons
                if row.get("best_existing_kernel") is None and row.get("best_existing_ms") is None
                and row.get("candidate_ms", 0) > 0 and row["configuration_id"] in tested]
    if coverage:
        return {"eligible": True, "reason": "executable_coverage_added", "configuration_ids": coverage}
    return {"eligible": False, "reason": "kernel_speedup_gate_not_met"}


def enqueuePromotion(store, job):
    parent = store.getJob(job["id"])
    if parent is None or parent["mode"] != "evaluate" or parent["status"] != "completed":
        raise ValueError("Promotion requires a completed evaluation")
    config = store.getGoal(parent["goal_id"])["metadata"].get("promotion_recipe")
    if not promotionEligibility(parent, config)["eligible"]:
        return None
    # Persist the operator's recipe now, so queued jobs cannot silently change goals.
    rows = store.query("""INSERT INTO kernel_bench.jobs
        (id, goal_id, target_id, idea, source, kernel_name, backend, mode, agent_id, parent_job_id, result)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'promote', %s, %s, %s)
        ON CONFLICT (parent_job_id) WHERE mode = 'promote' DO UPDATE SET parent_job_id = EXCLUDED.parent_job_id
        RETURNING id""", (uuid.uuid4(), parent["goal_id"], parent["target_id"], parent["idea"], parent["source"],
                             parent["kernel_name"], parent["backend"], parent.get("agent_id"), parent["id"],
                             Jsonb({"recipe": config})))
    return str(rows[0]["id"])


def enqueueGoalValidation(store, goal_id):
    config = store.getGoal(goal_id)["metadata"].get("promotion_recipe")
    if config is None:
        return None
    targets = store.getTargets(goal_id)
    if not targets:
        raise ValueError("Cannot validate a goal with no reference targets")
    rows = store.query("""INSERT INTO kernel_bench.jobs
        (id, goal_id, target_id, idea, source, kernel_name, backend, mode, result)
        VALUES (%s, %s, %s, 'Register and measure logical goal', '', 'Goal baseline', 'cpu', 'promote', %s)
        ON CONFLICT (goal_id) WHERE mode = 'promote' AND parent_job_id IS NULL
        DO UPDATE SET goal_id = EXCLUDED.goal_id RETURNING id""",
                       (uuid.uuid4(), goal_id, targets[0]["id"], Jsonb({"recipe": config, "baseline_only": True})))
    return str(rows[0]["id"])


def validateReport(report):
    if report.get("correctness_passed") is not True:
        raise ValueError("Goal runner did not pass correctness")
    if not isinstance(report.get("logical_graph"), dict) or not report["logical_graph"]:
        raise ValueError("Goal runner must provide the logical graph definition")
    buckets = report.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("Goal runner returned no buckets")
    for bucket in buckets:
        weight = bucket.get("weight")
        samples = bucket.get("samples_ms")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("Bucket weights must be finite and nonnegative")
        if not isinstance(bucket.get("bucket"), dict) or not isinstance(bucket.get("kernel_ids"), list):
            raise ValueError("Bucket descriptors and selected kernel IDs are required")
        if weight == 0 and samples == []:
            bucket["median_ms"] = 0.0
            continue
        if not isinstance(samples, list) or len(samples) < 8 or not all(
                not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                for value in samples):
            raise ValueError("Each bucket requires at least eight finite positive timing samples")
        bucket["median_ms"] = statistics.median(samples)
    total_weight = sum(bucket["weight"] for bucket in buckets)
    if total_weight <= 0:
        raise ValueError("At least one bucket must have positive weight")
    report["weighted_ms"] = sum(bucket["weight"] * bucket["median_ms"] for bucket in buckets) / total_weight
    return report


def goalDefinition(report, hardware):
    # Kernel choices, timings, source commits, and compiler settings are evidence,
    # not part of the logical workload's identity.
    return {"logical_graph": report["logical_graph"],
            "buckets": [row["bucket"] for row in report["buckets"]],
            "bucket_weights": [row["weight"] for row in report["buckets"]],
            "hardware_system": hardware}


def verifyDiscoveryGoal(report, cache_path):
    from utils.binary import loadCacheFile
    entries = loadCacheFile(cache_path, lazy=True)
    metadata = next((entry for entry in entries if entry["type"] == "metadata"), None)
    buckets = [entry["graph"]["bucket"] for entry in entries if entry["type"] == "compiled_bucket"]
    if metadata is None:
        raise ValueError("Discovery cache contains no goal metadata")
    declared = {"buckets": buckets, "bucket_weights": metadata["bucketWeights"]}
    measured = {"buckets": [row["bucket"] for row in report["buckets"]],
                "bucket_weights": [row["weight"] for row in report["buckets"]]}
    # Cache readers return integer map keys; JSON runners return string keys.
    declared = json.loads(json.dumps(declared))
    if fingerprint(declared) != fingerprint(measured) or report["logical_graph"].get("root") != metadata["rootId"]:
        raise ValueError("Validation recipe differs from the discovery goal's root, buckets, or weights; configure a matching goal cache")


def promoteJob(store, job):
    from .jobs import PROJECT_ROOT, TIMEOUTS, getPythonExe, hardwareIdentity, runCmd

    config = job["result"]["recipe"]
    metadata = store.getGoal(job["goal_id"])["metadata"]
    source_commit = metadata.get("source_commit")
    if not source_commit:
        raise ValueError("This snapshot predates promotion support; submit against a current goal")
    hardware = hardwareIdentity()
    if hardware != metadata["hardware"]:
        raise ValueError("Promotion hardware differs from the evaluated goal snapshot")

    def command(argv, workspace, stage, timeout=TIMEOUTS["build"]):
        store.addEvent(job["id"], stage, " ".join(argv))
        result = runCmd(argv, workspace, timeout, env={**os.environ, "KERNEL_BENCH_SERVER_JOB": "1", "KERNEL_BENCH_JOB_ID": str(job["id"])})
        store.addArtifact(job["id"], stage + ".log", result["output"].encode())
        if result["exit_code"] != 0:
            raise RuntimeError(f"{stage} failed: {result['output']}")
        return result

    with tempfile.TemporaryDirectory(prefix="kernel-promotion-") as temp_path:
        workspace = Path(temp_path) / "repo"
        command(["git", "clone", "--shared", "--no-checkout", str(PROJECT_ROOT), str(workspace)], PROJECT_ROOT, "checkout")
        command(["git", "checkout", "--detach", source_commit], workspace, "baseline_checkout")
        # Frozen snapshot comparison prevents validating a different native baseline
        # when discovery was initialized from a modified working tree.
        import io
        import zipfile
        archived = store.query("SELECT source_archive, cache_data FROM kernel_bench.goals WHERE id = %s", (job["goal_id"],))[0]
        with zipfile.ZipFile(io.BytesIO(bytes(archived["source_archive"]))) as archive:
            for name in archive.namelist():
                if not (workspace / name).is_file() or (workspace / name).read_bytes() != archive.read(name):
                    raise ValueError("Discovery snapshot differs from its Git commit; deploy committed sources before promoting")

        # Planning needs finite costs for existing kernels and the new candidate.
        # Preserve the operator corpus and supplement it with measured parent
        # configurations, remapped by name into each freshly built registry.
        records_path = Path(config.get("compiler_records", PROJECT_ROOT / "benchmarks/records.bin"))
        if not records_path.is_file():
            raise ValueError("Compiler calibration corpus is missing: " + str(records_path))
        compiler_records = records_path.read_bytes()
        supplemental = store.query("""SELECT DISTINCT ON (m.kernel_id, c.fingerprint)
            m.kernel_id, m.kernel_name, COALESCE(m.job_id = %s AND m.is_candidate, false) AS is_candidate,
            m.median_ms, c.configuration
            FROM kernel_bench.measurements m JOIN kernel_bench.configurations c ON c.id = m.configuration_id
            JOIN kernel_bench.jobs j ON j.id = m.job_id JOIN kernel_bench.goals g ON g.id = j.goal_id
            WHERE j.status = 'completed' AND g.metadata->'hardware' = %s
            ORDER BY m.kernel_id, c.fingerprint, j.completed_at DESC, j.id DESC""",
                                   (job.get("parent_job_id"), Jsonb(metadata["hardware"])))
        store.addArtifact(job["id"], "compiler_records.bin.gz", gzip.compress(compiler_records, compresslevel=1), "application/gzip")
        store.addArtifact(job["id"], "compiler_measurements.json", json.dumps(supplemental).encode(), "application/json")
        store.addEvent(job["id"], "compiler_calibration", "Corpus SHA256: " + hashlib.sha256(compiler_records).hexdigest())

        def measure(label, candidate_run=False, plan_only=False):
            command([getPythonExe(), "build.py", "--targets", "bindings", "--opencl", "0", "--log-level", "WARNING"], workspace, label + "_build")
            target_records = workspace / "benchmarks/records.bin"
            target_records.parent.mkdir(exist_ok=True)
            target_records.write_bytes(compiler_records)
            measurements = [row for row in supplemental if candidate_run or not row["is_candidate"]]
            if measurements:
                measurement_path = Path(temp_path) / (label + "_measurements.json")
                measurement_path.write_text(json.dumps(measurements))
                command([getPythonExe(), "kernel_bench/cost_records.py", "--records", str(target_records),
                         "--measurements", str(measurement_path)], workspace, label + "_calibration")
            report_path, cache_path = Path(temp_path) / (label + ".json"), Path(temp_path) / (label + ".dirty-cache")
            recipe_command = config["planning_command"] if plan_only else config["command"]
            argv = [arg.replace("{python}", getPythonExe()).replace("{report}", str(report_path)).replace("{cache}", str(cache_path))
                    for arg in recipe_command]
            command(argv, workspace, label + "_goal", config["timeout_seconds"])
            report = json.loads(report_path.read_text())
            if plan_only:
                if report.get("plan_only") is not True or not report.get("buckets") or not report.get("logical_graph"):
                    raise ValueError("Planning runner returned no valid plan")
                if any(not isinstance(bucket.get("kernel_ids"), list) or
                       not isinstance(bucket.get("weight"), (int, float)) or
                       not math.isfinite(bucket["weight"]) or bucket["weight"] < 0 for bucket in report["buckets"]):
                    raise ValueError("Planning runner returned invalid bucket metadata")
                if sum(bucket["weight"] for bucket in report["buckets"]) <= 0:
                    raise ValueError("Planning runner requires a positive-weight bucket")
            else:
                report = validateReport(report)
            store.addArtifact(job["id"], label + ".json", json.dumps(report, allow_nan=False).encode(), "application/json")
            store.addArtifact(job["id"], label + ".dirty-cache", cache_path.read_bytes(), "application/octet-stream")
            return report

        # Integrate against the currently deployed main revision, even when the
        # winning microbenchmark came from an older immutable discovery snapshot.
        current = command(["git", "rev-parse", "HEAD"], PROJECT_ROOT, "current_base")["stdout"].strip()
        if not current:
            raise ValueError("Could not identify the current integration base commit")
        if current != source_commit:
            command(["git", "checkout", "--detach", current], workspace, "current_checkout")
            source_commit = current
        source_hash = hashlib.sha256(job["source"].encode()).hexdigest()
        candidate_path = workspace / "tensor_graphs_cpp/kernels/generated/promoted" / (source_hash + (".cu" if job["backend"] == "cuda" else ".hpp"))
        if not job["result"].get("baseline_only") and candidate_path.is_file() and candidate_path.read_text() == job["source"]:
            store.completeJob(job["id"], {"recipe": config, "measurements": [], "source_commit": source_commit,
                                          "outcome": "already_integrated", "accepted": False})
            return
        if config.get("planning_command") and not job["result"].get("baseline_only"):
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(job["source"])
            plan = measure("candidate_plan", candidate_run=True, plan_only=True)
            discovery_cache = Path(temp_path) / "discovery.dirty-cache"
            discovery_cache.write_bytes(bytes(archived["cache_data"]))
            verifyDiscoveryGoal(plan, discovery_cache)
            uid_map = json.loads((workspace / "tensor_graphs_cpp/generated/kernel_uids.json").read_text())
            plan_ids = {str(uid) for uid, info in uid_map.items() if str(uid).isdigit() and info.get("name") == job["kernel_name"]}
            if not plan_ids:
                plan_ids = {str(uid) for uid, info in uid_map.items() if str(uid).isdigit() and
                            info.get("path") == candidate_path.relative_to(workspace / "tensor_graphs_cpp").as_posix()}
            if len(plan_ids) != 1:
                raise ValueError("Expected exactly one candidate UID in selection preflight")
            selected = sum(str(uid) in plan_ids for bucket in plan["buckets"] if bucket["weight"] > 0 for uid in bucket["kernel_ids"])
            if selected == 0:
                store.completeJob(job["id"], {"recipe": config, "measurements": [], "source_commit": source_commit,
                    "outcome": "unselected", "selected_count": 0, "accepted": False,
                    "selection_reason": "Planner selected no candidate instructions in positive-weight buckets; timing skipped"})
                return
            candidate_path.unlink()
        baseline = measure("baseline")
        discovery_cache = Path(temp_path) / "discovery.dirty-cache"
        discovery_cache.write_bytes(bytes(archived["cache_data"]))
        verifyDiscoveryGoal(baseline, discovery_cache)
        definition = goalDefinition(baseline, hardware)
        definition_id = fingerprint(definition)
        with store.connection() as connection:
            connection.execute("INSERT INTO kernel_bench.goal_definitions(id, definition) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                               (definition_id, Jsonb(definition)))
            prior = connection.execute("SELECT definition_id FROM kernel_bench.goals WHERE id = %s FOR UPDATE", (job["goal_id"],)).fetchone()
            if prior["definition_id"] not in (None, definition_id):
                raise ValueError("Validation recipe changed the goal's logical graph, buckets, weights, or hardware")
            connection.execute("UPDATE kernel_bench.goals SET definition_id = %s WHERE id = %s", (definition_id, job["goal_id"]))
        if job["result"].get("baseline_only"):
            store.completeJob(job["id"], {"recipe": config, "baseline_only": True, "measurements": [],
                                          "definition_id": definition_id, "baseline": {key: value for key, value in baseline.items() if key != "logical_graph"},
                                          "source_commit": source_commit, "outcome": "goal_registered"})
            return
        source_hash = hashlib.sha256(job["source"].encode()).hexdigest()
        candidate_path = workspace / "tensor_graphs_cpp/kernels/generated/promoted" / (source_hash + (".cu" if job["backend"] == "cuda" else ".hpp"))
        if candidate_path.is_file() and candidate_path.read_text() == job["source"]:
            store.completeJob(job["id"], {"recipe": config, "measurements": [], "definition_id": definition_id,
                                          "source_commit": source_commit, "outcome": "already_integrated", "accepted": False})
            return
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(job["source"])
        candidate = measure("candidate", candidate_run=True)
        if fingerprint(goalDefinition(candidate, hardware)) != definition_id:
            raise ValueError("Candidate changed the logical goal definition")
        uid_map = json.loads((workspace / "tensor_graphs_cpp/generated/kernel_uids.json").read_text())
        candidate_ids = {str(uid) for uid, info in uid_map.items() if str(uid).isdigit() and info.get("name") == job["kernel_name"]}
        if not candidate_ids:
            # Generated metadata uses opName in some native interface versions.
            candidate_ids = {str(uid) for uid, info in uid_map.items() if str(uid).isdigit() and info.get("path") == candidate_path.relative_to(workspace / "tensor_graphs_cpp").as_posix()}
        if len(candidate_ids) != 1:
            raise ValueError("Expected exactly one generated candidate kernel UID")
        selected_count = sum(str(uid) in candidate_ids for bucket in candidate["buckets"] if bucket["weight"] > 0 for uid in bucket["kernel_ids"])
        speedup = baseline["weighted_ms"] / candidate["weighted_ms"]
        pairs = [{"baseline_ms": baseline["weighted_ms"], "candidate_ms": candidate["weighted_ms"],
                  "speedup": speedup, "selected_count": selected_count}]
        if selected_count and abs(speedup / config["goal_speedup"] - 1) <= 0.03:
            for round_index in (1, 2):
                paired = {}
                for label in (("candidate", "baseline") if round_index == 1 else ("baseline", "candidate")):
                    if label == "candidate":
                        candidate_path.write_text(job["source"])
                    else:
                        candidate_path.unlink(missing_ok=True)
                    evidence = measure(f"{label}_pair_{round_index}", candidate_run=label == "candidate")
                    if fingerprint(goalDefinition(evidence, hardware)) != definition_id:
                        raise ValueError("Paired measurement changed the logical goal definition")
                    paired[label + "_ms"] = evidence["weighted_ms"]
                    if label == "candidate":
                        paired["selected_count"] = sum(str(uid) in candidate_ids for bucket in evidence["buckets"]
                            if bucket["weight"] > 0 for uid in bucket["kernel_ids"])
                paired["speedup"] = paired["baseline_ms"] / paired["candidate_ms"]
                pairs.append(paired)
            candidate_path.write_text(job["source"])
            speedup = statistics.geometric_mean(pair["speedup"] for pair in pairs)
        lower_bound = speedup
        dispersion = 0.0
        if len(pairs) > 1:
            logs = [math.log(pair["speedup"]) for pair in pairs]
            dispersion = statistics.stdev(logs)
            lower_bound = math.exp(statistics.mean(logs) - 4.303 * dispersion / math.sqrt(len(logs)))
        selection_consistent = all(pair["selected_count"] > 0 for pair in pairs)
        accepted = selection_consistent and lower_bound >= config["goal_speedup"]
        result = {"recipe": config, "measurements": [], "definition_id": definition_id, "source_commit": source_commit,
                  "source_sha256": source_hash, "baseline_ms": statistics.geometric_mean(pair["baseline_ms"] for pair in pairs),
                  "candidate_ms": statistics.geometric_mean(pair["candidate_ms"] for pair in pairs),
                  "speedup": speedup, "selected_count": selected_count, "accepted": accepted,
                  "selection_consistent": selection_consistent,
                  "paired_measurements": pairs, "speedup_lower_95": lower_bound if len(pairs) > 1 else None,
                  "log_speedup_stddev": dispersion, "timing_method": "paired alternating order; t interval for three borderline pairs",
                  "outcome": "validated" if accepted else "unselected" if selected_count == 0 else "no_goal_speedup"}
        # Persist validation before GitHub operations; retries can inspect this evidence.
        store.execute("""INSERT INTO kernel_bench.goal_measurements
            (job_id, definition_id, baseline, candidate, speedup, selected_count, accepted)
            VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT(job_id) DO UPDATE SET
            baseline = EXCLUDED.baseline, candidate = EXCLUDED.candidate, speedup = EXCLUDED.speedup,
            selected_count = EXCLUDED.selected_count, accepted = EXCLUDED.accepted""",
                      (job["id"], definition_id, Jsonb({key: value for key, value in baseline.items() if key != "logical_graph"}),
                       Jsonb({key: value for key, value in candidate.items() if key != "logical_graph"}), speedup, selected_count, accepted))
        store.execute("UPDATE kernel_bench.jobs SET result = %s WHERE id = %s", (Jsonb(result), job["id"]))
        if accepted and config.get("github_repo"):
            branch = "kernel-promotion/" + str(job["id"])
            command(["git", "checkout", "-b", branch], workspace, "pr_branch")
            command(["git", "add", str(candidate_path.relative_to(workspace))], workspace, "pr_add")
            command(["git", "-c", "user.name=KernelBench", "-c", "user.email=kernelbench@localhost", "commit", "-m",
                     "Add validated kernel " + job["kernel_name"]], workspace, "pr_commit")
            remote = "https://github.com/" + config["github_repo"] + ".git"
            published = command(["git", "ls-remote", "--heads", remote, "refs/heads/" + branch], workspace, "pr_remote")
            if published["stdout"].strip():
                command(["git", "fetch", "--no-tags", remote, "refs/heads/" + branch], workspace, "pr_fetch")
                # A crash after push must reuse the published tree, never force-push
                # over a branch that someone has subsequently modified.
                command(["git", "diff", "--quiet", "HEAD", "FETCH_HEAD"], workspace, "pr_verify")
            else:
                command(["git", "push", remote, "HEAD:refs/heads/" + branch], workspace, "pr_push")
            body_path = Path(temp_path) / "pr.md"
            body_path.write_text(f"Adds `{job['kernel_name']}` after kernel correctness and full goal validation.\n\n"
                                 f"Weighted execution: {result['baseline_ms']:.4f} → {result['candidate_ms']:.4f} ms "
                                 f"({speedup:.3f}×). Selected {selected_count} instructions in weighted buckets.\n\n"
                                 f"Goal definition: `{definition_id}`. Base commit: `{source_commit}`.\n"
                                 f"Validation job: `{job['id']}`; parent: `{job['parent_job_id']}`.\n\n"
                                 f"[Validation logs](https://bench.endeavoringorb.com/api/jobs/{job['id']}/logs) · "
                                 f"[Goal progress](https://bench.endeavoringorb.com/progress?goal_id={job['goal_id']})\n\n"
                                 "Measurements apply to this pinned base revision; revalidate if the PR base changes.\n")
            existing = command(["gh", "pr", "list", "--repo", config["github_repo"], "--head", branch,
                                "--state", "all", "--json", "url"], workspace, "pr_lookup")
            rows = json.loads(existing["stdout"])
            if rows:
                result["pr_url"] = rows[0]["url"]
            else:
                response = command(["gh", "pr", "create", "--repo", config["github_repo"], "--base", "main", "--head", branch,
                                    "--title", "Add validated kernel " + job["kernel_name"], "--body-file", str(body_path)], workspace, "pr_create")
                result["pr_url"] = response["stdout"].strip()
            result["outcome"] = "pr_created"
        store.completeJob(job["id"], result)


def installPromotion(app, store, goal_id):
    currentGoalId = goal_id if callable(goal_id) else lambda: goal_id
    @app.get("/api/goals")
    def logicalGoals():
        rows = store.query("""SELECT d.id, d.created_at,
            d.definition->'hardware_system' AS hardware_system,
            d.definition->'buckets' AS buckets, d.definition->'bucket_weights' AS bucket_weights,
            array_agg(g.id ORDER BY g.id) AS snapshot_ids
            FROM kernel_bench.goal_definitions d JOIN kernel_bench.goals g ON g.definition_id = d.id
            GROUP BY d.id ORDER BY d.created_at DESC""")
        return jsonify({"goals": rows})

    @app.get("/api/goals/<definition_id>")
    def logicalGoal(definition_id):
        rows = store.query("SELECT id, definition, created_at FROM kernel_bench.goal_definitions WHERE id = %s", (definition_id,))
        if not rows:
            return jsonify({"error": "Unknown logical goal"}), 404
        return jsonify(rows[0])

    @app.get("/progress")
    def progressPage():
        return render_template("progress.html")

    @app.get("/api/goals/<int:snapshot_id>/workloads")
    def executedWorkloads(snapshot_id):
        rows = store.query("""SELECT j.id AS job_id, COALESCE(gm.baseline, j.result->'baseline') AS baseline
            FROM kernel_bench.jobs j LEFT JOIN kernel_bench.goal_measurements gm ON gm.job_id = j.id
            WHERE j.goal_id = %s AND j.status = 'completed' AND j.mode = 'promote'
            AND COALESCE(gm.baseline, j.result->'baseline') IS NOT NULL
            ORDER BY j.completed_at DESC LIMIT 1""", (snapshot_id,))
        if not rows or not rows[0]["baseline"].get("buckets") or any(
                "executed_workloads" not in bucket for bucket in rows[0]["baseline"].get("buckets", [])):
            return jsonify({"status": "unavailable", "reason": "Run goal validation to capture executed workloads",
                            "goal_id": snapshot_id, "buckets": []})
        buckets = []
        for bucket in rows[0]["baseline"].get("buckets", []):
            groups = {}
            for instruction in bucket.get("executed_workloads", []):
                signature = {key: instruction[key] for key in ("kernel_id", "kernel_name", "engines")}
                signature["inputs"] = [{key: view[key] for key in ("shape", "strides", "dtype")} for view in instruction["inputs"]]
                signature["output"] = {key: instruction["output"][key] for key in ("shape", "strides", "dtype")}
                signature["input_memory_spaces"] = [{key: memory[key] for key in ("memSpaceIdx", "memSpaceType")}
                                                     for memory in instruction["input_memory"]]
                signature["output_memory_space"] = {key: instruction["output_memory"][key] for key in ("memSpaceIdx", "memSpaceType")}
                key = fingerprint(signature)
                group = groups.setdefault(key, {**signature, "execution_count": 0, "estimated_ms": 0.0,
                                                 "estimate_complete": True})
                group["execution_count"] += 1
                if instruction["estimated_ms"] is None:
                    group["estimate_complete"] = False
                else:
                    group["estimated_ms"] += instruction["estimated_ms"]
            buckets.append({"bucket": bucket["bucket"], "weight": bucket["weight"],
                            "workloads": list(groups.values()), "median_ms": bucket["median_ms"]})
        return jsonify({"goal_id": snapshot_id, "job_id": str(rows[0]["job_id"]), "buckets": buckets,
                        "scope": "executed baseline instructions; costs are planner estimates, not measured per-instruction timings"})

    @app.post("/api/goals/validate")
    def validateGoal():
        job_id = enqueueGoalValidation(store, currentGoalId())
        if job_id is None:
            return jsonify({"error": "Promotion is not configured for this snapshot"}), 409
        return jsonify({"job_id": job_id}), 202

    @app.post("/api/jobs/<uuid:job_id>/retry-promotion")
    def retryPromotion(job_id):
        rows = store.query("""UPDATE kernel_bench.jobs SET status = 'queued', step = 'queued',
            error = NULL, error_details = NULL, started_at = NULL, completed_at = NULL
            WHERE id = %s AND goal_id = %s AND mode = 'promote' AND status = 'failed' RETURNING id""", (job_id, currentGoalId()))
        if not rows:
            return jsonify({"error": "Expected a failed promotion for this goal"}), 409
        return jsonify({"job_id": str(rows[0]["id"])}), 202

    @app.post("/api/jobs/<uuid:job_id>/promote")
    def promote(job_id):
        parent = store.getJob(job_id)
        if parent is None or parent["goal_id"] != currentGoalId():
            return jsonify({"error": "Unknown job for this goal"}), 404
        promotion_id = enqueuePromotion(store, parent)
        if promotion_id is None:
            eligibility = promotionEligibility(parent, store.getGoal(parent["goal_id"])["metadata"].get("promotion_recipe"))
            return jsonify({"error": "Promotion is ineligible", "promotion_eligibility": eligibility}), 409
        return jsonify({"job_id": promotion_id}), 202

    @app.get("/api/progress")
    def progressData():
        selected_goal = int(request.args["goal_id"]) if "goal_id" in request.args else currentGoalId()
        goal = store.query("SELECT id, name, definition_id FROM kernel_bench.goals WHERE id = %s", (selected_goal,))
        if not goal:
            return jsonify({"error": "Unknown goal"}), 404
        definition_id = goal[0]["definition_id"]
        scope = "(g.definition_id = %s OR g.id = %s)" if definition_id else "g.id = %s"
        params = (definition_id, selected_goal) if definition_id else (selected_goal,)
        history = store.query("""SELECT j.id AS job_id, j.target_id, j.kernel_name, j.completed_at,
            m.configuration_id, c.fingerprint AS configuration_fingerprint, c.configuration, m.is_candidate, m.median_ms, m.samples_ms
            FROM kernel_bench.measurements m JOIN kernel_bench.jobs j ON j.id = m.job_id
            JOIN kernel_bench.goals g ON g.id = j.goal_id JOIN kernel_bench.configurations c ON c.id = m.configuration_id
            WHERE j.status = 'completed' AND """ + scope + " ORDER BY j.completed_at, j.id, m.configuration_id, m.kernel_id", params)
        baselines = store.query("""SELECT c.target_id, c.id AS configuration_id,
            c.fingerprint AS configuration_fingerprint, c.configuration, b.measurements, b.measured_at
            FROM kernel_bench.baselines b JOIN kernel_bench.configurations c ON c.id = b.configuration_id
            JOIN kernel_bench.goals g ON g.id = c.goal_id
            WHERE b.status = 'measured' AND """ + scope + " ORDER BY b.measured_at, c.id", params)
        promotions = store.query("""SELECT j.id AS job_id, j.kernel_name, j.target_id, j.status, j.error, j.result, j.completed_at,
            gm.baseline - 'buckets' AS baseline, gm.candidate - 'buckets' AS candidate,
            gm.speedup, gm.selected_count, gm.accepted
            FROM kernel_bench.jobs j JOIN kernel_bench.goals g ON g.id = j.goal_id
            LEFT JOIN kernel_bench.goal_measurements gm ON gm.job_id = j.id
            WHERE j.mode = 'promote' AND """ + scope + " ORDER BY j.created_at, j.id", params)
        definitions = store.query("SELECT definition FROM kernel_bench.goal_definitions WHERE id = %s", (definition_id,)) if definition_id else []
        return jsonify({"goal": goal[0], "definition": definitions[0]["definition"] if definitions else None,
                        "targets": store.getTargets(selected_goal), "kernel_history": history, "baselines": baselines, "promotions": promotions,
                        "promotion_enabled": store.getGoal(selected_goal)["metadata"].get("promotion_recipe") is not None})
