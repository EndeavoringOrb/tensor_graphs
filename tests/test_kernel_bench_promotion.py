"""Promotion gates, stable workload identity, shared scheduling, and portable lint."""
import copy
import io
import json
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_bench.onboarding import starterFiles
from kernel_bench.promotion import enqueuePromotion, goalDefinition, promoteJob, promotionConfig, validateReport
from kernel_bench.storage import Store, fingerprint
from kernel_linter import lintCandidate


def report():
    return {"correctness_passed": True, "logical_graph": {"root": 1, "nodes": {"1": {"op": "ADD"}}},
            "buckets": [{"bucket": {"inputDirtyRegions": {"0": []}}, "weight": 0,
                         "kernel_ids": ["1"], "samples_ms": [10] * 8},
                        {"bucket": {"inputDirtyRegions": {"1": []}}, "weight": 3,
                         "kernel_ids": ["2"], "samples_ms": [2] * 8}]}


def testWeightedGoalAndStableIdentity():
    baseline = validateReport(report())
    assert baseline["weighted_ms"] == 2
    zero_skipped = report()
    zero_skipped["buckets"][0]["samples_ms"] = []
    assert validateReport(zero_skipped)["weighted_ms"] == 2
    candidate = copy.deepcopy(baseline)
    candidate["buckets"][1]["kernel_ids"] = ["new-kernel"]
    candidate["buckets"][1]["samples_ms"] = [1] * 8
    validateReport(candidate)
    assert fingerprint(goalDefinition(baseline, {"gpu": "box"})) == fingerprint(goalDefinition(candidate, {"gpu": "box"}))
    for mutation in [lambda r: r["logical_graph"].update(root=2),
                     lambda r: r["buckets"][0].update(weight=1),
                     lambda r: r["buckets"][0]["bucket"].update(outputNeededRegion=[1])]:
        altered = copy.deepcopy(baseline)
        mutation(altered)
        assert fingerprint(goalDefinition(altered, {"gpu": "box"})) != fingerprint(goalDefinition(baseline, {"gpu": "box"}))
    assert fingerprint(goalDefinition(baseline, {"gpu": "other"})) != fingerprint(goalDefinition(baseline, {"gpu": "box"}))


@pytest.mark.parametrize("mutation", [lambda r: r.update(correctness_passed=False),
    lambda r: r.update(logical_graph={}), lambda r: r.update(buckets=[]),
    lambda r: r["buckets"][1].update(weight=float("nan")),
    lambda r: r["buckets"][1].update(weight=0),
    lambda r: r["buckets"][1].update(samples_ms=[0] * 8),
    lambda r: r["buckets"][1].update(samples_ms=[1] * 7),
    lambda r: r["buckets"][1].update(samples_ms=[float("inf")] * 8)])
def testInvalidEvidenceFailsLoud(mutation):
    evidence = report()
    mutation(evidence)
    with pytest.raises(ValueError):
        validateReport(evidence)


def testFrozenPromotionRecipeAndGate():
    parent = {"id": "parent", "goal_id": 1, "target_id": "target", "mode": "evaluate", "status": "completed",
              "idea": "idea", "source": "source", "kernel_name": "candidate", "backend": "cpu", "result": {"comparisons": []}}
    calls = []
    store = SimpleNamespace(getJob=lambda _: parent,
                            getGoal=lambda _: {"metadata": {"promotion_recipe": {"kernel_speedup": 1.02}}},
                            query=lambda sql, args: calls.append((sql, args)) or [{"id": "promotion"}])
    assert enqueuePromotion(store, parent) is None
    parent["result"]["comparisons"] = [{"speedup": None}, {"speedup": 1.01}]
    assert enqueuePromotion(store, parent) is None
    parent["result"]["comparisons"].append({"speedup": 1.1})
    assert enqueuePromotion(store, parent) == "promotion"
    assert "ON CONFLICT (parent_job_id)" in calls[0][0]
    assert calls[0][1][-1].obj == {"recipe": {"kernel_speedup": 1.02}}
    parent["status"] = "failed"
    with pytest.raises(ValueError):
        enqueuePromotion(store, parent)


def testOneQueueForEveryMode():
    queries = []
    store = Store("test")
    store.query = lambda sql, args: queries.append((sql, args)) or [{"mode": "promote"}]
    assert store.nextJob()["mode"] == "promote"
    assert "mode <>" not in queries[0][0]
    assert "ORDER BY created_at, id" in queries[0][0]


def testOperatorRecipeValidation(monkeypatch, tmp_path):
    path = tmp_path / "recipe.json"
    monkeypatch.setenv("KERNEL_BENCH_PROMOTION_CONFIG", str(path))
    for config in [{}, {"command": "echo"}, {"command": ["echo", "{report}"]},
                   {"command": ["{cache}", "{report}"], "goal_speedup": 1},
                   {"command": ["{cache}", "{report}"], "timeout_seconds": True}]:
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError):
            promotionConfig()
    path.write_text(json.dumps({"command": ["{python}", "runner.py", "{report}", "{cache}"]}))
    assert promotionConfig()["goal_speedup"] == 1.01


def testPackagedLinterMatchesBuildAndServer(tmp_path):
    import build
    import kernel_linter
    assert build.KernelLinter is kernel_linter.KernelLinter
    assert build.lintCandidate is lintCandidate
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        for name in ("kernel_linter.py", "kernel_linter_rules.py"):
            archive.write(name)
    with zipfile.ZipFile(io.BytesIO(archive_bytes.getvalue())) as archive:
        for name, content in starterFiles(archive).items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    candidate = tmp_path / "candidate.hpp"
    candidate.write_text('bool matchCopy(const std::vector<TensorNode>& inputs, const TensorNode& output) { return inputs.size() == 1; }\n'
                         'REGISTER_KERNEL("Candidate", 1, 1, matchCopy, runCopy, refCopy);\n')
    with pytest.raises(SystemExit):
        lintCandidate(candidate)

    packaged = subprocess.run([sys.executable, "kernel_linter.py", str(candidate)], cwd=tmp_path, capture_output=True, text=True)
    assert packaged.returncode == 1
    assert "input_count_check" in packaged.stdout
    candidate.write_text('bool matchCopy(const std::vector<TensorNode>& inputs, const TensorNode& output) { return true; }\n'
                         'REGISTER_KERNEL("Candidate", 1, 1, matchCopy, runCopy, refCopy);\n')
    lintCandidate(candidate)
    assert subprocess.run([sys.executable, "kernel_linter.py", str(candidate)], cwd=tmp_path, capture_output=True).returncode == 0
    candidate.write_text("// REGISTER_KERNEL fake\n")
    with pytest.raises(SystemExit):
        lintCandidate(candidate)


@pytest.mark.parametrize("selected,candidate_ms,github,outcome", [(True, 1, False, "validated"), (False, 1, False, "unselected"), (True, 3, False, "no_goal_speedup"), (True, 1, True, "pr_created")])
def testFullPromotionStages(monkeypatch, selected, candidate_ms, github, outcome):
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w"):
        pass
    commands = []
    class Connection:
        def execute(self, *args):
            return self
        def fetchone(self):
            return {"definition_id": None}
    class FakeStore:
        def getGoal(self, goal_id):
            return {"metadata": {"source_commit": "commit", "hardware": {"gpu": "box"}}}
        def query(self, *args):
            return [{"source_archive": archive_bytes.getvalue(), "cache_data": b"cache"}]
        @contextmanager
        def connection(self):
            yield Connection()
        def execute(self, *args):
            pass
        def addEvent(self, *args):
            pass
        def addArtifact(self, *args):
            pass
        def completeJob(self, job_id, result):
            self.result = result
    def run(argv, cwd, timeout, **kwargs):
        commands.append(argv)
        if argv[:2] == ["git", "clone"]:
            Path(argv[-1]).mkdir()
        elif "build.py" in argv:
            generated = cwd / "tensor_graphs_cpp/generated"
            generated.mkdir(parents=True, exist_ok=True)
            (generated / "kernel_uids.json").write_text(json.dumps({"99": {"name": "candidate"}, "0x63": {"name": "candidate"}}))
        elif "runner.py" in argv:
            evidence = report()
            if "candidate" in argv[-2]:
                evidence["buckets"][1]["samples_ms"] = [candidate_ms] * 8
                evidence["buckets"][1]["kernel_ids"] = ["99" if selected else "2"]
            Path(argv[-2]).write_text(json.dumps(evidence))
            Path(argv[-1]).write_bytes(b"cache")
        stdout = "commit" if argv[:2] == ["git", "rev-parse"] else ""
        if argv[:3] == ["gh", "pr", "list"]:
            stdout = "[]"
        if argv[:3] == ["gh", "pr", "create"]:
            stdout = "https://github.com/owner/repo/pull/1"
        return {"exit_code": 0, "output": "", "stdout": stdout}
    monkeypatch.setattr("kernel_bench.promotion.verifyDiscoveryGoal", lambda *_: None)
    monkeypatch.setattr("kernel_bench.jobs.runCmd", run)
    monkeypatch.setattr("kernel_bench.jobs.hardwareIdentity", lambda: {"gpu": "box"})
    store = FakeStore()
    job = {"id": "job", "goal_id": 1, "parent_job_id": "parent", "source": "source", "backend": "cpu",
           "kernel_name": "candidate", "result": {"recipe": {"command": ["{python}", "runner.py", "{report}", "{cache}"],
                                                                 "timeout_seconds": 60, "goal_speedup": 1.01}}}
    if github:
        job["result"]["recipe"]["github_repo"] = "owner/repo"
    promoteJob(store, job)
    if github:
        assert store.result["pr_url"] == "https://github.com/owner/repo/pull/1"
        assert any(argv[:2] == ["git", "push"] for argv in commands)
        assert all("--force" not in argv for argv in commands)
    assert store.result["outcome"] == outcome
    assert store.result["selected_count"] == int(selected)
    assert store.result["speedup"] == 2 / candidate_ms
    assert sum("build.py" in argv for argv in commands) == 2


def testQueuedGoalPreparationKeepsSharedWorkerLock(monkeypatch):
    from kernel_bench import admin, jobs
    calls = []
    def discover(store, cache_path, **kwargs):
        calls.append((cache_path, kwargs))
        return 17
    monkeypatch.setattr(jobs, "initializeGoal", discover)
    monkeypatch.setattr(admin, "enqueueGoalValidation", lambda store, goal_id: "validation-17")
    store = SimpleNamespace(execute=lambda sql, params: calls.append(params),
                            completeJob=lambda job_id, result: calls.append(result))
    admin.prepareGoal(store, {"id": "setup", "result": {"cache_path": "/goal", "recipe": {"command": ["runner"]}}})
    assert calls[0][1]["already_locked"] is True
    assert calls[-1]["goal_id"] == 17
    assert calls[-1]["validation_job_id"] == "validation-17"


def testDiscoveryGoalRejectsDifferentBuckets(monkeypatch):
    from kernel_bench.promotion import verifyDiscoveryGoal
    from utils import binary
    value = report()
    entries = [{"type": "metadata", "rootId": 1, "bucketWeights": [0, 3]}] + [
        {"type": "compiled_bucket", "graph": {"bucket": bucket["bucket"]}} for bucket in value["buckets"]]
    monkeypatch.setattr(binary, "loadCacheFile", lambda *args, **kwargs: entries)
    verifyDiscoveryGoal(value, "cache")
    value["buckets"][1]["weight"] = 1
    with pytest.raises(ValueError, match="differs"):
        verifyDiscoveryGoal(value, "cache")
