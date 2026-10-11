"""Integration checks against PostgreSQL and real CUDA native bindings.

Run explicitly outside the sandbox. Test data belongs to a unique goal and is
removed afterward; existing goals and measurements are never modified.
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
from dotenv import load_dotenv

from kernel_bench.app import createApp
from kernel_bench.jobs import (
    PROJECT_ROOT,
    JobFailure,
    createJob,
    evaluateJob,
    initializeGoal,
    sourceArchive,
)
from kernel_bench.storage import Store


@pytest.fixture(scope="module")
def databaseGoal(tmp_path_factory):
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get("KERNEL_BENCH_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set KERNEL_BENCH_TEST_DATABASE_URL to run PostgreSQL/CUDA integration tests")
    store = Store(database_url)
    store.initialize()
    workspace = tmp_path_factory.mktemp("kernel_bench_integration")
    cache_path = workspace / "goal.bin"
    executable = PROJECT_ROOT / "tensor_graphs_cpp/test_kernel_bench"
    subprocess.run([str(executable), str(cache_path)], check=True, capture_output=True, timeout=120)
    goal_id = initializeGoal(store, cache_path)
    app = createApp(database_url=database_url, token="integration-test-token", goal_id=goal_id, start_worker=False)
    app.testing = True
    yield store, goal_id, app, workspace
    with store.connection() as connection:
        connection.execute("DELETE FROM kernel_bench.events WHERE job_id IN (SELECT id FROM kernel_bench.jobs WHERE goal_id = %s)", (goal_id,))
        connection.execute("DELETE FROM kernel_bench.measurements WHERE job_id IN (SELECT id FROM kernel_bench.jobs WHERE goal_id = %s)", (goal_id,))
        connection.execute("DELETE FROM kernel_bench.jobs WHERE goal_id = %s", (goal_id,))
        connection.execute("DELETE FROM kernel_bench.configurations WHERE goal_id = %s", (goal_id,))
        connection.execute("DELETE FROM kernel_bench.goals WHERE id = %s", (goal_id,))


@pytest.fixture
def goal(databaseGoal):
    store, goal_id, _app, _workspace = databaseGoal
    yield databaseGoal
    with store.connection() as connection:
        connection.execute("DELETE FROM kernel_bench.events WHERE job_id IN (SELECT id FROM kernel_bench.jobs WHERE goal_id = %s)", (goal_id,))
        connection.execute("DELETE FROM kernel_bench.measurements WHERE job_id IN (SELECT id FROM kernel_bench.jobs WHERE goal_id = %s)", (goal_id,))
        connection.execute("DELETE FROM kernel_bench.jobs WHERE goal_id = %s", (goal_id,))


def addTarget(store, goal_id):
    return next(target for target in store.getTargets(goal_id) if target["reference_expression"] == "ADD($0,$1,)")["id"]


def candidateSource(name="KernelBench_Integration_Add", wrong=False, zero_match=False):
    source = (PROJECT_ROOT / "tensor_graphs_cpp/kernels/fused/add/F32_ND_CUDA.cu").read_text()
    source = source.replace("add_f32_nd_kernel", "kernel_bench_integration_add")
    source = source.replace("matchAddF32_CUDA_ND", "matchKernelBenchIntegrationAdd")
    source = source.replace("runAddF32_CUDA_ND", "runKernelBenchIntegrationAdd")
    source = source.replace('"Add_F32_ND_CUDA"', f'"{name}"')
    if wrong:
        source = source.replace("A[idx] + B[idx]", "A[idx] - B[idx]")
    if zero_match:
        source = source.replace("if (output.dtype", "if (countElements(output.getShape()) < 1024) return false;\n    if (output.dtype")
    return source


def testDiscoveryAndAuthenticatedApi(goal):
    store, goal_id, app, _workspace = goal
    assert sourceArchive() == sourceArchive()
    client = app.test_client()
    assert client.get("/api/targets").status_code == 401
    headers = {"X-Bench-Token": "integration-test-token"}
    target_id = addTarget(store, goal_id)
    response = client.get(f"/api/targets/{target_id}", headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["configurations"]
    assert data["configurations"][0]["configuration"]["output_shape"] == [8]
    assert any(row["configuration"]["input_constants"][1] for row in data["configurations"])
    assert any(kernel["name"] == "Add_F32_ND_CUDA" for kernel in data["kernels"])
    path = "tensor_graphs_cpp/kernels/fused/add/ref.hpp"
    assert "refFactoryAdd" in client.get("/api/source", query_string={"path": path}, headers=headers).get_json()["content"]
    assert client.get("/api/source", query_string={"path": "../../.env"}, headers=headers).status_code == 404
    assert client.post("/api/iteration/submit", json=[], headers=headers).status_code == 400
    assert client.post("/api/iteration/submit", json={"source": 123}, headers=headers).status_code == 400
    assert client.post("/api/iteration/submit", json={"target_id": target_id, "source": candidateSource() + candidateSource()}, headers=headers).status_code == 400


def testConcurrentSubmissionsHaveNoTargetReservation(goal):
    store, goal_id, _app, _workspace = goal
    target_id = addTarget(store, goal_id)
    with ThreadPoolExecutor(max_workers=4) as executor:
        ids = list(executor.map(lambda _: createJob(store, goal_id, candidateSource(), target_id), range(8)))
    assert len(set(ids)) == 8
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = list(executor.map(lambda _: store.nextJob(), range(8)))
    assert {str(job["id"]) for job in jobs} == set(ids)
    assert all(job["status"] == "running" for job in jobs)


def testRealCandidateCorrectnessBenchmarksAndPersistence(goal):
    store, goal_id, _app, workspace = goal
    job_id = createJob(store, goal_id, candidateSource(), addTarget(store, goal_id))
    job = store.nextJob()
    assert str(job["id"]) == job_id
    evaluateJob(store, job, workspace / "build")
    persisted = Store(store.database_url).getJob(job_id)
    assert persisted["status"] == "completed"
    result = persisted["result"]
    assert result["tested_configurations"] > 0
    assert result["tested_configurations"] == result["total_configurations"]
    assert result["comparisons"][0]["best_existing_kernel"]
    assert result["comparisons"][0]["speedup"] > 0
    rows = store.query("SELECT * FROM kernel_bench.measurements WHERE job_id = %s", (job_id,))
    assert len(rows) == len(result["measurements"])
    assert any(row["is_candidate"] for row in rows)
    assert all(len(row["samples_ms"]) == 16 for row in rows)
    assert not (workspace / "build/tensor_graphs_cpp/kernels/generated/kernel_bench/candidate.cu").exists()


@pytest.mark.parametrize("wrong,zero_match,error", [(True, False, "correctness test failed"), (False, True, "matches zero")])
def testInvalidCandidatesFailWithoutMeasurements(goal, wrong, zero_match, error):
    store, goal_id, _app, workspace = goal
    job_id = createJob(store, goal_id, candidateSource(wrong=wrong, zero_match=zero_match), addTarget(store, goal_id))
    job = store.nextJob()
    try:
        evaluateJob(store, job, workspace / "build")
    except JobFailure as exception:
        assert error in str(exception)
        store.failJob(job_id, str(exception), exception.details, exception.result)
        persisted = Store(store.database_url).getJob(job_id)
        assert persisted["error_details"]["artifact_ids"]
        if wrong:
            details = persisted["error_details"]
            assert details["kind"] == "correctness_mismatch"
            assert details["configuration_id"]
            assert details["reference"] != details["actual"]
            assert details["tolerance"]["absolute"] == pytest.approx(1e-4)
            assert details["reproduction"]["seed"] == 42
            assert persisted["result"]["correctness_results"][-1]["passed"] is False
        else:
            assert persisted["error_details"]["kind"] == "no_matching_configurations"
    else:
        logs = store.query("SELECT message FROM kernel_bench.events WHERE job_id = %s AND step = 'test_bench' ORDER BY id", (job_id,))
        pytest.fail(f"Invalid candidate accepted; source changed correctly: {'A[idx] - B[idx]' in job['source']}; logs: {logs}")
    assert not store.query("SELECT * FROM kernel_bench.measurements WHERE job_id = %s", (job_id,))
    assert not (workspace / "build/tensor_graphs_cpp/kernels/generated/kernel_bench/candidate.cu").exists()


@pytest.mark.parametrize("kind", ["lint_error", "compilation_error", "registration_error"])
def testActionableFailuresAndDurableArtifacts(goal, kind):
    store, goal_id, app, workspace = goal
    source = candidateSource()
    if kind == "lint_error":
        source = source.replace("if (output.dtype", "if (inputs.size() != 2) return false;\n    if (output.dtype")
    elif kind == "compilation_error":
        source += "\n#error KernelBench diagnostic compile error\n"
    else:
        source = source.replace("#ifdef TG_USE_CUDA", "#if defined(TG_BENCH_DISABLED)")
    job_id = createJob(store, goal_id, source, addTarget(store, goal_id))
    job = store.nextJob()
    with pytest.raises(JobFailure) as exception:
        evaluateJob(store, job, workspace / "build")
    failure = exception.value
    assert failure.details["kind"] == kind
    store.failJob(job_id, str(failure), failure.details, failure.result)
    store.addEvent(job_id, "error", str(failure))
    client = app.test_client()
    headers = {"X-Bench-Token": "integration-test-token"}
    base = f"/api/jobs/{job_id}"
    persisted = client.get(base, headers=headers).get_json()
    assert persisted["error_details"]["kind"] == kind
    assert persisted["error"] == str(failure)
    artifacts = client.get(base + "/artifacts", headers=headers).get_json()["artifacts"]
    by_name = {row["name"]: row for row in artifacts}
    assert set(failure.details["artifact_ids"]) == {row["artifact_id"] for row in artifacts}
    assert client.get(by_name["submitted_source"]["url"], headers=headers).data.decode() == source
    candidate_name = "tensor_graphs_cpp/kernels/generated/kernel_bench/candidate.cu"
    assert client.get(by_name[candidate_name]["url"], headers=headers).data.decode() == source
    if kind == "lint_error":
        assert "build_diagnostics/build_metadata.json" not in by_name
        assert persisted["error_details"]["stage"] == "lint"
    else:
        assert "build_diagnostics/build_metadata.json" in by_name
    if kind == "compilation_error":
        stderr_artifacts = [row for row in artifacts if row["name"].startswith("build_diagnostics/compiler-") and row["name"].endswith(".stderr")]
        assert any(b"KernelBench diagnostic compile error" in client.get(row["url"], headers=headers).data for row in stderr_artifacts)
        assert "KernelBench diagnostic compile error" in persisted["error_details"]["diagnostic"]
    elif kind == "lint_error":
        assert "Input Count Check" in persisted["error_details"]["diagnostic"]
        assert "Input Count Check" in persisted["error"]
    else:
        assert persisted["error_details"]["observed_match_count"] == 0
        assert "TG_USE_CUDA" in persisted["error_details"]["enabled_capabilities"]
    latest = client.get(base + "/logs?tail=1&stage=error", headers=headers).get_json()
    cursor = latest["next_after_id"]
    assert len(latest["events"]) == 1
    assert not client.get(base + f"/logs?after_id={cursor}", headers=headers).get_json()["events"]
    store.addEvent(job_id, "error", "literal %_ diagnostic")
    assert len(client.get(base + f"/logs?after_id={cursor}&grep=%25_", headers=headers).get_json()["events"]) == 1
    other_id = createJob(store, goal_id, candidateSource(), addTarget(store, goal_id))
    artifact_id = by_name["submitted_source"]["artifact_id"]
    assert client.get(f"/api/jobs/{other_id}/artifacts/{artifact_id}", headers=headers).status_code == 404


@pytest.mark.parametrize("mode", ["lint", "compile"])
def testValidationModes(goal, mode):
    store, goal_id, app, workspace = goal
    target_id = addTarget(store, goal_id)
    source = candidateSource("KernelBench_Validation_" + mode)
    client = app.test_client()
    response = client.post("/api/iteration/lint" if mode == "lint" else "/api/iteration/submit",
                           headers={"X-Bench-Token": "integration-test-token"},
                           json={"target_id": target_id, "source": source, "backend": "cuda", "mode": mode})
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    job = store.getJob(job_id)
    assert job["mode"] == mode
    evaluateJob(store, job, workspace / ("validation_" + mode))
    completed = store.getJob(job_id)
    assert completed["status"] == "completed"
    assert completed["result"]["mode"] == mode
    assert completed["result"]["measurements"] == []
    assert "correctness" not in completed["result"]["passed_stages"]
    assert not store.query("SELECT * FROM kernel_bench.measurements WHERE job_id = %s", (job_id,))


def testPhaseThreeContractAndSkeletons(goal):
    from kernel_bench.onboarding import SKELETONS
    store, goal_id, app, workspace = goal
    client = app.test_client()
    headers = {'X-Bench-Token': 'integration-test-token'}
    contract = client.get('/api/kernel-contract', headers=headers).get_json()
    assert 'TG_USE_CUDA' in contract['enabled_capabilities']
    assert contract['run'] == 'void run(const KernelContext &ctx)'
    import hashlib
    from psycopg.types.json import Jsonb
    from kernel_bench.storage import fingerprint
    expression = 'COPY_TO($0,)'
    copy_target = hashlib.sha256(expression.encode()).hexdigest()
    configuration = {'input_shapes': [[8]], 'input_strides': [[1]], 'input_dtypes': [0],
                     'input_constants': [[]], 'output_shape': [8], 'output_strides': [1], 'output_dtype': 0}
    with store.connection() as connection:
        connection.execute("INSERT INTO kernel_bench.targets(id, expression) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                           (copy_target, expression))
        connection.execute("""INSERT INTO kernel_bench.configurations
            (goal_id, target_id, fingerprint, configuration, matching_kernels)
            VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
            (goal_id, copy_target, fingerprint(configuration), Jsonb(configuration), Jsonb({})))
    for backend, source in SKELETONS.items():
        job_id = createJob(store, goal_id, source, copy_target, backend=backend, mode='compile')
        evaluateJob(store, store.getJob(job_id), workspace / ('skeleton_' + backend))
        result = store.getJob(job_id)['result']
        assert 'registration' in result['passed_stages']
