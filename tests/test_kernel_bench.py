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
    except RuntimeError as exception:
        assert error in str(exception)
    else:
        logs = store.query("SELECT message FROM kernel_bench.events WHERE job_id = %s AND step = 'test_bench' ORDER BY id", (job_id,))
        pytest.fail(f"Invalid candidate accepted; source changed correctly: {'A[idx] - B[idx]' in job['source']}; logs: {logs}")
    assert not store.query("SELECT * FROM kernel_bench.measurements WHERE job_id = %s", (job_id,))
    assert not (workspace / "build/tensor_graphs_cpp/kernels/generated/kernel_bench/candidate.cu").exists()
