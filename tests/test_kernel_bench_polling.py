"""Polling contracts and additive PostgreSQL migration checks."""
import os
import json
import threading
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from flask import Flask
from psycopg.types.json import Jsonb

from kernel_bench import native_runner
from kernel_bench.agent import KernelBenchClient
from kernel_bench.app import createApp
from kernel_bench.polling import decodeCursor, encodeCursor, pollJob
from kernel_bench.storage import Store


def testCursor():
    row = {'job_id': uuid.uuid4(), 'created_at': datetime.now(timezone.utc)}
    assert decodeCursor(encodeCursor(row)) == (row['created_at'], row['job_id'])
    for invalid in ['bad', 'W10=', 'WyJhIiwiYiJd', 'WyIyMDI2LTAxLTAxVDAwOjAwOjAwKzAwOjAwIiwgMTIzXQ==', '']:
        with pytest.raises(ValueError):
            decodeCursor(invalid)


def testPromotionPersistenceAndProgress(pollingDatabase, monkeypatch):
    from kernel_bench.promotion import enqueuePromotion
    store, _, goal_id, target_id, configuration_id = pollingDatabase
    recipe = {"command": ["{python}", "runner.py", "{report}", "{cache}"], "kernel_speedup": 1.02}
    store.execute("UPDATE kernel_bench.goals SET metadata = metadata || %s WHERE id = %s",
                  (Jsonb({"promotion_recipe": recipe}), goal_id))
    parent_id = uuid.uuid4()
    store.execute("""INSERT INTO kernel_bench.jobs(id, goal_id, target_id, idea, source, kernel_name, backend)
        VALUES (%s, %s, %s, 'idea', 'source', 'candidate', 'cpu')""", (parent_id, goal_id, target_id))
    store.completeJob(parent_id, {"measurements": [], "comparisons": [{"speedup": 1.2}]})
    parent = store.getJob(parent_id)
    promotion_id = enqueuePromotion(store, parent)
    assert enqueuePromotion(store, parent) == promotion_id
    promotion = store.getJob(promotion_id)
    assert promotion["mode"] == "promote"
    assert promotion["result"]["recipe"] == recipe
    assert store.queueInfo(goal_id, promotion)["position"] >= 1
    store.addEvent(promotion_id, "baseline_goal", "test")
    assert store.getJob(promotion_id)["revision"] > promotion["revision"]
    monkeypatch.setattr("kernel_bench.app.Store", lambda _: store)
    client = createApp(token="test", goal_id=goal_id, start_worker=False).test_client()
    assert client.get("/progress").status_code in (302, 401)
    headers = {"X-Bench-Token": "test"}
    assert client.get("/progress", headers=headers).status_code == 200
    response = client.get("/api/progress", headers=headers)
    assert response.status_code == 200
    assert response.get_json()["promotions"][0]["job_id"] == promotion_id


@pytest.mark.parametrize('url,token', [('', 'token'), ('localhost:8080', 'token'),
                                     ('https://user:pass@example.org', 'token'),
                                     ('https://example.org?token=x', 'token'),
                                     ('https://example.org:bad', 'token'),
                                     ('https://example.org', ''), ('https://example.org', 'bad\nheader')])
def testClientConfiguration(url, token, tmp_path):
    with pytest.raises(ValueError):
        KernelBenchClient(url, token, tmp_path / 'identity')


def testProjectedPolling():
    app = Flask(__name__)
    row = {'id': 'job', 'goal_id': 1, 'revision': 3, 'status': 'running', 'step': 'build', 'error': None}
    with app.test_request_context('/?fields=status,step,error,revision'):
        response = pollJob(None, lambda _: dict(row), 'job', threading.Event())
        assert response.get_json() == {key: row[key] for key in ['status', 'step', 'error', 'revision']}
        etag = response.headers['ETag']
    with app.test_request_context('/?fields=status,step,error,revision', headers={'If-None-Match': etag}):
        assert pollJob(None, lambda _: dict(row), 'job', threading.Event()).status_code == 304
    row['revision'] += 1
    with app.test_request_context('/?fields=status,step,error,revision', headers={'If-None-Match': etag}):
        assert pollJob(None, lambda _: dict(row), 'job', threading.Event()).status_code == 200
    for query in ['fields=bad', 'fields=status,status', 'after_revision=-1', 'wait=26', 'wait=nan', 'wait=1']:
        with app.test_request_context('/?' + query), pytest.raises(ValueError):
            pollJob(None, lambda _: dict(row), 'job', threading.Event())


def testLongPollingWakesOnRevision():
    app = Flask(__name__)
    rows = iter([{'id': 'job', 'revision': 1, 'status': 'running'}, {'id': 'job', 'revision': 2, 'status': 'running'}])
    with app.test_request_context('/?fields=revision&after_revision=1&wait=1'):
        response = pollJob(None, lambda _: next(rows), 'job', threading.Event())
        assert response.get_json() == {'revision': 2}


def testBaselineMethodology(monkeypatch, tmp_path):
    kernels = [{'kernel_id': index, 'name': str(index), 'reference_expression': 'ADD',
                'is_view': False, 'has_reference_factory': True, 'is_reference': index == 0}
               for index in range(3)]
    calls = []
    def benchmark(kernel_id, configuration, iterations):
        calls.append((kernel_id, iterations))
        return [float(kernel_id), float(kernel_id) + 0.5]
    native = SimpleNamespace(kernel_bench_kernels=lambda: kernels,
                             kernel_bench_matches=lambda *args: True,
                             kernel_bench_test_diagnostics=lambda *args: {'passed': True},
                             kernel_bench_benchmark=benchmark, kernel_bench_capabilities=list)
    monkeypatch.setattr(native_runner, 'native', native)
    payload = {'configuration': {'id': 1, 'configuration': {}}, 'reference_expression': 'ADD',
               'output_path': str(tmp_path / 'result.json')}
    result = native_runner.calibrate(payload, {'result': {}})
    assert calls == [(1, 8), (2, 8), (2, 8), (1, 8)]
    assert result['winner']['kernel_id'] == 1
    assert result['winner']['samples_ms'] == [1, 1.5, 1, 1.5]
    native.kernel_bench_test_diagnostics = lambda *args: {'passed': False, 'kind': 'correctness_mismatch'}
    with pytest.raises(native_runner.NativeFailure):
        native_runner.calibrate(payload, {'result': {}})


@pytest.fixture
def pollingDatabase():
    load_dotenv('.env')
    url = os.environ.get('KERNEL_BENCH_TEST_DATABASE_URL')
    if not url:
        pytest.skip('Set KERNEL_BENCH_TEST_DATABASE_URL for PostgreSQL checks')
    store = Store(url)
    store.initialize()
    store.initialize()  # migrations are repeatable
    key = str(uuid.uuid4())
    goal_id = store.query("""INSERT INTO kernel_bench.goals(fingerprint, name, cache_sha256, cache_data, source_archive, metadata)
        VALUES (%s, 'polling test', '', '', '', %s) RETURNING id""", (key, Jsonb({'kernels': [], 'native_interface_version': 1})))[0]['id']
    store.execute('INSERT INTO kernel_bench.targets(id, expression) VALUES (%s, %s)', (key, key))
    configuration_id = store.query("""INSERT INTO kernel_bench.configurations(goal_id, target_id, fingerprint, configuration, matching_kernels)
        VALUES(%s, %s, %s, '{"input_dtypes": [], "input_constants": []}', '{}') RETURNING id""", (goal_id, key, key))[0]['id']
    app = createApp(database_url=url, token='polling-test', goal_id=goal_id, start_worker=False)
    app.testing = True
    yield store, app.test_client(), goal_id, key, configuration_id
    with store.connection() as connection:
        connection.execute('DELETE FROM kernel_bench.events WHERE job_id IN (SELECT id FROM kernel_bench.jobs WHERE goal_id = %s)', (goal_id,))
        connection.execute('DELETE FROM kernel_bench.jobs WHERE goal_id = %s', (goal_id,))
        connection.execute('DELETE FROM kernel_bench.configurations WHERE goal_id = %s', (goal_id,))
        connection.execute('DELETE FROM kernel_bench.targets WHERE id = %s', (key,))
        connection.execute('DELETE FROM kernel_bench.active_goal WHERE goal_id = %s', (goal_id,))
        connection.execute('DELETE FROM kernel_bench.goals WHERE id = %s', (goal_id,))


def testDatabasePollingAndBaselines(pollingDatabase):
    store, client, goal_id, target_id, configuration_id = pollingDatabase
    headers = {'X-Bench-Token': 'polling-test'}
    ids = [uuid.uuid4() for _ in range(3)]
    for job_id in ids:
        store.execute("""INSERT INTO kernel_bench.jobs(id, goal_id, target_id, idea, source, kernel_name, backend, mode, created_at)
            VALUES(%s, %s, %s, '', '', 'test', 'cpu', 'lint', '2026-01-01')""", (job_id, goal_id, target_id))
    data = client.get('/api/jobs?limit=2&mode=lint&status=queued', headers=headers).get_json()
    assert data['has_more'] is True
    next_page = client.get('/api/jobs?limit=2&cursor=' + data['next_cursor'], headers=headers).get_json()
    assert len(next_page['jobs']) == 1
    assert {row['job_id'] for row in data['jobs'] + next_page['jobs']} == {str(job_id) for job_id in ids}
    job_id = str(ids[0])
    path = f'/api/jobs/{job_id}?fields=status,step,error,revision'
    response = client.get(path, headers=headers)
    etag = response.headers['ETag']
    assert client.get(path, headers={**headers, 'If-None-Match': etag}).status_code == 304
    store.execute("UPDATE kernel_bench.jobs SET status='running', started_at=now(), step='build' WHERE id=%s", (job_id,))
    store.addEvent(job_id, 'test', 'testing')
    updated = client.get(path, headers={**headers, 'If-None-Match': etag})
    assert updated.status_code == 200
    assert updated.get_json()['revision'] == 3
    queue = client.get('/api/status', headers=headers).get_json()['queue']
    assert queue['depth_by_mode']['lint']['queued'] >= 2
    assert any(row['stage'] == 'build' for row in queue['recent_stage_durations'])
    full = client.get(f'/api/jobs/{job_id}', headers=headers).get_json()
    assert full['queue']['position'] is None
    detail = client.get('/api/targets/' + target_id, headers=headers).get_json()
    assert detail['calibrated_baselines'][0]['status'] == 'pending'
    store.execute("INSERT INTO kernel_bench.baselines(configuration_id,status,error) VALUES(%s,'failed','no eligible kernel')", (configuration_id,))
    detail = client.get('/api/targets/' + target_id, headers=headers).get_json()
    assert detail['calibrated_baselines'][0]['error'] == 'no eligible kernel'
    assert client.get('/api/status').get_json()['error'].startswith('Authentication required')
    assert client.get('/api/status', headers={'X-Bench-Token': 'wrong'}).get_json()['error'] == 'Invalid X-Bench-Token'


def testCalibrationRecovery(pollingDatabase, monkeypatch, tmp_path):
    from kernel_bench import jobs
    store, client, goal_id, target_id, configuration_id = pollingDatabase
    headers = {'X-Bench-Token': 'polling-test'}
    assert client.post('/api/baselines/calibrate', json={}, headers=headers).status_code == 202
    workspace = tmp_path / ('baseline-' + str(goal_id))
    workspace.mkdir()
    (workspace / '.baseline_built').touch()
    measurement = {'kernel_id': 1, 'kernel_name': 'baseline', 'samples_ms': [1, 2], 'median_ms': 1.5}
    monkeypatch.setattr(jobs, 'hardwareIdentity', lambda: {'machine': 'test'})
    monkeypatch.setattr(jobs, 'runNative', lambda *args: ({'measurements': [measurement], 'winner': measurement, 'enabled_capabilities': []}, ''))
    row = {'id': configuration_id, 'goal_id': goal_id, 'expression': target_id, 'configuration': {}, 'matching_kernels': {'1': 'baseline'}}
    store.execute("UPDATE kernel_bench.baselines SET status='running' WHERE configuration_id=%s", (configuration_id,))
    jobs.calibrateConfiguration(store, row, tmp_path)
    baseline = store.getBaselines(goal_id, target_id)[0]
    assert baseline['status'] == 'measured'
    assert baseline['winner'] == measurement
    assert baseline['identity']['methodology']['rounds'] == 2
    assert baseline['measured_at'] is not None
    assert client.post('/api/baselines/calibrate', json={'retry_failed': True}, headers=headers).status_code == 202
    assert store.getBaselines(goal_id, target_id)[0]['status'] == 'measured'
    def fail(*args):
        raise jobs.JobFailure('native crash', {'kind': 'native_crash'})
    monkeypatch.setattr(jobs, 'runNative', fail)
    jobs.calibrateConfiguration(store, row, tmp_path)
    assert store.getBaselines(goal_id, target_id)[0]['error'] == 'native crash'
    client.post('/api/baselines/calibrate', json={'retry_failed': True}, headers=headers)
    assert store.getBaselines(goal_id, target_id)[0]['status'] == 'pending'


def testNativeBaselineHarness(tmp_path):
    """Exercise calibration against the prebuilt native harness, without modifying it."""
    import subprocess

    from kernel_bench.jobs import PROJECT_ROOT, runNative
    libraries = list(PROJECT_ROOT.glob('kernel_bench_native*.so'))
    executable = PROJECT_ROOT / 'tensor_graphs_cpp/test_kernel_bench'
    if not libraries or not executable.is_file():
        pytest.skip('Requires prebuilt native harness and cache fixture executable')
    cache_path = tmp_path / 'goal.bin'
    subprocess.run([str(executable), str(cache_path)], check=True, capture_output=True, timeout=120)
    (tmp_path / libraries[0].name).symlink_to(libraries[0])
    discovery, _log = runNative(tmp_path, {'action': 'discover', 'cache_path': str(cache_path)}, 120)
    call = next(row for row in discovery['calls'] if row['reference_expression'] == 'ADD($0,$1,)' and row['kernel_matches'])
    keys = ('input_shapes', 'input_strides', 'input_dtypes', 'input_constants', 'output_shape', 'output_strides', 'output_dtype')
    result, _log = runNative(tmp_path, {'action': 'baseline', 'reference_expression': call['reference_expression'],
                                       'configuration': {'id': 1, 'configuration': {key: call[key] for key in keys}}}, 120)
    assert result['measurements']
    assert result['winner']['median_ms'] > 0
    assert result['winner'] == min(result['measurements'], key=lambda row: row['median_ms'])


def testLongPollCapacity():
    from contextlib import ExitStack

    from werkzeug.exceptions import HTTPException

    from kernel_bench.polling import waitSlot
    app = Flask(__name__)
    with app.test_request_context('/'), ExitStack() as stack:
        for _ in range(8):
            stack.enter_context(waitSlot(1))
        with pytest.raises(HTTPException) as error, waitSlot(1):
            pytest.fail('A ninth wait should be rejected')
        assert error.value.response.status_code == 429
        with waitSlot(0):
            pass  # Immediate requests retain capacity.
    with app.test_request_context('/'), waitSlot(1):
        pass  # All slots are released after the earlier requests finish.


def testUnavailableBaselinePersistsAcrossRetry(pollingDatabase, monkeypatch, tmp_path):
    from kernel_bench import jobs
    store, client, goal_id, target_id, configuration_id = pollingDatabase
    def forbidden(*args, **kwargs):
        pytest.fail('Unavailable baseline attempted a build')
    monkeypatch.setattr(jobs, 'runCmd', forbidden)
    headers = {'X-Bench-Token': 'polling-test'}
    assert client.post('/api/baselines/calibrate', json={'goal_id': goal_id}, headers=headers).status_code == 202
    row = {'id': configuration_id, 'goal_id': goal_id, 'matching_kernels': {}}
    jobs.calibrateConfiguration(store, row, tmp_path)
    assert store.getBaselines(goal_id, target_id)[0]['status'] == 'unavailable'
    client.post('/api/baselines/calibrate', json={'retry_failed': True}, headers=headers)
    assert store.getBaselines(goal_id, target_id)[0]['status'] == 'unavailable'
    requested = store.query('SELECT requested FROM kernel_bench.baselines WHERE configuration_id = %s', (configuration_id,))[0]
    assert requested['requested'] is True
    store.execute("UPDATE kernel_bench.goals SET metadata = '{}' WHERE id = %s", (goal_id,))
    response = client.post('/api/baselines/calibrate', json={'goal_id': goal_id}, headers=headers)
    assert response.status_code == 409
    assert response.get_json()['error_details']['kind'] == 'incompatible_native_interface'


def testAdminGoalSelectionAndQueuedPreparation(pollingDatabase, monkeypatch, tmp_path):
    store, client, goal_id, target_id, _ = pollingDatabase
    monkeypatch.setenv("KERNEL_BENCH_ADMIN_TOKEN", "operator-only")
    headers = {"X-Bench-Token": "polling-test"}
    assert client.get("/api/admin/goals", headers=headers).status_code == 403
    headers["X-Admin-Token"] = "operator-only"
    assert client.get("/api/admin/goals", headers=headers).get_json()["active_goal_id"] == goal_id
    response = client.post("/api/admin/goals/activate", headers=headers, json={"goal_id": goal_id})
    assert response.status_code == 200
    assert store.query("SELECT goal_id FROM kernel_bench.active_goal WHERE singleton")[0]["goal_id"] == goal_id
    cache = tmp_path / "goal.bin"
    cache.write_bytes(b"compiled")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"command": ["{python}", "run.py", "{cache}", "{report}"]}))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"gemma": {"name": "Gemma", "cache_path": str(cache), "promotion_config": str(recipe)}}))
    monkeypatch.setenv("KERNEL_BENCH_GOAL_CATALOG", str(catalog))
    assert client.post("/api/admin/goals/prepare", headers=headers, json={"preset_id": "unknown"}).status_code == 400
    response = client.post("/api/admin/goals/prepare", headers=headers, json={"preset_id": "gemma"})
    assert response.status_code == 202
    job = store.getJob(response.get_json()["job_id"])
    assert job["mode"] == "goal_setup" and job["status"] == "queued"
    assert job["result"]["cache_path"] == str(cache)
    assert job["result"]["recipe"]["kernel_speedup"] == 1.02
    assert client.get("/api/jobs?mode=goal_setup", headers=headers).status_code == 200
    assert client.get("/api/status", headers=headers).get_json()["queue"]["depth_by_mode"]["goal_setup"]["queued"] >= 1
