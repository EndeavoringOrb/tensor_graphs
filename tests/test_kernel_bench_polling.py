"""Polling contracts and additive PostgreSQL migration checks."""
import os
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
        VALUES (%s, 'polling test', '', '', '', %s) RETURNING id""", (key, Jsonb({'kernels': []})))[0]['id']
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
    row = {'id': configuration_id, 'goal_id': goal_id, 'expression': target_id, 'configuration': {}}
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
