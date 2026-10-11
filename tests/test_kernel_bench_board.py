"""Board API contracts; set KERNEL_BENCH_BOARD_TEST_DATABASE_URL for isolated SQL checks."""
import os
import stat
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from psycopg.types.json import Jsonb

from kernel_bench.agent import KernelBenchClient
from kernel_bench.app import createApp
from kernel_bench.board import messageData
from kernel_bench.storage import Store


@pytest.mark.parametrize('data', [[], {}, {'body': ''}, {'body': 'x' * 16001},
                                      {'body': 'x', 'goal_id': True}, {'body': 'x', 'goal_id': 1.5},
                                      {'body': 'x', 'idempotency_key': ''}, {'body': 'x', 'job_id': 'bad'}])
def testMessageValidation(data):
    with pytest.raises((ValueError, TypeError)):
        messageData(data)


def testClientIdentityRestart(monkeypatch, tmp_path):
    identity = {'id': str(uuid.uuid4()), 'display_name': 'Agent', 'agent_token': 'private'}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return identity

    monkeypatch.setattr('requests.Session.post', lambda *args, **kwargs: Response())
    monkeypatch.setattr('requests.Session.get', lambda *args, **kwargs: Response())
    path = tmp_path / 'identity.json'
    client = KernelBenchClient('https://test.example', 'bench', path)
    assert client.registerAgent('Agent')['id'] == identity['id']
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    restarted = KernelBenchClient('https://test.example', 'bench', path)
    assert restarted.session.headers['X-Agent-Token'] == 'private'
    assert restarted.getIdentity()['id'] == identity['id']
    with pytest.raises(ValueError, match='another server'):
        KernelBenchClient('https://other.example', 'bench', path)


@pytest.fixture
def boardApp():
    url = os.environ.get('KERNEL_BENCH_BOARD_TEST_DATABASE_URL')
    if not url:
        pytest.skip('Set KERNEL_BENCH_BOARD_TEST_DATABASE_URL to an isolated PostgreSQL database')
    store = Store(url)
    store.initialize()
    fingerprint = str(uuid.uuid4())
    goal_id = store.query("""INSERT INTO kernel_bench.goals(fingerprint, name, cache_sha256, cache_data, source_archive, metadata)
        VALUES (%s, 'board test', '', '', '', %s) RETURNING id""", (fingerprint, Jsonb({'kernels': []})))[0]['id']
    app = createApp(database_url=url, goal_id=goal_id, token='board-test', start_worker=False)
    app.testing = True
    yield app, store, goal_id
    # This fixture deliberately requires a disposable database: no production URL fallback.
    with store.connection() as connection:
        connection.execute('TRUNCATE kernel_bench.messages, kernel_bench.feedback, kernel_bench.agents, kernel_bench.jobs CASCADE')
        connection.execute('DELETE FROM kernel_bench.configurations WHERE goal_id = %s', (goal_id,))
        connection.execute('DELETE FROM kernel_bench.goals WHERE id = %s', (goal_id,))


def testBoardAttributionPaginationAndRetries(boardApp):
    app, store, goal_id = boardApp
    client = app.test_client()
    bench = {'X-Bench-Token': 'board-test'}
    assert client.get('/api/channels').status_code == 401
    agents = [client.post('/api/agents/register', headers=bench, json={'display_name': 'Same name'}).get_json() for _ in range(2)]
    assert agents[0]['id'] != agents[1]['id']
    assert agents[0]['agent_token'] != agents[1]['agent_token']
    assert client.post('/api/agents/register', headers=bench, json={'display_name': ''}).status_code == 400
    assert client.get('/api/agents/me', headers=bench).status_code == 401
    headers = [{**bench, 'X-Agent-Token': a['agent_token']} for a in agents]
    assert client.get('/api/agents/me', headers=headers[0]).get_json()['id'] == agents[0]['id']
    assert client.get('/api/agents/me', headers={**bench, 'X-Agent-Token': 'bad'}).status_code == 401
    channels = client.get('/api/channels', headers=bench).get_json()['channels']
    assert {c['id'] for c in channels} == {'general', 'suggestions'}
    endpoint = '/api/channels/general/messages'
    payload = {'body': '<script>alert(1)</script> **hello**', 'idempotency_key': 'retry', 'goal_id': goal_id,
               'author_id': agents[1]['id']}
    assert client.post(endpoint, headers=bench, json=payload).status_code == 401
    first = client.post(endpoint, headers=headers[0], json=payload)
    assert first.status_code == 201
    row = first.get_json()
    assert row['author_id'] == agents[0]['id']
    assert 'credential_hash' not in row and 'agent_token' not in row
    assert client.post(endpoint, headers=headers[0], json=payload).get_json()['id'] == row['id']
    assert client.post(endpoint, headers=headers[0], json={**payload, 'body': 'changed'}).status_code == 400
    second = client.post(endpoint, headers=headers[1], json={'body': 'reply', 'idempotency_key': 'retry', 'reply_to_message_id': row['id']}).get_json()
    assert second['author_id'] == agents[1]['id']
    page = client.get(endpoint+'?limit=1', headers=bench).get_json()
    assert page['messages'][0]['id'] == row['id'] and page['has_more']
    next_page = client.get(endpoint+f"?cursor={page['next_cursor']}", headers=bench).get_json()
    assert [m['id'] for m in next_page['messages']] == [second['id']]
    assert client.get('/api/messages/'+str(row['id']), headers=bench).get_json()['body'] == payload['body']
    assert client.get(endpoint+'?limit=201', headers=bench).status_code == 400
    assert client.post('/api/channels/suggestions/messages', headers=headers[0], json={'body': 'wrong reply', 'reply_to_message_id': row['id']}).status_code == 400
    assert client.post(endpoint, headers=headers[0], json={'body': 'bad link', 'goal_id': 9223372036854775807}).status_code == 400
    assert client.get('/api/channels/absent/messages', headers=bench).status_code == 404
    # Two concurrent retries commit only one message.
    def postRetry(_):
        return app.test_client().post(endpoint, headers=headers[0], json={'body': 'concurrent', 'idempotency_key': 'concurrent'}).get_json()['id']
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(postRetry, range(2)))
    assert len(set(ids)) == 1
    legacy = {'title': 'Improve docs', 'description': 'Please add examples', 'idempotency_key': 'suggestion'}
    saved = client.post('/api/suggestions', headers=headers[0], json=legacy).get_json()
    assert client.post('/api/suggestions', headers=headers[0], json=legacy).get_json() == saved
    message = client.get('/api/messages/'+str(saved['message_id']), headers=bench).get_json()
    assert message['channel_id'] == 'suggestions' and message['author_id'] == agents[0]['id']
    feedback = store.query('SELECT * FROM kernel_bench.feedback WHERE id = %s', (saved['suggestion_id'],))[0]
    assert str(feedback['agent_id']) == agents[0]['id']
    target_id = str(uuid.uuid4())
    store.execute("INSERT INTO kernel_bench.targets(id, expression) VALUES (%s, 'BOARD')", (target_id,))
    store.execute("INSERT INTO kernel_bench.configurations(goal_id, target_id, fingerprint, configuration, matching_kernels) VALUES (%s, %s, 'board', '{}', '[]')", (goal_id, target_id))
    submitted = client.post('/api/iteration/submit', headers=headers[0], json={
        'source': 'REGISTER_KERNEL("BoardTest", unused)', 'target_id': target_id, 'backend': 'cpu'})
    assert submitted.status_code == 202
    job_id = submitted.get_json()['job_id']
    assert str(store.getJob(job_id)['agent_id']) == agents[0]['id']
    assert client.post(endpoint, headers=headers[0], json={'body': 'linked', 'job_id': job_id, 'goal_id': goal_id, 'target_id': target_id}).status_code == 201
    assert client.post(endpoint, headers=headers[0], json={'body': 'bad context', 'job_id': job_id, 'target_id': 'missing'}).status_code == 400
    assert client.post('/api/reports', headers=headers[1], json={'result': 'working'}).status_code == 201
    report = store.query("SELECT agent_id FROM kernel_bench.feedback WHERE kind = 'report'")[0]
    assert str(report['agent_id']) == agents[1]['id']
    count = len(store.query('SELECT * FROM kernel_bench.feedback'))
    assert client.post('/api/suggestions', headers=headers[0], json={**legacy, 'job_id': str(uuid.uuid4())}).status_code == 400
    assert len(store.query('SELECT * FROM kernel_bench.feedback')) == count
    unattributed = client.post('/api/suggestions', headers=bench, json={'description': 'legacy'}).get_json()
    assert client.get('/api/messages/'+str(unattributed['message_id']), headers=bench).get_json()['author_type'] == 'unknown'
    client.set_cookie('bench_token', 'board-test')
    assert client.post(endpoint, json={'body': 'operator'}).get_json()['author_type'] == 'operator'
    assert client.post(endpoint, headers={'Origin': 'https://other.example'}, json={'body': 'cross-origin'}).status_code == 401
    assert client.get('/board').status_code == 200


def testBrowserMarkdownIsSafe():
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser Markdown safety check')
    template = Path('kernel_bench/templates/board.html').read_text()
    script = template.split('<script>', 1)[1].split('</script>', 1)[0]
    subprocess.run([node, '--check'], input=script, text=True, check=True, capture_output=True)
    markdown = script.split('function markdown(', 1)[1].split('function render(', 1)[0]
    harness = '''
const assert = require('node:assert/strict');
const document = {
 createTextNode: text => ({type: 'text', text}),
 createElement: type => ({type, textContent: '', children: [], append(...items){this.children.push(...items);}})
};
const location = {origin: 'https://bench.example'};
'''
    harness += 'function markdown(' + markdown
    harness += '''
const parent=document.createElement('div');
markdown(parent, '<script>alert(1)</script> **bold** *italic* `code` [safe](https://example.com) [bad](javascript:alert)');
assert.equal(parent.children[0].type, 'text');
assert.ok(parent.children[0].text.includes('<script>'));
assert.equal(parent.children.filter(n=>n.type==='a').length, 1);
assert.ok(parent.children.some(n=>n.type==='strong' && n.textContent==='bold'));
assert.ok(parent.children.some(n=>n.type==='code' && n.textContent==='code'));
assert.equal(parent.children.find(n=>n.type==='a').href, 'https://example.com/');
'''
    subprocess.run([node], input=harness, text=True, check=True, capture_output=True)
