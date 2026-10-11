"""Phase-three contracts are discoverable without a native build."""
import struct

import pytest

from kernel_bench.app import createApp
from kernel_bench.contract import decodeConstants
from kernel_bench.onboarding import SKELETONS
from kernel_linter_rules import REDUNDANCY_PATTERNS, getRules


def testConstants():
    assert decodeConstants({'input_dtypes': [0, 1, 3, 8], 'input_constants':
                            [list(struct.pack('=f', 128)), list(struct.pack('=i', -3)),
                             list(struct.pack('=H', 0x3f80)), [1]]}) == [[128.0], [-3], [1.0], None]
    with pytest.raises(ValueError, match='byte count'):
        decodeConstants({'input_dtypes': [0], 'input_constants': [[1]]})


def testDiscovery(monkeypatch):
    kernel = {'kernel_id': 1, 'name': 'baseline', 'reference_expression': 'COPY($0,)',
              'source_path': 'tensor_graphs_cpp/kernels/fused/copy/CUDA_CUDA.cu'}

    class FakeStore:
        def __init__(self, database_url):
            pass

        def initialize(self):
            pass

        def getGoal(self, goal_id):
            return {'metadata': {'kernels': [kernel, kernel], 'enabled_capabilities': ['TG_USE_CUDA']}}

        def getConfigurations(self, goal_id, target_id):
            return [{'id': 1, 'configuration': {'input_dtypes': [0], 'input_constants': [list(struct.pack('=f', 2.5))]}}]

        def query(self, sql, parameters):
            return [{'expression': 'COPY($0,)'}]

        def getBestMeasurements(self, goal_id, target_id):
            return []

        def getTargets(self, goal_id):
            return [{'id': 'copy', 'reference_expression': 'COPY($0,)', 'matching_kernels': [{'1': 'baseline'}, {'1': 'baseline'}]}]

    monkeypatch.setattr('kernel_bench.app.Store', FakeStore)
    client = createApp(token='test', goal_id='goal', start_worker=False).test_client()
    headers = {'X-Bench-Token': 'test'}
    assert client.get('/api/kernel-contract', headers=headers).get_json()['skeletons'] == SKELETONS
    assert client.get('/api/linter/rules', headers=headers).get_json()['rules'] == getRules()
    assert len(getRules()) > len(REDUNDANCY_PATTERNS)
    assert client.get('/api/kernels/list?name=baseline', headers=headers).get_json()['available'] is False
    assert client.get('/api/kernels/list?name=new', headers=headers).get_json()['available'] is True
    target = client.get('/api/targets', headers=headers).get_json()['targets'][0]
    assert target['matching_kernels'] == {'1': 'baseline'}
    assert target['source_paths'] == [kernel['source_path']]
    detail = client.get('/api/targets/copy', headers=headers).get_json()
    assert detail['kernels'] == [kernel]
    assert detail['source_paths'] == [kernel['source_path']]
    configuration = detail['configurations'][0]['configuration']
    assert configuration['decoded_input_constants'] == [[2.5]]
    assert configuration['input_constants'] == [list(struct.pack('=f', 2.5))]


def testCoverageSurvivesCorrectnessFailure(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from kernel_bench import native_runner
    candidate = {'kernel_id': 1, 'name': 'candidate', 'reference_expression': 'ADD',
                 'has_reference_factory': True, 'is_reference': False, 'is_view': False}
    native = SimpleNamespace(
        kernel_bench_capabilities=lambda: [], kernel_bench_kernels=lambda: [candidate],
        kernel_bench_match_diagnostics=lambda kernel_id, configuration: configuration,
        kernel_bench_test_diagnostics=lambda *args: {'passed': False, 'kind': 'correctness_mismatch'})
    monkeypatch.setattr(native_runner, 'native', native)
    payload = {'kernel_name': 'candidate', 'reference_expression': 'ADD', 'action': 'evaluate',
               'output_path': str(tmp_path / 'result.json'),
               'configurations': [{'id': 1, 'configuration': {'matched': True, 'reasons': []}},
                                  {'id': 2, 'configuration': {'matched': False, 'reasons': ['dtype']}},
                                  {'id': 3, 'configuration': {'matched': False, 'reasons': ['custom_match_rejected']}}]}
    state = {'result': {}}
    with pytest.raises(native_runner.NativeFailure):
        native_runner.evaluate(payload, state)
    assert state['result']['matched_configuration_ids'] == [1]
    assert state['result']['unmatched_configurations'] == [
        {'configuration_id': 2, 'reasons': ['dtype']},
        {'configuration_id': 3, 'reasons': ['custom_match_rejected']}]


def testNativeBuiltInReasons():
    native = pytest.importorskip('kernel_bench_native')
    if not hasattr(native, 'kernel_bench_match_diagnostics'):
        pytest.skip('Rebuild native bindings for phase-three diagnostics')
    configuration = {'input_shapes': [[8], [8]], 'input_strides': [[1], [1]],
                     'input_dtypes': [0, 0], 'input_constants': [[], []],
                     'output_shape': [8], 'output_strides': [1], 'output_dtype': 0}
    kernel = next(kernel for kernel in native.kernel_bench_kernels()
                  if kernel['is_reference'] and kernel['reference_expression'] == 'MUL($0,$1,)'
                  and native.kernel_bench_matches(kernel['kernel_id'], configuration))
    diagnose = lambda: native.kernel_bench_match_diagnostics(kernel['kernel_id'], configuration)
    assert diagnose() == {'matched': True, 'reasons': []}
    configuration['input_dtypes'] = [2, 2]
    assert diagnose()['reasons'] == ['dtype']
    configuration['input_dtypes'] = [0, 0]
    configuration['input_shapes'] = [[8], [3]]
    assert diagnose()['reasons'] == ['custom_match_rejected']
    configuration['input_shapes'] = [[8]]
    configuration['input_strides'] = [[1]]
    configuration['input_dtypes'] = [0]
    configuration['input_constants'] = [[]]
    assert diagnose()['reasons'] == ['input_count']
