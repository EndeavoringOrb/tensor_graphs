"""Baseline selection and snapshot interface contracts."""
import pytest

from kernel_bench import jobs
from kernel_bench.policy import NATIVE_INTERFACE_VERSION, baselineEligible


@pytest.mark.parametrize('field', ['is_view', 'is_reference', 'has_reference_factory', None])
def testEligibility(field):
    kernel = {'is_view': False, 'is_reference': False, 'has_reference_factory': True}
    if field:
        kernel[field] = not kernel[field]
    assert baselineEligible(kernel) == (field is None)


class FakeStore:
    def __init__(self, version=NATIVE_INTERFACE_VERSION):
        self.version = version
        self.calls = []
    def getGoal(self, goal_id):
        return {'metadata': {'native_interface_version': self.version}}
    def execute(self, sql, parameters=()):
        self.calls.append((sql, parameters))
    def query(self, sql, parameters=()):
        self.calls.append((sql, parameters))
        return []


def testAutomaticCalibrationScopesActiveGoal(tmp_path):
    store = FakeStore()
    assert jobs.calibrateNext(store, tmp_path, 42) is False
    assert all(parameters == (42,) for sql, parameters in store.calls)
    assert 'WHERE goal_id = %s' in store.calls[0][0]
    assert 'c.goal_id = %s OR b.requested' in store.calls[1][0]


def testUnavailableBaselineDoesNotBuild(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail('Unavailable baseline attempted native work')
    monkeypatch.setattr(jobs, 'runCmd', forbidden)
    monkeypatch.setattr(jobs, 'runNative', forbidden)
    store = FakeStore()
    jobs.calibrateConfiguration(store, {'id': 1, 'goal_id': 42, 'matching_kernels': {}}, tmp_path)
    assert "status = 'unavailable'" in store.calls[-1][0]
    assert 'reference factory' in store.calls[-1][1][0]


@pytest.mark.parametrize('version', [None, 0, NATIVE_INTERFACE_VERSION + 1])
def testIncompatibleGoalRejectedBeforeBuild(monkeypatch, tmp_path, version):
    def forbidden(*args, **kwargs):
        pytest.fail('Incompatible snapshot attempted native work')
    monkeypatch.setattr(jobs, 'runCmd', forbidden)
    with pytest.raises(jobs.JobFailure) as error:
        jobs.evaluateJob(FakeStore(version), {'goal_id': 42}, tmp_path)
    assert error.value.details['kind'] == 'incompatible_native_interface'
    assert not (tmp_path / 'build.py').exists()
