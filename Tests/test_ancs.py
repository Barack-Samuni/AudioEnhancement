import numpy as np
import pytest

NLMS_FILTER_WINDOW = 1024
NLMS_MU = 0.1
RLS_N_TAPS = 64
DEFAULT_SAMPLE_RATE = 16000


def test_rls(resampled_noise, resampled_sig):
    assert resampled_noise is not None
    assert resampled_sig is not None


def test_lms(resampled_noise, resampled_sig):
    assert resampled_noise is not None
    assert resampled_sig is not None


def test_nkf(resampled_noise, resampled_sig):
    assert resampled_noise is not None
    assert resampled_sig is not None


@pytest.fixture
def resampled_sig():
    return np.array([4, 5, 6])


@pytest.fixture
def resampled_noise():
    return np.array([1, 2, 3])
