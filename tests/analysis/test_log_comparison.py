import numpy as np
import pytest

from tests.utils import assert_logs_match, capture_log_data


@pytest.mark.parametrize(
    ("actual", "desired", "message"),
    [
        ([], [], "actual plot logs are empty"),
        ([[1]], [], "desired plot logs are empty"),
        ([[1], [2], [3]], [[1], [2]], "record count differs"),
        ([[1], [2, 3]], [[1]], "record count differs"),
        ([[1], [2, 3]], [[1], [[2, 3]]], "record 1: shape differs"),
        ([[1], [2, 3]], [[1], [2, 4]], "record 1"),
        ([[[1], [2, 3]]], [[[1], [2, 4]]], "record 0, component 1"),
        ([[[1], [2, 3]]], [[[1], [2, 3, 4]]], "record 0, component 1: shape differs"),
        ([[[1], [2, 3]]], [[[1], [2, 3], [4]]], "record 0: component count differs"),
    ],
)
def test_assert_logs_match_rejects_mismatches(actual, desired, message):
    with pytest.raises(AssertionError, match=message):
        assert_logs_match(actual, desired)


def test_assert_logs_match_accepts_ragged_records():
    actual = [np.array([[1, 2], [3, 4]]), [[0, 1], [0, 0.5, 1]]]
    desired = [np.array([[1, 2], [3, 4.000001]]), [[0, 1], [0, 0.5, 1]]]
    assert_logs_match(actual, desired)


def test_missing_data_logs_fail_comparison():
    def plot(*, logger):
        logger.debug("renamed", curve=[1, 2])

    actual = capture_log_data(plot)
    desired = capture_log_data(plot)
    with pytest.raises(AssertionError, match="actual plot logs are empty"):
        assert_logs_match(actual, desired)
