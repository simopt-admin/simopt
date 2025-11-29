import pytest

from ..exceptions import (
    MaxEvalError,
    TargetSuccess,
    CallbackSuccess,
    FeasibleSuccess,
)


def test_max_eval_error():
    with pytest.raises(MaxEvalError):
        raise MaxEvalError()


def test_target_success():
    with pytest.raises(TargetSuccess):
        raise TargetSuccess()


def test_callback_success():
    with pytest.raises(CallbackSuccess):
        raise CallbackSuccess()


def test_feasible_success():
    with pytest.raises(FeasibleSuccess):
        raise FeasibleSuccess()
