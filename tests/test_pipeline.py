import pytest

from cairn import Pipeline, step
from cairn.exceptions import CyclicDependencyError, StepFailedError, UnknownStepError


def test_runs_steps_in_dependency_order():
    order = []

    @step
    def a():
        order.append("a")
        return 1

    @step
    def b(a):
        order.append("b")
        return a + 1

    @step
    def c(b):
        order.append("c")
        return b + 1

    pipeline = Pipeline("linear")
    pipeline.add(a)
    pipeline.add(b, depends_on=[a])
    pipeline.add(c, depends_on=[b])

    results = pipeline.run()

    assert order == ["a", "b", "c"]
    assert results["c"] == 3


def test_independent_steps_both_run():
    @step
    def left():
        return "left"

    @step
    def right():
        return "right"

    @step
    def join(left, right):
        return f"{left}+{right}"

    pipeline = Pipeline("fan-in")
    pipeline.add(left)
    pipeline.add(right)
    pipeline.add(join, depends_on=[left, right])

    results = pipeline.run()
    assert results["join"] == "left+right"


def test_initial_inputs_are_available_to_steps():
    @step
    def double(x):
        return x * 2

    pipeline = Pipeline("with-input")
    pipeline.add(double)

    results = pipeline.run(x=21)
    assert results["double"] == 42


def test_cyclic_dependency_is_detected():
    @step
    def a(b):
        return b

    @step
    def b(a):
        return a

    pipeline = Pipeline("cyclic")
    pipeline.add(a, depends_on=["b"])
    pipeline.add(b, depends_on=["a"])

    with pytest.raises(CyclicDependencyError):
        pipeline.run()


def test_unknown_dependency_raises():
    @step
    def a():
        return 1

    pipeline = Pipeline("bad-dep")
    pipeline.add(a, depends_on=["does_not_exist"])

    with pytest.raises(UnknownStepError):
        pipeline.run()


def test_step_retries_before_succeeding():
    attempts = {"count": 0}

    @step(retries=2, retry_delay=0)
    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("not yet")
        return "ok"

    pipeline = Pipeline("retrying")
    pipeline.add(flaky)

    results = pipeline.run()
    assert results["flaky"] == "ok"
    assert attempts["count"] == 3


def test_step_raises_after_exhausting_retries():
    @step(retries=1, retry_delay=0)
    def always_fails():
        raise ValueError("nope")

    pipeline = Pipeline("failing")
    pipeline.add(always_fails)

    with pytest.raises(StepFailedError):
        pipeline.run()
