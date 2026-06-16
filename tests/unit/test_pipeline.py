"""Unit tests for pipeline utility helpers based on OrderedDict."""

from collections import OrderedDict
from functools import partial

import jax.numpy as jnp
import pytest

from crazyflow.sim import Sim
from crazyflow.sim.pipeline import (
    append_fn,
    insert_fn_after,
    insert_fn_before,
    prepend_fn,
    remove_fn,
    replace_fn,
)


def fn_a(x: int) -> int:
    return x + 1


def fn_b(x: int) -> int:
    return x * 2


def fn_c(x: int) -> int:
    return x - 3


@pytest.mark.unit
def test_append_and_prepend_order_and_names():
    pipeline: OrderedDict[str, object] = OrderedDict()
    append_fn(pipeline, fn_a)
    append_fn(pipeline, fn_b)
    prepend_fn(pipeline, fn_c)
    assert tuple(pipeline.keys()) == ("fn_c", "fn_a", "fn_b")
    assert tuple(pipeline.values()) == (fn_c, fn_a, fn_b)


@pytest.mark.unit
def test_insert_before_after():
    pipeline = OrderedDict([("fn_a", fn_a), ("fn_b", fn_b)])
    insert_fn_before(pipeline, "fn_b", fn_c)
    assert tuple(pipeline.keys()) == ("fn_a", "fn_c", "fn_b")
    remove_fn(pipeline, "fn_c")
    insert_fn_after(pipeline, "fn_a", fn_c)
    assert tuple(pipeline.keys()) == ("fn_a", "fn_c", "fn_b")


@pytest.mark.unit
def test_replace_keeps_position_and_name():
    pipeline = OrderedDict([("fn_c", fn_c), ("fn_a", fn_a), ("fn_b", fn_b)])
    replace_fn(pipeline, fn_b, "fn_a")
    assert tuple(pipeline.keys()) == ("fn_c", "fn_a", "fn_b")
    assert tuple(pipeline.values())[1] == fn_b


@pytest.mark.unit
def test_remove_existing_stage():
    pipeline = OrderedDict([("fn_a", fn_a), ("fn_b", fn_b)])
    remove_fn(pipeline, "fn_a")
    assert tuple(pipeline.keys()) == ("fn_b",)


@pytest.mark.unit
def test_unique_names():
    pipeline = OrderedDict([("fn_a", fn_a)])
    with pytest.raises(KeyError, match="already exists"):
        append_fn(pipeline, fn_a)
    with pytest.raises(KeyError, match="already exists"):
        insert_fn_after(pipeline, "fn_a", fn_b, name="fn_a")


@pytest.mark.unit
def test_anonymous_fn_requires_name():
    pipeline: OrderedDict[str, object] = OrderedDict()
    with pytest.raises(ValueError, match="explicit name"):
        append_fn(pipeline, partial(fn_a))
    append_fn(pipeline, partial(fn_a), name="fn_a_partial")
    assert tuple(pipeline.keys()) == ("fn_a_partial",)


@pytest.mark.unit
def test_missing_stage():
    pipeline = OrderedDict([("fn_a", fn_a)])
    with pytest.raises(KeyError, match="No stage named 'missing'"):
        insert_fn_before(pipeline, "missing", fn_b)
    with pytest.raises(KeyError, match="No stage named 'missing'"):
        replace_fn(pipeline, fn_b, "missing")
    with pytest.raises(KeyError, match="No stage named 'missing'"):
        remove_fn(pipeline, "missing")


@pytest.mark.unit
def test_explicit_name_with_append():
    pipeline = OrderedDict([("fn_a", fn_a)])
    append_fn(pipeline, fn_c, name="fn_c_named")
    assert tuple(pipeline.keys()) == ("fn_a", "fn_c_named")


@pytest.mark.unit
def test_pipeline_snapshot():
    """Modifications after building must not leak into the compiled step function."""
    sim = Sim()
    sim.build_step_fn()
    steps = sim.data.core.steps

    def fail_stage(data):  # noqa: ANN001, ANN202
        raise AssertionError("Stage added after build_step_fn must not be traced")

    append_fn(sim.step_pipeline, fail_stage)
    sim.step()  # Traces the compiled function on first call, must use the snapshot
    assert jnp.all(sim.data.core.steps == steps + 1)
    sim.close()
