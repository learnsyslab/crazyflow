"""Unit tests for the named simulation pipeline datastructure."""

from functools import partial

import jax.numpy as jnp
import pytest

from crazyflow.sim import Sim
from crazyflow.sim.pipeline import Pipeline


def fn_a(x: int) -> int:
    return x + 1


def fn_b(x: int) -> int:
    return x * 2


def fn_c(x: int) -> int:
    return x - 3


@pytest.mark.unit
def test_names_and_iteration():
    pipeline = Pipeline((fn_a, fn_b))
    assert pipeline.names == ("fn_a", "fn_b")
    assert tuple(pipeline) == (fn_a, fn_b)
    assert len(pipeline) == 2
    assert "fn_a" in pipeline
    assert "missing" not in pipeline


@pytest.mark.unit
def test_insert_before_after():
    pipeline = Pipeline((fn_a, fn_b))
    pipeline.insert_before("fn_b", fn_c)
    assert pipeline.names == ("fn_a", "fn_c", "fn_b")
    pipeline.remove("fn_c")
    pipeline.insert_after("fn_a", fn_c)
    assert pipeline.names == ("fn_a", "fn_c", "fn_b")


@pytest.mark.unit
def test_append_prepend_replace_remove():
    pipeline = Pipeline((fn_a,))
    pipeline.append(fn_b)
    pipeline.prepend(fn_c)
    assert pipeline.names == ("fn_c", "fn_a", "fn_b")
    pipeline.replace("fn_a", fn_b)
    assert pipeline.names == ("fn_c", "fn_a", "fn_b"), "Replace must keep position and name"
    assert tuple(pipeline)[1] == fn_b
    pipeline.remove("fn_c")
    assert pipeline.names == ("fn_a", "fn_b")


@pytest.mark.unit
def test_unique_names():
    pipeline = Pipeline((fn_a,))
    with pytest.raises(KeyError, match="already exists"):
        pipeline.append(fn_a)
    with pytest.raises(KeyError, match="already exists"):
        pipeline.insert_after("fn_a", fn_b, name="fn_a")


@pytest.mark.unit
def test_anonymous_fn_requires_name():
    pipeline = Pipeline()
    with pytest.raises(ValueError, match="explicit name"):
        pipeline.append(partial(fn_a))
    pipeline.append(partial(fn_a), name="fn_a_partial")
    assert pipeline.names == ("fn_a_partial",)


@pytest.mark.unit
def test_missing_stage():
    pipeline = Pipeline((fn_a,))
    with pytest.raises(ValueError, match="No pipeline stage named 'missing'"):
        pipeline.insert_before("missing", fn_b)


@pytest.mark.unit
def test_sums():
    pipeline = Pipeline((fn_a,))
    pipeline = pipeline + fn_b
    pipeline.append(fn_c, "fn_c_named")
    assert pipeline.names == ("fn_a", "fn_b", "fn_c_named")
    pipeline = Pipeline((fn_a,))
    pipeline += fn_b
    assert pipeline.names == ("fn_a", "fn_b")
    pipeline = (fn_c,) + Pipeline((fn_a,))
    assert pipeline.names == ("fn_c", "fn_a")
    with pytest.raises(ValueError, match="no __name__"):
        Pipeline((fn_a,)) + ("fn_b_named", fn_b)  # ``+`` only appends named functions


@pytest.mark.unit
def test_sum_returns_new_pipeline():
    pipeline = Pipeline((fn_a,))
    extended = pipeline + fn_b
    assert pipeline.names == ("fn_a",), "Sums must not modify the original pipeline"
    assert extended.names == ("fn_a", "fn_b")


@pytest.mark.unit
def test_sim_pipeline_names():
    """The default sim pipelines expose the documented stage names."""
    sim = Sim(control="state")
    assert sim.step_pipeline.names == (
        "step_state_controller",
        "step_attitude_controller",
        "step_force_torque_controller",
        "integration",
        "increment_steps",
        "clip_floor_pos",
    )
    assert len(sim.reset_pipeline) == 0
    sim.close()


@pytest.mark.unit
def test_pipeline_snapshot():
    """Modifications after building must not leak into the compiled step function."""
    sim = Sim()
    sim.build_step_fn()
    steps = sim.data.core.steps

    def fail_stage(data):  # noqa: ANN001, ANN202
        raise AssertionError("Stage added after build_step_fn must not be traced")

    sim.step_pipeline.append(fail_stage)
    sim.step()  # Traces the compiled function on first call, must use the snapshot
    assert jnp.all(sim.data.core.steps == steps + 1)
    sim.close()
