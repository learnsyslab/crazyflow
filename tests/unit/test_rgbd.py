"""Unit tests for the RGB-D sensor."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("warp", reason="the RGB-D sensor requires the optional rgbd extra")

from crazyflow.sim import Sim  # noqa: E402
from crazyflow.sim.sensors.rgbd import build_render_rgbd_fn  # noqa: E402


@pytest.mark.unit
def test_build_render_rgbd_fn(device: str):
    """Test that the MuJoCo renderer returns RGB-D images that follow the drone poses."""
    sim = Sim(n_worlds=2, n_drones=2, device=device)
    pos = sim.data.states.pos.at[..., 2].set(jnp.array([[1.0], [2.0]]))
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
    render_fn = build_render_rgbd_fn(sim, drones=1, resolution=(32, 24))
    rgbd = render_fn(sim.data)
    assert rgbd.shape == (2, 1, 24, 32, 4), f"Expected shape (2, 1, 24, 32, 4), got {rgbd.shape}"
    assert rgbd.device == jax.devices(device)[0], f"Expected device {device}, got {rgbd.device}"
    assert jnp.all((rgbd[..., :3] >= 0) & (rgbd[..., :3] <= 1)), "RGB must be in [0, 1]"
    # The floor starts at the drone height / tan(fovy / 2) in the image
    fovy = np.deg2rad(sim.mj_model.cam("1:fpv_cam").fovy[0])
    expected = np.array([1.0, 2.0]) / np.tan(fovy / 2)
    assert np.allclose(rgbd[:, 0, -1, 16, 3], expected, rtol=0.05), "Depth does not match drone pos"
