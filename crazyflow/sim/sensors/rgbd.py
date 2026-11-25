"""RGB-D camera sensors built on the MuJoCo Warp ray tracer.

Renders batched RGB-D images from any model camera.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from mujoco import mjx

if TYPE_CHECKING:
    from typing import Callable, Sequence

    from jax import Array
    from mujoco.mjx import Data, Model
    from mujoco.mjx.warp.render_context import RenderContext

    from crazyflow.sim.data import SimData
    from crazyflow.sim.sim import Sim


def build_render_rgbd_fn(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    camera: str = "fpv_cam",
) -> Callable[[SimData], Array]:
    """Build an RGB-D renderer for a drone selection, camera, and resolution.

    Args:
        sim: The simulation.
        drones: Drones whose cameras are rendered. ``None`` renders every drone.
        resolution: Image resolution as (width, height).
        background: RGB background color with values in [0, 1].
        camera: Drone camera name.

    Returns:
        Renderer of RGB in [0, 1] followed by depth in meters along the camera's optical axis, of
        shape (n_worlds, n_selected, height, width, 4). Depth is 0 where nothing is hit.
    """
    drone_ids = range(sim.n_drones) if drones is None else np.atleast_1d(drones)
    camera_ids = [sim.mj_model.cam(f"{d}:{camera}").id for d in drone_ids]
    context = mjx.create_render_context(
        sim.mj_model,
        nworld=sim.n_worlds,
        devices=[wp.device_from_jax(sim.device).alias],
        cam_res=resolution,
        render_rgb=True,
        render_depth=True,
        cam_active=camera_ids,
        background_color=(*background, 1.0),
        enabled_geom_groups=[0, 1, 2],
    )
    mjx_model = mjx.put_model(sim.mj_model, device=sim.device, impl="warp")
    make_data = partial(mjx.make_data, sim.mj_model, device=sim.device, impl="warp")
    mjx_data = jax.vmap(lambda _: make_data())(jnp.arange(sim.n_worlds))
    return jax.jit(
        partial(
            _render, mjx_data=mjx_data, mjx_model=mjx_model, context=context, n_cams=len(camera_ids)
        )
    )


def _render(
    data: SimData, mjx_data: Data, mjx_model: Model, context: RenderContext, n_cams: int
) -> Array:
    """Render the RGB-D images of every active camera with each drone at its current pose."""
    context = context.pytree()  # We need to unwrap in the function to avoid dropping render buffers
    ids = data.core.drone_mocap_ids
    quat = jnp.roll(data.states.quat, 1, axis=-1)  # MuJoCo quats are [w, x, y, z]
    mjx_data = mjx_data.replace(
        mocap_pos=mjx_data.mocap_pos.at[:, ids].set(data.states.pos),
        mocap_quat=mjx_data.mocap_quat.at[:, ids].set(quat),
    )
    mjx_data = jax.vmap(mjx.forward, in_axes=(None, 0))(mjx_model, mjx_data)
    mjx_data = mjx.refit_bvh(mjx_model, mjx_data, context)  # Move the geometry to the new poses
    rgb, depth, _ = mjx.render(mjx_model, mjx_data, context)
    rgb = jnp.stack([mjx.get_rgb(context, i, rgb) for i in range(n_cams)], axis=1)
    return jnp.concatenate([rgb, depth.reshape(*rgb.shape[:-1], 1)], axis=-1)
