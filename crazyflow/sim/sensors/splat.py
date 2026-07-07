"""Gaussian splat camera sensor built on `splax <https://github.com/amacati/splax>`_.

Renders batched RGB images of the splats attached via :func:`crazyflow.sim.splat.attach_splats`
from any model camera. splax rasterizes with CUDA kernels only, so this module requires the
``splats`` extra and a simulation constructed with ``device="gpu"``.

Note:
    Splat-based depth images are not supported yet because splax's depth path lacks dynamic
    transform support.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import splax
from scipy.spatial.transform import RigidTransform
from scipy.spatial.transform import Rotation as R

from crazyflow.sim.sim import requires_mujoco_sync
from crazyflow.sim.splat import SPLAT_KEYS, SPLAT_SLICES_KEY, requires_gpu, requires_splats

if TYPE_CHECKING:
    from typing import Callable

    import mujoco
    from jax import Array

    from crazyflow.sim.sim import Sim

# TODO: Add splat-based depth camera sensor


@requires_mujoco_sync
@requires_splats
@requires_gpu
def render_splat_rgb(
    sim: Sim,
    camera: int = 0,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    exclude_drone: int | None = None,
) -> Array:
    """Render RGB images of the attached splats from a model camera in all worlds.

    Scene gaussians are rendered at their static pose, drone gaussians follow the drone poses.

    Args:
        sim: The simulation to render.
        camera: Camera index.
        resolution: Image resolution as (width, height).
        background: RGB background color with values in [0, 1].
        exclude_drone: If not None, hide this drone's splat from the image (e.g. the drone that
            carries the camera).

    Returns:
        RGB images with values in [0, 1] and shape (n_worlds, height, width, 3).
    """
    f, c = camera_intrinsics(sim.mj_model, camera, resolution)
    arrays = tuple(sim.data.plugins[key] for key in SPLAT_KEYS)
    slices = tuple((int(start), int(stop)) for start, stop in sim.data.plugins[SPLAT_SLICES_KEY])
    return _render_splats(
        *arrays,
        cam_xpos=sim.mjx_data.cam_xpos[:, camera],
        cam_xmat=sim.mjx_data.cam_xmat[:, camera],
        pos=sim.data.states.pos,
        quat=sim.data.states.quat,
        slices=slices,
        img_shape=(resolution[1], resolution[0]),
        f=f,
        c=c,
        background=background,
        exclude_drone=exclude_drone,
    )


@requires_splats
@requires_gpu
def build_render_splat_fn(
    sim: Sim,
    camera: int = 0,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    exclude_drone: int | None = None,
) -> Callable[[Sim], Array]:
    """Build a splat renderer function for a given camera and resolution.

    Mirrors :func:`crazyflow.sim.sensors.depth.build_render_depth_fn`: the camera intrinsics and
    the static slice metadata are baked into the returned function, which takes a ``Sim`` object
    and returns RGB images of shape (n_worlds, height, width, 3). Improves performance compared to
    :func:`crazyflow.sim.sensors.splat.render_splat_rgb`. Requires splats to have been attached via
    :func:`crazyflow.sim.splat.attach_splats`.
    """
    f, c = camera_intrinsics(sim.mj_model, camera, resolution)
    slices = tuple((int(start), int(stop)) for start, stop in sim.data.plugins[SPLAT_SLICES_KEY])
    render_fn = partial(
        _render_splats,
        slices=slices,
        img_shape=(resolution[1], resolution[0]),
        f=f,
        c=c,
        background=background,
        exclude_drone=exclude_drone,
    )

    @requires_mujoco_sync
    def render_splat_fn(sim: Sim) -> Array:
        arrays = tuple(sim.data.plugins[key] for key in SPLAT_KEYS)
        return render_fn(
            *arrays,
            cam_xpos=sim.mjx_data.cam_xpos[:, camera],
            cam_xmat=sim.mjx_data.cam_xmat[:, camera],
            pos=sim.data.states.pos,
            quat=sim.data.states.quat,
        )

    return render_splat_fn


@jax.jit
def viewmats(cam_xpos: Array, cam_xmat: Array) -> Array:
    """World-to-camera matrices in OpenCV convention for MuJoCo cameras.

    MuJoCo cameras look along -z with +y up (OpenGL convention), whereas splax expects OpenCV
    convention cameras (+z forward, +y down). Flipping the y and z camera axes converts between
    the two.

    Args:
        cam_xpos: Camera positions of shape (..., 3).
        cam_xmat: Camera rotation matrices (camera-to-world) of shape (..., 3, 3).

    Returns:
        World-to-camera matrices of shape (..., 4, 4).
    """
    rot_c2w = cam_xmat * jnp.array([1.0, -1.0, -1.0])  # Flip the y and z camera axes (columns)
    rot = jnp.swapaxes(rot_c2w, -1, -2)
    trans = -(rot @ cam_xpos[..., None])[..., 0]
    return RigidTransform.from_components(trans, R.from_matrix(rot, assume_valid=True)).as_matrix()


def camera_intrinsics(
    mj_model: mujoco.MjModel, camera: int, resolution: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pinhole intrinsics of a model camera for a given image resolution.

    Args:
        mj_model: MuJoCo model containing the camera.
        camera: Camera index.
        resolution: Image resolution as (width, height).

    Returns:
        Focal lengths (fx, fy) and principal point (cx, cy) in pixels.
    """
    width, height = resolution
    fov_y = np.deg2rad(mj_model.cam_fovy[camera])
    focal = float(height / (2.0 * np.tan(fov_y / 2.0)))
    return (focal, focal), (width / 2.0, height / 2.0)


@jax.jit(static_argnames=("slices", "img_shape", "f", "c", "background", "exclude_drone"))
def _render_splats(
    means: Array,
    scales: Array,
    quats: Array,
    colors: Array,
    opacities: Array,
    cam_xpos: Array,
    cam_xmat: Array,
    pos: Array,
    quat: Array,
    slices: tuple[tuple[int, int], ...],
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    background: tuple[float, float, float],
    exclude_drone: int | None,
) -> Array:
    """Render the splat buffer for all worlds with the drones at their current poses."""
    vm = viewmats(cam_xpos, cam_xmat)
    bg = jnp.asarray(background, dtype=means.dtype)
    render = partial(
        splax.inference.render,
        means,
        scales,
        quats,
        colors,
        opacities,
        background=bg,
        img_shape=img_shape,
        f=f,
        c=c,
    )
    if not slices:
        return jax.vmap(lambda v: render(viewmat=v))(vm)
    tfs = RigidTransform.from_components(pos, R.from_quat(quat)).as_matrix()
    if exclude_drone is not None:
        # Slices are static, so drones cannot be dropped dynamically. Instead, teleport the
        # excluded drone's gaussians far below the scene where they are culled.
        far = jnp.eye(4, dtype=tfs.dtype).at[2, 3].set(-1e4)
        tfs = tfs.at[:, exclude_drone].set(far)
    render_world = lambda v, t: render(viewmat=v, gaussian_transforms=t, gaussian_slices=slices)  # noqa: E731
    return jax.vmap(render_world)(vm, tfs)
