"""Gaussian splat camera sensor built on `splax <https://github.com/learnsyslab/splax>`_.

Renders batched RGB images of the splats attached via :func:`crazyflow.sim.splat.attach_splats`
from any model camera. splax rasterizes with CUDA kernels only, so this module requires the
``splats`` extra and a simulation constructed with ``device="gpu"``.

Note:
    Splat-based depth images are not supported yet.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import splax
from scipy.spatial.transform import RigidTransform
from scipy.spatial.transform import Rotation as R

from crazyflow.sim.sim import requires_mujoco_sync
from crazyflow.sim.splat import SPLAT_KEYS, SPLAT_SLICES_KEY, requires_gpu, requires_splats

if TYPE_CHECKING:
    from typing import Callable, Sequence

    from jax import Array

    from crazyflow.sim.sim import Sim

# TODO: Add splat-based depth camera sensor


@requires_mujoco_sync
@requires_splats
@requires_gpu
def render_splat_rgb(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    exclude_self: bool = False,
    camera_prefix: str = "fpv_cam",
) -> Array:
    """Render RGB images of the attached splats from the drone fpv cameras in all worlds.

    Renders the first person camera of each drone across all worlds in a single vmapped call. Scene
    gaussians are rendered at their static pose, drone gaussians follow the drone poses.

    Args:
        sim: The simulation to render.
        drones: Drones whose fpv cameras are rendered. ``None`` renders every drone. A single int
            renders one drone and drops the drone axis from the output. A sequence renders that
            subset in order.
        resolution: Image resolution as (width, height).
        background: RGB background color with values in [0, 1].
        exclude_self: If True, hide each drone's own splat from its own camera so a drone sees the
            others but not itself.
        camera_prefix: Camera name prefix, resolved to ``{camera_prefix}:{drone}`` for each drone.

    Returns:
        RGB images with values in [0, 1]. Shape (n_worlds, n_drones, height, width, 3), or
        (n_worlds, height, width, 3) when ``drones`` selects a single drone.
    """
    drone_ids, single = _resolve_drones(sim, drones)
    cameras = [_camera(sim.mj_model, camera_prefix, d) for d in drone_ids]
    f, c = camera_intrinsics(sim.mj_model, cameras[0], resolution)
    arrays = tuple(sim.data.plugins[key] for key in SPLAT_KEYS)
    slices = tuple((int(start), int(stop)) for start, stop in sim.data.plugins[SPLAT_SLICES_KEY])
    img = _render_splats(
        *arrays,
        cam_xpos=sim.mjx_data.cam_xpos[:, cameras],
        cam_xmat=sim.mjx_data.cam_xmat[:, cameras],
        pos=sim.data.states.pos,
        quat=sim.data.states.quat,
        slices=slices,
        img_shape=(resolution[1], resolution[0]),
        f=f,
        c=c,
        background=background,
        exclude=drone_ids if exclude_self else None,
    )
    return img[:, 0] if single else img


def _resolve_drones(sim: Sim, drones: int | Sequence[int] | None) -> tuple[tuple[int, ...], bool]:
    """Normalize a drone selection to a tuple of indices and whether to squeeze the drone axis."""
    if isinstance(drones, (int, np.integer)):
        return (int(drones),), True
    ids = range(sim.n_drones) if drones is None else drones
    return tuple(int(d) for d in ids), False


def _camera(mj_model: mujoco.MjModel, prefix: str, drone: int) -> int:
    """Camera index of a drone for the given camera name prefix."""
    name = f"{prefix}:{drone}"
    camera = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if camera < 0:
        raise ValueError(f"Camera '{name}' not found in the model")
    return camera


@requires_splats
@requires_gpu
def build_render_splat_fn(
    sim: Sim,
    drones: int | Sequence[int] | None = None,
    resolution: tuple[int, int] = (640, 480),
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    exclude_self: bool = False,
    camera_prefix: str = "fpv_cam",
) -> Callable[[Sim], Array]:
    """Build a splat renderer function for a given drone selection, camera prefix, and resolution.

    Mirrors :func:`crazyflow.sim.sensors.depth.build_render_depth_fn`: the camera intrinsics and
    the static slice metadata are baked into the returned function, which takes a ``Sim`` object
    and returns RGB images shaped like :func:`render_splat_rgb`'s output for the same ``drones``.
    Improves performance compared to :func:`crazyflow.sim.sensors.splat.render_splat_rgb`. Requires
    splats to have been attached via :func:`crazyflow.sim.splat.attach_splats`.
    """
    drone_ids, single = _resolve_drones(sim, drones)
    cameras = [_camera(sim.mj_model, camera_prefix, d) for d in drone_ids]
    f, c = camera_intrinsics(sim.mj_model, cameras[0], resolution)
    slices = tuple((int(start), int(stop)) for start, stop in sim.data.plugins[SPLAT_SLICES_KEY])
    render_fn = partial(
        _render_splats,
        slices=slices,
        img_shape=(resolution[1], resolution[0]),
        f=f,
        c=c,
        background=background,
        exclude=drone_ids if exclude_self else None,
    )

    @requires_mujoco_sync
    def render_splat_fn(sim: Sim) -> Array:
        arrays = tuple(sim.data.plugins[key] for key in SPLAT_KEYS)
        img = render_fn(
            *arrays,
            cam_xpos=sim.mjx_data.cam_xpos[:, cameras],
            cam_xmat=sim.mjx_data.cam_xmat[:, cameras],
            pos=sim.data.states.pos,
            quat=sim.data.states.quat,
        )
        return img[:, 0] if single else img

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


@jax.jit(static_argnames=("slices", "img_shape", "f", "c", "background", "exclude"))
def _render_splats(
    means: Array,
    log_scales: Array,
    quats: Array,
    sh_colors: Array,
    logit_opacities: Array,
    cam_xpos: Array,
    cam_xmat: Array,
    pos: Array,
    quat: Array,
    slices: tuple[tuple[int, int], ...],
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    background: tuple[float, float, float],
    exclude: tuple[int, ...] | None,
) -> Array:
    """Render (n_worlds, n_cams, height, width, 3) with each drone at its current pose.

    ``cam_xpos`` is (n_worlds, n_cams, 3) and ``cam_xmat`` (n_worlds, n_cams, 3, 3). ``exclude``, if
    given, holds the drone index culled from each camera, hiding a drone from its own view.
    """
    # viewmats needs a single leading batch axis, the render is then nested-vmapped over both axes.
    vm = viewmats(cam_xpos.reshape(-1, 3), cam_xmat.reshape(-1, 3, 3))
    vm = vm.reshape(*cam_xpos.shape[:2], 4, 4)
    bg = jnp.asarray(background, dtype=means.dtype)
    render = partial(
        splax.render,
        means,
        log_scales,
        quats,
        sh_colors,
        logit_opacities,
        background=bg,
        img_shape=img_shape,
        f=f,
        c=c,
    )
    if not slices:
        return jax.vmap(jax.vmap(lambda v: render(viewmat=v)[0]))(vm)
    tfs = RigidTransform.from_components(pos, R.from_quat(quat)).as_matrix()  # (n_worlds, n_drones)
    cam_axis = None
    if exclude is not None:
        # Slices are static, so teleport each camera's own drone far below the scene to cull it.
        far = jnp.eye(4, dtype=tfs.dtype).at[2, 3].set(-1e4)
        n_cams = cam_xpos.shape[1]
        tfs = jnp.broadcast_to(tfs[:, None], (tfs.shape[0], n_cams, *tfs.shape[1:]))
        tfs = tfs.at[:, jnp.arange(n_cams), jnp.asarray(exclude)].set(far)
        cam_axis = 0
    render_cam = lambda v, t: render(viewmat=v, gaussian_transforms=t, gaussian_slices=slices)[0]  # noqa: E731
    return jax.vmap(jax.vmap(render_cam, in_axes=(0, cam_axis)), in_axes=(0, 0))(vm, tfs)
