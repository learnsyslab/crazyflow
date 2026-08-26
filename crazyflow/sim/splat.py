"""Gaussian splat support built on [splax](https://github.com/learnsyslab/splax).

This module owns the splat plugin state and its visualization.
[attach_splats][crazyflow.sim.splat.attach_splats] loads 3D gaussian splatting ``.ply`` files and
stores them in the simulation's plugin data, and [SplatViewer][crazyflow.sim.splat.SplatViewer]
streams the splats to a web-based viewer.

The batched RGB camera sensor that renders splats lives in [crazyflow.sim.sensors.splat][].

Splat files must be aligned to the simulation frames. The viewer renders in the browser and works on
any device. Camera sensors rasterize on the GPU only.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import splax
from splax.viewer import Viewer

from crazyflow.sim.sim import requires_mujoco_sync

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from crazyflow.sim.sim import Sim

Params = ParamSpec("Params")
Return = TypeVar("Return")

SPLAT_KEYS = (
    "splat_means",
    "splat_log_scales",
    "splat_quats",
    "splat_sh_colors",
    "splat_logit_opacities",
)
"""Keys under which the splat arrays are stored in ``sim.data.plugins``.

The gaussians form a single buffer per array, laid out as ``[scene | drone:0 | ... | drone:n-1]``.
Scene gaussians are not part of any slice and remain static. Each drone's slice follows that drone's
world pose.
"""

SPLAT_SLICES_KEY = "splat_slices"
"""Key of the (n_drones, 2) array of per-drone (start, stop) gaussian buffer slices."""


def requires_splats(fn: Callable[Params, Return]) -> Callable[Params, Return]:
    """Decorator to ensure that the simulation has splats attached."""

    @wraps(fn)
    def wrapper(sim: Sim, *args: Any, **kwargs: Any) -> Return:
        if not all(k in sim.data.plugins for k in SPLAT_KEYS):
            raise RuntimeError("No splats attached to this simulation, call attach_splats first")
        return fn(sim, *args, **kwargs)

    return wrapper


def requires_gpu(fn: Callable[Params, Return]) -> Callable[Params, Return]:
    """Decorator to ensure that the simulation is running on the GPU."""

    @wraps(fn)
    def wrapper(sim: Sim, *args: Any, **kwargs: Any) -> Return:
        if sim.device.platform != "gpu":
            raise RuntimeError("Gaussian splatting requires running on the GPU.")
        return fn(sim, *args, **kwargs)

    return wrapper


def attach_splats(sim: Sim, scene: Path | None = None, drone: Path | None = None):
    """Load gaussian splat ``.ply`` files and attach them to the simulation.

    The drone splat is replicated once per drone so that each drone has its own slice of gaussians.
    All splat state, including the drone slice metadata, is inserted into ``sim.data.plugins`` and
    the default data is rebuilt so it survives resets.

    Args:
        sim: The simulation to attach the splats to.
        scene: Path to a static scene splat, aligned to the MuJoCo world frame.
        drone: Path to a drone splat in the drone body frame, replicated for each drone.
    """
    if scene is None and drone is None:
        raise ValueError("At least one of scene or drone must be provided")
    parts, n_splats, slices = [], 0, ()
    if scene is not None:
        scene_arrays = splax.io.load_ply(scene)
        n_splats = scene_arrays[0].shape[0]
        parts.append(scene_arrays)
    if drone is not None:
        drone_arrays = splax.io.load_ply(drone)
        n_drone_splats = drone_arrays[0].shape[0]
        parts.extend([drone_arrays] * sim.n_drones)
        starts = [n_splats + i * n_drone_splats for i in range(sim.n_drones)]
        slices = tuple((start, start + n_drone_splats) for start in starts)
    coefficients = {part[3].shape[1] for part in parts}
    assert len(coefficients) == 1, f"Splats must share their SH degree, got {coefficients=}"
    arrays = [jnp.concatenate(x, axis=0) for x in zip(*parts)]
    splat_data = dict(zip(SPLAT_KEYS, arrays))
    splat_data[SPLAT_SLICES_KEY] = jnp.asarray(slices, dtype=jnp.int32).reshape(-1, 2)
    splat_data = jax.device_put(splat_data, sim.device)
    sim.data = sim.data.replace(plugins=sim.data.plugins | splat_data)
    sim.build_default_data()


class SplatViewer:
    """Web-based gaussian splat viewer.

    Starts a ``splax.viewer.Viewer`` (viser web server) and uploads all attached splats once.
    [update][crazyflow.sim.splat.SplatViewer.update] then only pushes the current drone poses, so
    the viewer runs at real-time rates on any device. The viewer is owned by its creator and is
    independent of ``sim.render()``.

    Args:
        sim: The simulation to visualize. Requires
            [attach_splats][crazyflow.sim.splat.attach_splats] to have been called.
        port: Port of the web server.
    """

    def __init__(self, sim: Sim, port: int = 8080):
        if not all(k in sim.data.plugins for k in SPLAT_KEYS):
            raise RuntimeError("No splats attached to this simulation, call attach_splats first")
        arrays = tuple(sim.data.plugins[key] for key in SPLAT_KEYS)
        slices = tuple((start, stop) for start, stop in sim.data.plugins[SPLAT_SLICES_KEY])
        self.viewer = Viewer(port=port)
        n_splats = slices[0][0] if slices else arrays[0].shape[0]
        if n_splats > 0:
            self.viewer.add_splats("scene", *(x[:n_splats] for x in arrays))
        for i, (start, stop) in enumerate(slices):
            self.viewer.add_splats(f"drone:{i}", *(x[start:stop] for x in arrays))
        self._n_drones = len(slices)

    def update(self, sim: Sim, world: int = 0):
        """Push the current drone poses of one world to the viewer.

        Args:
            sim: The simulation to visualize.
            world: Index of the world whose drone poses are shown.
        """
        if self._n_drones == 0:
            return
        pos, quat = self._drone_mocap_poses(sim, world)
        for i in range(self._n_drones):
            self.viewer.update_pose(f"drone:{i}", pos[i], quat[i])

    def close(self):
        """Shut down the web server."""
        self.viewer.close()

    @staticmethod
    @requires_mujoco_sync
    def _drone_mocap_poses(sim: Sim, world: int) -> tuple[NDArray, NDArray]:
        """Drone mocap positions and wxyz quaternions of one world, synced with the MuJoCo data."""
        ids = sim.data.core.drone_mocap_ids
        pos = np.asarray(sim.mjx_data.mocap_pos[world, ids])
        quat = np.asarray(sim.mjx_data.mocap_quat[world, ids])  # MuJoCo quats are already wxyz
        return pos, quat
