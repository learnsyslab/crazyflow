"""Gaussian splat support built on [splax](https://github.com/learnsyslab/splax).

This module contains the splat plugin state. [attach_splats][crazyflow.sim.splat.attach_splats]
loads 3D gaussian splatting ``.ply`` files and stores them in the simulation's plugin data. The
interactive viewer of ``sim.render()`` shows attached splats.

The batched RGB camera sensor that renders splats lives in [crazyflow.sim.sensors.splat][].

Splat files must be aligned to the simulation frames. Camera sensors rasterize significantly faster
on the GPU.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import splax
import warp as wp
from flax.struct import dataclass, field

from crazyflow.utils import CORE_NDIM_KEY

if TYPE_CHECKING:
    from pathlib import Path

    from jax import Array

    from crazyflow.sim.sim import Sim

# splax's tiled kernels need cooperative blocks to render correctly on the CPU
wp.config.enable_cpu_blocks = True

Params = ParamSpec("Params")
Return = TypeVar("Return")

SPLATS_KEY = "splats"
"""Key of the [SplatData][crazyflow.sim.splat.SplatData] in ``sim.data.plugins``."""


@dataclass
class SplatData:
    """Gaussian splat plugin data.

    Gaussians are stored in a single array as ``[scene | 0:drone | ... | n-1:drone]``. Scene
    gaussians are not part of any slice and remain static. Drone slices follow the drone's pose.
    """

    means: Array = field(metadata={CORE_NDIM_KEY: 2})
    """Gaussian centers, shape (N, 3)."""
    log_scales: Array = field(metadata={CORE_NDIM_KEY: 2})
    """Log of the per-axis scales, shape (N, 3)."""
    quats: Array = field(metadata={CORE_NDIM_KEY: 2})
    """Rotations as wxyz quaternions, shape (N, 4)."""
    sh_colors: Array = field(metadata={CORE_NDIM_KEY: 3})
    """Spherical harmonics color coefficients, shape (N, K, 3)."""
    logit_opacities: Array = field(metadata={CORE_NDIM_KEY: 1})
    """Opacity logits, shape (N,)."""
    slices: tuple[tuple[int, int], ...] = field(pytree_node=False)
    """Start and stop index of each drone's gaussians."""

    @property
    def params(self) -> tuple[Array, Array, Array, Array, Array]:
        """The gaussian parameter arrays in splax's argument order."""
        return self.means, self.log_scales, self.quats, self.sh_colors, self.logit_opacities


def requires_splats(fn: Callable[Params, Return]) -> Callable[Params, Return]:
    """Decorator to ensure that the simulation has splats attached."""

    @wraps(fn)
    def wrapper(sim: Sim, *args: Any, **kwargs: Any) -> Return:
        if SPLATS_KEY not in sim.data.plugins:
            raise RuntimeError("No splats attached to this simulation, call attach_splats first")
        return fn(sim, *args, **kwargs)

    return wrapper


def attach_splats(sim: Sim, scene: Path | None = None, drone: Path | None = None):
    """Load gaussian splat ``.ply`` files and attach them to the simulation.

    The drone splat is replicated once per drone so that each drone has its own slice of gaussians.
    The splat data is inserted into ``sim.data.plugins`` and the default data is rebuilt so it
    survives resets.

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
    splats = jax.device_put(SplatData(*arrays, slices=slices), sim.device)
    sim.data = sim.data.replace(plugins=sim.data.plugins | {SPLATS_KEY: splats})
    sim.build_default_data()
