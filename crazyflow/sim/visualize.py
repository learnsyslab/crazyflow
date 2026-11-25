import numpy as np
from numpy.typing import NDArray

from crazyflow.sim import Sim


def draw_line(sim: Sim, points: NDArray, rgba: NDArray | None = None, thickness: float = 0.01):
    """Draw a line into the simulation.

    Args:
        sim: The simulation.
        points: An array of [N, 3] points that make up the line.
        rgba: The color of the line. Lines ignore the opacity.
        thickness: The thickness of the line in meters.

    Note:
        This function has to be called every time before the sim.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    sim.viewer.add_lines(np.stack([points[:-1], points[1:]], axis=1), rgba, thickness)


def draw_points(sim: Sim, points: NDArray, rgba: NDArray | None = None, size: float = 0.01):
    """Draw points into the simulation.

    Args:
        sim: The simulation.
        points: An array of [N, 3] points to draw.
        rgba: The color of the points.
        size: The radius of the points.
    """
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    points = np.atleast_2d(points)
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    sim.viewer.add_spheres(points, size, rgba)


def draw_capsule(
    sim: Sim,
    p1: NDArray,
    p2: NDArray,
    radius: float = 0.05,
    rgba: NDArray | None = None,
    cylinder: bool = False,
):
    """Draw a capsule (pill) or cylinder between two points.

    Args:
        sim: The simulation.
        p1: Start point [3,]
        p2: End point [3,]
        radius: The thickness of the geom in [m].
        rgba: The color of the object.
        cylinder: If True, draws a flat-ended cylinder. If False, draws a pill-shaped capsule.
    """
    if sim.viewer is None:
        return

    rgba = rgba if rgba is not None else np.array([1, 0, 0, 1.0])
    sim.viewer.add_cylinder(p1, p2, radius, rgba)
    if not cylinder:
        sim.viewer.add_spheres(np.stack([p1, p2]), radius, rgba)
