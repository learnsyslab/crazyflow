"""Render the FPV cameras of two drones looking at each other.

`build_render_rgbd_fn` compiles an RGB-D renderer that renders every drone in every world in a
single batched call. Requires warp (`pip install "crazyflow[rgbd]"`).

Run with `--plot` to plot the results, which requires matplotlib from the `tests` environment.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import TYPE_CHECKING

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim
from crazyflow.sim.sensors.rgbd import build_render_rgbd_fn

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def trajectory(t: float, n_worlds: int, n_drones: int) -> np.ndarray:
    """Circle both drones around the center."""
    angle = t + (2 * np.pi / n_drones) * np.arange(n_drones)
    cmd = np.zeros((n_worlds, n_drones, 16))
    cmd[..., 0] = 0.25 * np.cos(angle)
    cmd[..., 1] = 0.25 * np.sin(angle)
    cmd[..., 2] = 1.0
    yaw = angle + np.pi  # yaw faces the opposite point on the circle
    cmd[..., 9:13] = R.from_euler("z", yaw[:, None]).as_quat()
    return cmd


def main(plot: bool = False):
    """Render the RGB-D camera views of both drones."""
    sim = Sim(n_drones=2, control="state", device=jax.default_backend())
    resolution, fps = (320, 240), 50
    render = build_render_rgbd_fn(sim, resolution=resolution, background=(0.5, 0.5, 0.5))

    # Start on the circle so the controller does not have to fly in from the spawn points first
    cmd = trajectory(0.0, sim.n_worlds, sim.n_drones)
    states = sim.data.states.replace(pos=sim.data.states.pos.at[...].set(cmd[..., :3]))
    states = states.replace(quat=states.quat.at[...].set(cmd[..., 9:13]))
    sim.data = sim.data.replace(states=states)

    update_plot = plot_views(sim.n_drones, resolution) if plot else None
    for i in range(6 * fps):
        sim.state_control(trajectory(i / fps, sim.n_worlds, sim.n_drones))
        sim.step(sim.freq // fps)
        rgbd = render(sim.data)
        if plot:
            update_plot(rgbd[0])
    sim.close()


def plot_views(n_drones: int, resolution: tuple[int, int]) -> Callable[[NDArray], None]:
    """Open a window for the camera views and return a function that draws the next ones."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_drones, 2, figsize=(10, 7))
    ims = np.empty((n_drones, 2), dtype=object)
    for d in range(n_drones):
        ims[d, 0] = axes[d, 0].imshow(np.zeros((*resolution[::-1], 3)))
        ims[d, 1] = axes[d, 1].imshow(np.zeros(resolution[::-1]), cmap="turbo_r", vmin=0, vmax=3)
        axes[d, 0].set_title(f"Drone {d} RGB")
        axes[d, 1].set_title(f"Drone {d} depth")
        for ax in axes[d]:
            ax.axis("off")
    fig.colorbar(ims[0, 1], ax=axes[:, 1], label="depth along the optical axis (m)")
    plt.show(block=False)

    def update(rgbd: NDArray) -> None:
        for d in range(n_drones):
            ims[d, 0].set_data(rgbd[d, ..., :3])
            ims[d, 1].set_data(np.ma.masked_equal(rgbd[d, ..., 3], 0))  # 0 where nothing is hit
        fig.canvas.draw_idle()
        plt.pause(1e-12)

    return update


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    main(plot=parser.parse_args().plot)
