"""Animate gaussian splat rendering from two drones looking at each other.

Requires splax and a CUDA-capable GPU because the splat camera sensor uses splax's GPU rasterizer.
"""

from __future__ import annotations

import logging
import os

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from splax.io import fetch

from crazyflow.sim import Sim
from crazyflow.sim.sensors.splat import render_splat_rgb
from crazyflow.sim.splat import attach_splats

ASSETS_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main"

RADIUS = 0.5
HEIGHT = 1.0


def control(t: float, n_worlds: int, n_drones: int) -> np.ndarray:
    """Circle both drones around the center, each yawed to look across at its partner."""
    drones = np.arange(n_drones)[None, :]
    angle = t + (2 * np.pi / n_drones) * drones
    cmd = np.zeros((n_worlds, n_drones, 13))
    cmd[..., 0] = RADIUS * np.cos(angle)
    cmd[..., 1] = RADIUS * np.sin(angle)
    cmd[..., 2] = HEIGHT
    cmd[..., 9] = angle + np.pi  # yaw faces the opposite point on the circle, where the partner is
    return cmd


def main(show_plot: bool = False):
    """Render four splat camera views into a matplotlib window."""
    sim = Sim(n_worlds=2, n_drones=2, control="state", device="gpu")
    scene = fetch(f"{ASSETS_URL}/robot_hall.ply")
    drone = fetch(f"{ASSETS_URL}/{sim.drone}.ply")
    attach_splats(sim, scene=scene, drone=drone)

    angle = (2 * np.pi / sim.n_drones) * np.arange(sim.n_drones)
    x, y, z = RADIUS * np.cos(angle), RADIUS * np.sin(angle), np.full_like(angle, HEIGHT)
    pos = np.stack([x, y, z], axis=-1)
    quat = R.from_euler("z", (angle + np.pi)[:, None]).as_quat()
    states = sim.data.states.replace(pos=sim.data.states.pos.at[...].set(pos))
    states = states.replace(quat=sim.data.states.quat.at[...].set(quat))
    sim.data = sim.data.replace(states=states)

    duration = 6
    fps = 30
    n_frames = int(duration * fps)
    resolution = (320, 240)

    fig, axes = plt.subplots(sim.n_worlds, sim.n_drones, figsize=(10, 7))
    img = np.zeros((resolution[1], resolution[0], 3), dtype=np.float32)
    ims = np.empty((sim.n_worlds, sim.n_drones), dtype=object)
    for w in range(sim.n_worlds):
        for d in range(sim.n_drones):
            ax = axes[w, d]
            ims[w, d] = ax.imshow(img)
            ax.set_title(f"World {w}, drone {d} FPV")
            ax.axis("off")
    fig.tight_layout()

    if show_plot:
        plt.show(block=False)
        for _ in range(n_frames):
            t = sim.data.core.steps[0, 0] / sim.freq
            sim.state_control(control(t, sim.n_worlds, sim.n_drones))
            sim.step(sim.freq // fps)
            # Render each drone's fpv across all worlds, hiding each drone from its own view.
            views = np.asarray(render_splat_rgb(sim, resolution=resolution))
            for w in range(sim.n_worlds):
                for d in range(sim.n_drones):
                    ims[w, d].set_data(views[w, d])
            fig.canvas.draw_idle()
            plt.pause(1e-12)
        plt.close(fig)
    sim.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    main(show_plot=True)
