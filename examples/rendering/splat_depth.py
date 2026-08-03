"""Animate the splat depth camera of a drone flying an elliptical lap through the flight hall.

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
from crazyflow.sim.sensors.splat import build_render_splat_rgbd_fn
from crazyflow.sim.splat import attach_splats

ASSETS_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main"

# Elliptical lap, inset from the hall walls so the camera keeps them in view and clears them
CENTER = np.array([0.0, 0.0])
RADII = np.array([9.0, 3.5])
HEIGHT = 1.5
MAX_RANGE = 12.0
RESOLUTION = (320, 240)
DURATION = 40
FPS = 30


def control(t: float) -> np.ndarray:
    """State setpoint a fraction ``t`` into one lap of the ellipse.

    Args:
        t: Progress along the lap, wrapped into [0, 1).

    Returns:
        A state command placing the drone on the ellipse with its yaw along the tangent.
    """
    angle = 2 * np.pi * t
    cmd = np.zeros((1, 1, 13))
    cmd[..., :2] = CENTER + RADII * np.array([np.cos(angle), np.sin(angle)])
    cmd[..., 2] = HEIGHT
    cmd[..., 9] = np.arctan2(RADII[1] * np.cos(angle), -RADII[0] * np.sin(angle))
    return cmd


def main(show_plot: bool = False):
    """Fly the lap and animate the depth channel of the drone's fpv camera."""
    sim = Sim(control="state", device="gpu")
    scene = fetch(f"{ASSETS_URL}/robot_hall.ply")
    drone = fetch(f"{ASSETS_URL}/{sim.drone}.ply")
    attach_splats(sim, scene=scene, drone=drone)
    render = build_render_splat_rgbd_fn(sim, drones=0, resolution=RESOLUTION, max_range=MAX_RANGE)

    # Start on the lap so the controller does not have to fly in from the origin first
    cmd = control(0.0)
    states = sim.data.states.replace(pos=sim.data.states.pos.at[..., :].set(cmd[..., :3]))
    states = states.replace(quat=states.quat.at[...].set(R.from_euler("z", cmd[..., 9]).as_quat()))
    sim.data = sim.data.replace(states=states)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.zeros(RESOLUTION[::-1]), cmap="turbo_r", vmin=0.0, vmax=MAX_RANGE)
    ax.set_title("Splat depth camera, fpv")
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="depth along the optical axis (m)")
    fig.tight_layout()

    if show_plot:
        plt.show(block=False)
        for i in range(DURATION * FPS):
            sim.state_control(control(i / (DURATION * FPS)))
            sim.step(sim.freq // FPS)
            im.set_data(np.asarray(render(sim.data))[0, 0, ..., 3])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        plt.close(fig)
    sim.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    main(show_plot=True)
