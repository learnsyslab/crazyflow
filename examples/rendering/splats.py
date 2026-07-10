"""Animate gaussian splat rendering with matplotlib.

Requires splax and a CUDA-capable GPU because the splat camera sensor uses splax's GPU rasterizer.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from splax.io import fetch

from crazyflow.sim import Sim
from crazyflow.sim.sensors.splat import render_splat_rgb
from crazyflow.sim.splat import attach_splats

ASSETS_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main"


def control(t: float) -> np.ndarray:
    cmd = np.zeros((1, 1, 13))
    cmd[..., :3] = [0.5 * (np.cos(t) - 1), 0.5 * np.sin(t), 0.3 + 0.2 * np.sin(0.5 * t)]
    return cmd


def main(show_plot: bool = False):
    """Render splats into a matplotlib window."""
    sim = Sim(control="state", device="gpu")
    scene = fetch(f"{ASSETS_URL}/robot_hall.ply")
    drone = fetch(f"{ASSETS_URL}/{sim.drone}.ply")
    attach_splats(sim, scene=scene, drone=drone)

    duration = 3
    fps = 30
    n_frames = int(duration * fps)
    resolution = (320, 240)
    camera = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv_cam:0")
    if camera < 0:
        raise ValueError("Camera 'fpv_cam:0' not found in the model")

    fig, ax = plt.subplots(figsize=(8, 6))
    img = np.zeros((resolution[1], resolution[0], 3), dtype=np.float32)
    im = ax.imshow(img)
    ax.set_title("Gaussian Splats")
    ax.axis("off")
    fig.tight_layout()

    if show_plot:
        plt.show(block=False)
        for _ in range(n_frames):
            t = sim.data.core.steps[0, 0] / sim.freq
            sim.state_control(control(t))
            sim.step(sim.freq // fps)
            rgb = np.asarray(render_splat_rgb(sim, camera=camera, resolution=resolution))[0]
            im.set_data(rgb)
            fig.canvas.draw_idle()
            plt.pause(1e-12)
        plt.close(fig)
    sim.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    main(show_plot=True)
