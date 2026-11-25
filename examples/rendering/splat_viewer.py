"""Show the simulation as gaussian splats in the interactive viewer.

The viewer shows attached splats and moves each drone's splat with the drone. Requires splax
(`pip install "crazyflow[splats]"`).
"""

from __future__ import annotations

import logging
import os

os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np
from scipy.spatial.transform import Rotation as R
from splax.io import fetch

from crazyflow.sim import Sim
from crazyflow.sim.splat import attach_splats

logger = logging.getLogger(__name__)
ASSETS_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main"


def trajectory(t: float) -> np.ndarray:
    cmd = np.zeros((1, 1, 16))
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    cmd[..., :3] = [0.5 * (np.cos(t) - 1), 0.5 * np.sin(t), 1.0 + 0.2 * np.sin(0.5 * t)]
    return cmd


def main():
    sim = Sim(control="state")
    scene = fetch(f"{ASSETS_URL}/robot_hall.ply")
    drone = fetch(f"{ASSETS_URL}/{sim.drone}.ply")
    attach_splats(sim, scene=scene, drone=drone)
    fps = 50

    for i in range(int(10 * sim.control_freq)):
        sim.state_control(trajectory(i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    main()
