"""Render the simulation as gaussian splats in a web-based viewer.

Requires splax (`pip install crazyflow[splats]`) and gaussian splat .ply files aligned to the
simulation frames. The demo splats are downloaded from the crazyflow GitHub release assets on first
use and cached locally.

The viewer streams the splats to the browser once and then only updates the drone poses, so it
runs in real time on any device. The splat camera sensor renders with CUDA kernels and therefore
requires a GPU.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from splax import fetch

from crazyflow.sim import Sim
from crazyflow.sim.splat import SplatViewer, attach_splats

logger = logging.getLogger(__name__)
ASSETS_URL = "https://github.com/learnsyslab/crazyflow/releases/download/assets-v1"


def control(t: float) -> np.ndarray:
    cmd = np.zeros((1, 1, 13))
    cmd[..., :3] = [0.5 * (np.cos(t) - 1), 0.5 * np.sin(t), 0.3 + 0.2 * np.sin(0.5 * t)]
    return cmd


def main():
    scene = fetch(f"{ASSETS_URL}/hall.ply")
    drone = fetch(f"{ASSETS_URL}/drone.ply")
    sim = Sim(control="state")
    attach_splats(sim, scene=scene, drone=drone)
    viewer = SplatViewer(sim)

    for i in range(2000):
        t_start = time.perf_counter()
        sim.state_control(control(i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        viewer.update(sim)
        time.sleep(max(0.0, 1 / sim.control_freq - (time.perf_counter() - t_start)))

    viewer.close()
    sim.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    main()
