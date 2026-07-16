"""Benchmark gaussian splat rendering throughput inside a jax.lax.scan rollout.

Loads a splat scene and a drone splat, then renders RGB images from the drone camera for a fixed
number of frames inside a single scanned rollout, doubling the number of parallel worlds each run
(1, 2, 4, 8, ...). The whole rollout is jitted, so the loop runs entirely on device and the frame
count reported is the number of images XLA actually rasterizes.

Requires splax and a CUDA-capable GPU because the splat camera sensor uses splax's GPU rasterizer.

Run with::

    pixi run -e benchmark python benchmark/splat.py --resolution "(64,64)" --n_frames 100
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

# splax rasterizes with warp, which needs GPU memory outside JAX's pool. Disable JAX preallocation
# before it initializes so both share the device. Must run before the first jax import.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import fire
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax.errors import JaxRuntimeError
from splax.io import fetch

from crazyflow.sim import Sim
from crazyflow.sim.sensors.splat import _render_splats, camera_intrinsics
from crazyflow.sim.sim import sync_sim2mjx
from crazyflow.sim.splat import SPLAT_KEYS, SPLAT_SLICES_KEY, attach_splats

if TYPE_CHECKING:
    from jax import Array
    from mujoco.mjx import Data

    from crazyflow.sim.data import SimData

ASSETS_URL = "https://huggingface.co/datasets/amacati/splats/resolve/main"


def build_rollout(
    sim: Sim, camera: int, resolution: tuple[int, int], n_frames: int, steps_per_frame: int
) -> Callable[[SimData, Data], Array]:
    """Build a jitted rollout that steps the sim and renders one image per frame.

    The static splat buffers and camera intrinsics are closed over. Each frame advances the
    simulation, syncs the camera pose, rasterizes the splats for every world, and reduces the image
    to a scalar sum. Reducing inside the loop keeps XLA from eliminating the render as dead code
    while avoiding materializing the full (n_frames, n_worlds, H, W, 3) stack.
    """
    step_fn = sim.build_step_fn()
    mjx_model = sim.mjx_model
    f, c = camera_intrinsics(sim.mj_model, camera, resolution)
    slices = tuple((int(start), int(stop)) for start, stop in sim.data.plugins[SPLAT_SLICES_KEY])
    gaussians = tuple(sim.data.plugins[key] for key in SPLAT_KEYS)
    img_shape = (resolution[1], resolution[0])

    def render(data: SimData, mjx_data: Data) -> Array:
        return _render_splats(
            *gaussians,
            cam_xpos=mjx_data.cam_xpos[:, [camera]],
            cam_xmat=mjx_data.cam_xmat[:, [camera]],
            pos=data.states.pos,
            quat=data.states.quat,
            slices=slices,
            img_shape=img_shape,
            f=f,
            c=c,
            background=(0.0, 0.0, 0.0),
            exclude=None,
        )

    @jax.jit
    def rollout(data: SimData, mjx_data: Data) -> Array:
        def frame(carry: tuple[SimData, Data], _: None) -> tuple[tuple[SimData, Data], Array]:
            data, mjx_data = carry
            data = step_fn(data, n_steps=steps_per_frame)
            data, mjx_data = sync_sim2mjx(data, mjx_data, mjx_model)
            img = render(data, mjx_data)
            return (data, mjx_data), img.sum()

        (data, mjx_data), sums = jax.lax.scan(frame, (data, mjx_data), length=n_frames)
        return sums.sum()

    return rollout


def benchmark(
    resolution: tuple[int, int] = (64, 64),
    n_frames: int = 100,
    max_worlds_exp: int = 12,
    fps: int = 30,
    n_repeats: int = 3,
    scene_ply: str = "robot_hall.ply",
    drone_ply: str = "cf21B_500.ply",
):
    """Benchmark splat rendering throughput for a growing number of parallel worlds.

    Args:
        resolution: Rendered image resolution as (width, height).
        n_frames: Number of frames rendered per scanned rollout.
        max_worlds_exp: Largest world count is ``2 ** max_worlds_exp``.
        fps: Camera frame rate. Determines the physics steps taken between frames.
        n_repeats: Number of timed rollouts per world count. The fastest run is reported.
        scene_ply: Scene splat file name on the assets host.
        drone_ply: Drone splat file name on the assets host.
    """
    logging.info("Fetching splat assets...")
    scene = fetch(f"{ASSETS_URL}/{scene_ply}")
    drone = fetch(f"{ASSETS_URL}/{drone_ply}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = Path(__file__).parent / "data" / f"benchmark_results_{timestamp}.csv"
    csv_file.parent.mkdir(exist_ok=True)
    with open(csv_file, "w", newline="") as f:
        f.write(
            "test_type,n_drones,n_worlds,n_steps,total_time_s,avg_step_time_s,"
            "fps,real_time_factor,device\n"
        )

    print(
        f"\nSplat rendering benchmark, resolution {resolution[0]}x{resolution[1]}, {n_frames} "
        f"frames per rollout"
    )
    print("-" * 80)
    print(f"{'n_worlds':>10} {'rollout_s':>12} {'frame_ms':>12} {'fps':>14}")
    print("-" * 80)

    for n_worlds in [2**i for i in range(max_worlds_exp + 1)]:
        try:
            sim = Sim(n_worlds=n_worlds, control="state", device="gpu")
            attach_splats(sim, scene=scene, drone=drone)
            steps_per_frame = max(1, sim.freq // fps)

            camera = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv_cam:0")
            if camera < 0:
                raise ValueError("Camera 'fpv_cam:0' not found in the model")

            # Hold a constant target so the drone keeps moving and each frame renders a distinct
            # pose. A static scene would let XLA hoist the render out of the loop.
            cmd = np.zeros((sim.n_worlds, sim.n_drones, 13), dtype=np.float32)
            cmd[..., 2] = 0.5
            sim.reset()
            sim.state_control(jnp.asarray(cmd, device=sim.device))

            rollout = build_rollout(sim, camera, resolution, n_frames, steps_per_frame)

            # Warmup triggers JIT compilation of the full rollout.
            jax.block_until_ready(rollout(sim.data, sim.mjx_data))

            times = []
            for _ in range(n_repeats):
                tstart = time.perf_counter()
                jax.block_until_ready(rollout(sim.data, sim.mjx_data))
                times.append(time.perf_counter() - tstart)

            assert rollout._cache_size() == 1, "rollout must only be jitted once"

            rollout_s = min(times)
            frame_ms = rollout_s / n_frames * 1e3
            images_per_s = n_frames * n_worlds / rollout_s
            print(f"{n_worlds:>10} {rollout_s:>12.4f} {frame_ms:>12.4f} {images_per_s:>14.3e}")
            sim.close()

            real_time_factor = (n_frames / fps) * n_worlds / rollout_s
            with open(csv_file, "a", newline="") as f:
                f.write(
                    f"splat,{sim.n_drones},{n_worlds},{n_frames},{rollout_s},"
                    f"{rollout_s / n_frames},{images_per_s},{real_time_factor},gpu\n"
                )
        except (JaxRuntimeError, MemoryError) as e:
            print(f"{n_worlds:>10}   out of memory, stopping ({type(e).__name__})")
            break

    print("-" * 80)
    print("fps is total images rendered per second across all parallel worlds.\n")
    print(f"Benchmark results saved to {csv_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("jax").setLevel(logging.WARNING)
    fire.Fire(benchmark)
