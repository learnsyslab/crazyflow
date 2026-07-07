# Gaussian Splat Rendering

Crazyflow renders photorealistic images with 3D gaussian splatting through [splax](https://github.com/amacati/splax). A web-based viewer visualizes the simulation as splats. A camera sensor renders batched RGB images across all worlds. Both combine a static scene splat with one splat per drone that follows the drone's pose.

<!-- TODO: Add image of the web viewer showing the hall splat with a drone splat mid-flight -->

## Installation

Install crazyflow with the `splats` extra. It pulls in splax with its viewer dependencies from GitHub.

```bash
pip install "crazyflow[splats]"
```

The web viewer runs on any device. The camera sensor rasterizes with CUDA kernels and requires an NVIDIA GPU.

## Demo assets

Demo splats of a flight hall and a Crazyflie drone are published as [release assets](https://github.com/learnsyslab/crazyflow/releases/tag/assets-v1). splax's `fetch` downloads a URL into a local cache on first use and returns the cached path afterwards. The cache lives at `~/.cache/splax` and can be moved with the `SPLAX_CACHE` environment variable. Passing `force=True` refreshes the cache.

<!-- notest: requires splax and network access -->
```{ .python notest }
from splax import fetch

assets = "https://github.com/learnsyslab/crazyflow/releases/download/assets-v1"
scene = fetch(f"{assets}/hall.ply")
drone = fetch(f"{assets}/drone.ply")
```

## Attaching splats

Splats are loaded from 3D gaussian splatting `.ply` files and stored in the simulation's plugin data (`sim.data.plugins`). They therefore persist across resets and travel with the sim data through `jax.jit`. The files must be aligned with the simulation frames. The scene splat is aligned with the MuJoCo world frame, with +z up and metric scale. The drone splat is aligned with the drone body frame and centered on its center of mass.

<!-- notest: requires splax and splat .ply files -->
```{ .python notest }
from crazyflow.sim import Sim
from crazyflow.sim.splat import attach_splats

sim = Sim(n_worlds=4, n_drones=2)
attach_splats(sim, scene=scene, drone=drone)
```

The drone splat is replicated once per drone. Both arguments are optional. Pass only `scene` for a static environment, or only `drone` for splats in an empty world.

## Web viewer

`SplatViewer` renders the simulation in a web-based viewer served at `http://localhost:8080`. Constructing the viewer streams the splats to the browser once. `update` then only pushes the current drone poses, so the viewer runs at real-time rates on any device. The viewer is independent of `sim.render()`. Both can be open at the same time, e.g. to compare the mesh scene and the splat scene side by side.

<!-- notest: requires splax and splat .ply files -->
```{ .python notest }
from crazyflow.sim.splat import SplatViewer

viewer = SplatViewer(sim)
for _ in range(500):
    sim.step(sim.freq // sim.control_freq)
    viewer.update(sim, world=0)
viewer.close()
```

See `examples/rendering/splat_viewer.py` for a complete viewer demo.

## Camera sensor

`render_splat_rgb` from `crazyflow.sim.sensors.splat` renders RGB images from any model camera, batched over all worlds. It uses splax's CUDA rasterizer, so the simulation must run on the GPU. Scene gaussians stay static. Each drone's gaussians follow its current pose via splax's dynamic transforms, without copying the splat buffer.

<!-- notest: requires splax and a CUDA GPU -->
```{ .python notest }
from crazyflow.sim.sensors.splat import build_render_splat_fn, render_splat_rgb

sim = Sim(n_worlds=4, device="gpu")
attach_splats(sim, scene=scene, drone=drone)
imgs = render_splat_rgb(sim, camera=0, resolution=(320, 240))  # (4, 240, 320, 3) in [0, 1]

# Bake camera intrinsics and splat metadata into a compiled function for better performance
render_fn = build_render_splat_fn(sim, camera=0, resolution=(320, 240))
imgs = render_fn(sim)
```

`exclude_drone` hides one drone's splat from the image, e.g. the drone carrying the camera. Depth images are not supported yet.

See `examples/rendering/splats.py` for a matplotlib-based camera sensor demo.
