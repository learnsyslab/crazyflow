# Rendering
Crazyflow supports interactive rendering in the browser, offscreen RGB-D capture through ray tracing, depth capture through raycasting in MJX and [Gaussian splatting](#gaussian-splat-setup) through [splax](https://github.com/learnsyslab/splax).

<div class="example-media-grid">
  <figure>
    <video src="../../img/examples/generated/rendering/splat_camera/plot-01.mp4#only-light" autoplay loop muted playsinline></video><video src="../../img/examples/generated/rendering/splat_camera/plot-01-dark.mp4#only-dark" autoplay loop muted playsinline></video>
    <figcaption>Gaussian splat cameras across worlds and drones</figcaption>
  </figure>
  <figure>
    <video src="../../img/examples/generated/rendering/splat_depth/plot-01.mp4#only-light" autoplay loop muted playsinline></video><video src="../../img/examples/generated/rendering/splat_depth/plot-01-dark.mp4#only-dark" autoplay loop muted playsinline></video>
    <figcaption>Gaussian splat depth</figcaption>
  </figure>
  <figure>
    <video src="../../img/examples/generated/rendering/cameras/plot-01.mp4#only-light" autoplay loop muted playsinline></video><video src="../../img/examples/generated/rendering/cameras/plot-01-dark.mp4#only-dark" autoplay loop muted playsinline></video>
    <figcaption>Viewer captures from a drone camera</figcaption>
  </figure>
  <figure>
    <video src="../../img/examples/generated/rendering/cam_config/video-02.mp4" autoplay loop muted playsinline></video>
    <figcaption>Interactive viewer with a configured camera</figcaption>
  </figure>
</div>

| | Viewer | RGB-D camera | MJX raycasting | Gaussian splats |
|---|---|---|---|---|
| Interactive viewer | `sim.render()` | – | – | `sim.render()` |
| RGB | `sim.render(capture=True)` | [`build_render_rgbd_fn`][crazyflow.sim.sensors.rgbd.build_render_rgbd_fn] | – | [`render_splat_rgb`][crazyflow.sim.sensors.splat.render_splat_rgb] |
| Depth | – | [`build_render_rgbd_fn`][crazyflow.sim.sensors.rgbd.build_render_rgbd_fn] | [`render_depth`][crazyflow.sim.sensors.depth.render_depth] | [`render_splat_rgbd`][crazyflow.sim.sensors.splat.render_splat_rgbd] |
| Batched over worlds | ❌ | ✅ | ✅ | ✅ |
| Jittable | ❌ | ✅ | ✅ | ✅ |
| Differentiable | ❌ | ❌ | ❌ | ✅ |

<details class="example" markdown>
<summary>Example</summary>

<!-- notest: requires rendering -->
```{ .python notest }
sim.render()                                   # interactive browser viewer
rgb = sim.render(capture=True)                 # numpy array (H, W, 3) of the viewer
sim.render(world=1)                            # show another world
sim.close()                                    # close the viewer
```

</details>

## Interactive viewer
[`sim.render()`][crazyflow.sim.Sim.render] opens an interactive viewer built on [viser](https://github.com/viser-org/viser) in the browser. It is decoupled from the dynamics step so that `sim.render()` can be called independently of how fast the simulation runs. You can switch between worlds and cameras in the viewer.

<figure class="example-media">
  <img src="../../img/examples/generated/rendering/cam_config/still-video-02.png" alt="Viewer with a configured free camera">
</figure>

If Gaussian splats are attached to the simulation, the viewer shows only the splats, with each drone splat following its drone. The Controls tab switches the meshes on next to them.

??? note "Gaussian splat setup"
    Gaussian splatting requires crazyflow's `splats` extra.
    { #gaussian-splat-setup }

    ```bash
    pip install "crazyflow[splats]"
    ```

    You can find demo splats of a flight hall and a Crazyflie drone in the [amacati/splats](https://huggingface.co/datasets/amacati/splats) dataset, and can download them via splax's `fetch`.

    [`attach_splats`][crazyflow.sim.splat.attach_splats] loads 3D gaussian splatting `.ply` files into a [`SplatData`][crazyflow.sim.splat.SplatData] struct stored in the simulation's plugin data under `sim.data.plugins["splats"]`. They persist across resets and are compatible with `jax.jit`. Note that scene splats must be aligned with the simulation frame (metric scales and +z up), and drone splats should be aligned with the drone body frame.

<details class="example" markdown>
<summary>Example</summary>

```python
import jax
from splax.io import fetch

from crazyflow.sim import Sim
from crazyflow.sim.splat import attach_splats

assets = "https://huggingface.co/datasets/amacati/splats/resolve/main"
scene = fetch(f"{assets}/robot_hall.ply")
drone = fetch(f"{assets}/cf21B_500.ply")
sim = Sim(n_worlds=4, n_drones=2, device=jax.default_backend())
attach_splats(sim, scene=scene, drone=drone)
```

</details>

## RGB images

### RGB-D camera
[`build_render_rgbd_fn`][crazyflow.sim.sensors.rgbd.build_render_rgbd_fn] builds a compiled renderer of the MJX model. It renders RGB-D images from any model camera, batched over all worlds and drones, and requires crazyflow's `rgbd` extra (`pip install "crazyflow[rgbd]"`).

<figure class="example-media">
  <img src="../../img/examples/generated/rendering/rgbd_camera/still-plot-01.png#only-light" alt="RGB-D images from the FPV cameras of two drones">
  <img src="../../img/examples/generated/rendering/rgbd_camera/still-plot-01-dark.png#only-dark" alt="RGB-D images from the FPV cameras of two drones">
</figure>

<details class="example" markdown>
<summary>Example</summary>

```{ .python continuation }
from crazyflow.sim.sensors.rgbd import build_render_rgbd_fn

render_fn = build_render_rgbd_fn(sim, resolution=(320, 240))
rgbd = render_fn(sim.data)  # (4, 2, 240, 320, 4)
rgb, depth = rgbd[..., :3], rgbd[..., 3]
```

</details>

### Gaussian splats
[`render_splat_rgb`][crazyflow.sim.sensors.splat.render_splat_rgb] renders RGB images from any model camera, batched over all worlds and drones.

<figure class="example-media">
  <img src="../../img/examples/generated/rendering/splat_camera/still-plot-01.png#only-light" alt="Gaussian splat FPV images across worlds and drones">
  <img src="../../img/examples/generated/rendering/splat_camera/still-plot-01-dark.png#only-dark" alt="Gaussian splat FPV images across worlds and drones">
</figure>

<details class="example" markdown>
<summary>Example</summary>

```{ .python continuation }
from crazyflow.sim.sensors.splat import build_render_splat_fn, render_splat_rgb

imgs = render_splat_rgb(sim, resolution=(320, 240))  # (4, 2, 240, 320, 3) in [0, 1]
imgs = render_splat_rgb(sim, drones=0, resolution=(320, 240))  # single drone: (4, 1, 240, 320, 3)

# Bake input args into a compiled function. The result is a pure function
# of sim.data, so it can be traced, jitted, and differentiated.
render_fn = build_render_splat_fn(sim, resolution=(320, 240))
imgs = render_fn(sim.data)
```

</details>

### Viewer captures
While slow, it is also possible to get the image of the interactive viewer. `sim.render(capture=True)` returns the image the browser shows as a `(height, width, 3)` array.

<figure class="example-media">
  <img src="../../img/examples/generated/rendering/cameras/still-plot-01.png#only-light" alt="Viewer capture from a drone camera">
  <img src="../../img/examples/generated/rendering/cameras/still-plot-01-dark.png#only-dark" alt="Viewer capture from a drone camera">
</figure>



## Depth images
The RGB-D camera and splats return depth along the camera's optical axis in meters, while raycasting returns distances along each pixel's ray.

### RGB-D camera
The [RGB-D camera](#rgb-d-camera) also measures depth in meters in its fourth channel. Missing hits are set to zero.

### Raycasting in MJX
[`render_depth`][crazyflow.sim.sensors.depth.render_depth] casts one ray per pixel through the MJX geometry and returns the ray distances in meters for all worlds at once. [`build_render_depth_fn`][crazyflow.sim.sensors.depth.build_render_depth_fn] builds a fully compiled depth render function for repeated calls.

<figure class="example-media">
  <img src="../../img/examples/generated/rendering/raycasting/plot-02.svg#only-light" alt="Raycast depth image of a sphere above the floor">
  <img src="../../img/examples/generated/rendering/raycasting/plot-02-dark.svg#only-dark" alt="Raycast depth image of a sphere above the floor">
</figure>

<details class="example" markdown>
<summary>Example</summary>

<!-- notest: requires rendering -->
```{ .python notest }
import jax.numpy as jnp
from crazyflow.sim.sensors.depth import build_render_depth_fn, render_depth

# One-shot depth render
dist = render_depth(sim, camera=0, resolution=(100, 100), include_drone=False)
dist = dist.at[dist > 1.5].set(jnp.nan)

# Compiled variant for repeated calls
render_fn = build_render_depth_fn(
    sim.mjx_model,
    camera=0,
    resolution=(200, 200),
    geomgroup=(1, 1, 0, 1, 1, 1, 1, 1),  # exclude drone visual mesh
)
dist = render_fn(sim)
```

</details>

### Gaussian splats
[`render_splat_rgbd`][crazyflow.sim.sensors.splat.render_splat_rgbd] adds depth in meters as a fourth channel.

<figure class="example-media">
  <img src="../../img/examples/generated/rendering/splat_depth/still-plot-01.png#only-light" alt="Gaussian splat depth image">
  <img src="../../img/examples/generated/rendering/splat_depth/still-plot-01-dark.png#only-dark" alt="Gaussian splat depth image">
</figure>

<details class="example" markdown>
<summary>Example</summary>

```{ .python continuation }
from crazyflow.sim.sensors.splat import build_render_splat_rgbd_fn, render_splat_rgbd

rgbd = render_splat_rgbd(sim, resolution=(320, 240), max_range=8.0)  # (4, 2, 240, 320, 4)
rgb, depth = rgbd[..., :3], rgbd[..., 3]

# Bake camera intrinsics, sensor range, and splat metadata into a compiled function
render_fn = build_render_splat_rgbd_fn(sim, resolution=(320, 240), max_range=8.0)
rgbd = render_fn(sim.data)
rgbd.block_until_ready()  # Wait for the asynchronous render to finish
```

</details>

Since Gaussians are partially transparent, there is no clear surface to measure depth from. Instead, we count a pixel as "solid" once enough alpha accumulates. You can control this with `alpha_threshold`.

## Cameras
The default world contains a free camera and a camera per drone. To select which one you want to render from, pass a camera name or integer ID. The default (`camera=-1`) uses the free camera. The first-person view camera is named `<drone_index>:fpv_cam`:

<details class="example" markdown>
<summary>Example</summary>

<!-- notest: requires rendering -->
```{ .python notest }
sim.render(camera="0:fpv_cam")   # first-person view from drone 0
sim.render(camera=0)             # camera by integer ID
```

</details>

## Sync and performance
Render calls that use MuJoCo synchronize the JAX state in `sim.data` with the MuJoCo buffers in `sim.mjx_data`, which can be expensive. See [MuJoCo Integration](mujoco.md#synchronization) to avoid redundant syncs.

## Next steps

- [MuJoCo Integration](mujoco.md) — how the scene is built, how to add objects, and sync internals
- [Rendering examples](../examples/rendering.md) — cameras, depth sensing, and splats
