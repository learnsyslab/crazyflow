# Rendering examples

Crazyflow has two renderers. MuJoCo renders the simulated geometry, and gaussian splats render photorealistic images of a captured scene.

---

## MuJoCo rendering

Rendering RGB-D images, changing materials at runtime, raycasting depth, drawing trajectories, and configuring the viewer camera. See [MuJoCo Rendering](../user-guide/mujoco-rendering.md) for the rendering modes and cameras.

### Cameras and RGBD

--8<-- example "examples/rendering/cameras.py" media="plot,output"

---

### LED deck and materials

--8<-- example "examples/rendering/led_deck.py" media="video" cam_config="distance=1.6 elevation=-45 lookat=0.5,0.5,1.4"

---

### Raycasting and depth sensing

--8<-- example "examples/rendering/raycasting.py" media="plot"

---

### Drawing trajectories

--8<-- example "examples/rendering/draw_line.py"

---

### Camera configuration

--8<-- example "examples/rendering/cam_config.py" media="video"

---

## Gaussian splat rendering

Viewing the simulation as gaussian splats, rendering RGB-D images from the drone cameras, and differentiating through the renderer. The examples download the demo splats from the [amacati/splats](https://huggingface.co/datasets/amacati/splats) dataset on first run. See [Gaussian Splat Rendering](../user-guide/splat-rendering.md) for the setup.

### Splat viewer

--8<-- example "examples/rendering/splat_viewer.py"

---

### Splat camera

--8<-- example "examples/rendering/splat_camera.py" media="plot"

---

### Splat depth camera

--8<-- example "examples/rendering/splat_depth.py" media="plot"

---

### Differentiable splat rendering

--8<-- example "examples/rendering/splat_gradients.py" media="plot"
