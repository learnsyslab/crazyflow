# Rendering examples

Crazyflow has two renderers. MuJoCo renders the simulated geometry, and gaussian splats render photorealistic images of a captured scene.

---

## MuJoCo rendering

Capturing viewer images, RGB-D cameras, raycasting depth, drawing trajectories, and configuring the viewer camera. See [Rendering](../user-guide/rendering.md) for the viewer and cameras.

### Viewer captures

--8<-- example "examples/rendering/cameras.py" media="plot,output"

---

### RGB-D camera

--8<-- example "examples/rendering/rgbd_camera.py" media="plot"

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

Viewing the simulation as gaussian splats, rendering RGB-D images from the drone cameras, and differentiating through the renderer. The examples download the demo splats from the [amacati/splats](https://huggingface.co/datasets/amacati/splats) dataset on first run. See [Rendering](../user-guide/rendering.md) for the setup.

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
