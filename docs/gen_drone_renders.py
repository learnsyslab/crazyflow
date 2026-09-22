"""Render one image of every supported drone for the documentation.

This script is executed by the mkdocs-gen-files plugin during ``mkdocs build`` or
``mkdocs serve``. The images are written to ``img/drones/<name>.png`` in the virtual docs tree
and are never committed. Rendering is headless via EGL on Linux and CGL on macOS unless
``MUJOCO_GL`` is already set.
"""

import logging
import os
import sys
from pathlib import Path

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
elif sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

try:
    import mkdocs_gen_files
except ImportError:
    pass  # not running in a docs environment — nothing to generate
else:
    import imageio.v3 as iio
    import mujoco
    import numpy as np

    from crazyflow.drones import Drone

    log = logging.getLogger("mkdocs.plugins.gen_drone_renders")

    DRONE_DIR = Path("crazyflow/drones")
    WIDTH, HEIGHT = 2000, 1500
    MARGIN = 80

    def render(name: str) -> np.ndarray:
        """Render a drone in front of a transparent background, cropped to its bounding box."""
        spec = mujoco.MjSpec.from_file(str(DRONE_DIR / f"{name}.xml"))
        for body in list(spec.worldbody.bodies):
            if body.name != "drone":
                spec.delete(body)
        for material in spec.materials:
            if 0 < material.rgba[3] < 1:
                material.rgba[3] = 1
        spec.visual.global_.offwidth = WIDTH
        spec.visual.global_.offheight = HEIGHT
        spec.visual.quality.shadowsize = 8192
        spec.visual.quality.offsamples = 16
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
        spec.worldbody.add_light(
            pos=[0.4, -0.3, 0.6], dir=[-0.55, 0.4, -0.8], diffuse=[0.6, 0.6, 0.6], castshadow=True
        )
        spec.worldbody.add_light(
            pos=[-0.4, 0.3, 0.3], dir=[0.6, -0.45, -0.45], diffuse=[0.3, 0.3, 0.3], castshadow=False
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        camera = mujoco.MjvCamera()
        camera.lookat[:] = model.body("drone").pos
        camera.distance, camera.azimuth, camera.elevation = 0.28, 135, -28
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[2] = 1

        renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        try:
            renderer.update_scene(data, camera, option)
            rgb = renderer.render()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera, option)
            mask = renderer.render()[..., 0] != -1
        finally:
            renderer.close()

        rows, cols = np.nonzero(mask)
        y0, y1 = max(rows.min() - MARGIN, 0), min(rows.max() + MARGIN, HEIGHT)
        x0, x1 = max(cols.min() - MARGIN, 0), min(cols.max() + MARGIN, WIDTH)
        alpha = mask.astype(np.uint8) * 255
        return np.dstack([rgb, alpha])[y0:y1, x0:x1]

    for name in Drone:
        try:
            image = render(name)
        except Exception as e:
            log.warning(f"Could not render drone '{name}': {e!r}")
            continue
        with mkdocs_gen_files.open(f"img/drones/{name}.png", "wb") as fd:
            fd.write(iio.imwrite("<bytes>", image, extension=".png"))
