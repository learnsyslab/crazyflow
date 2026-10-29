"""Run an example headless like ``python <example> [--plot]`` and save the media it shows.

Usage: ``python docs/capture_example.py examples/<path>.py <kinds> <out_dir> [--camera <name>]
[--cam_config <config>] [--geomgroup <groups>]``, see ``hooks.py`` for the arguments.
"""

import argparse
import contextlib
import importlib.util
import io
import itertools
import logging
import sys
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import imageio_ffmpeg
import jax
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib import animation

from crazyflow.sim import Sim

WIDTH, HEIGHT = 960, 540
EXAMPLE_FPS, FPS = 50, 25
CRF = "28"
# The initial free camera of the interactive viewer
CAM_CONFIG = {"lookat": [0.0, 0.0, 0.0], "distance": 2.0, "azimuth": 90.0, "elevation": -45.0}
MAX_SVG_BYTES = 500_000


class Video:
    """H.264 video of every second frame of an example, which renders at 50 fps."""

    def __init__(self, path: Path):
        """Create the video, the file is written on the first frame."""
        self.path = path
        self.writer = None
        self.n_frames = 0
        self.t = None

    def add(self, t: float | None, render: Callable[[], np.ndarray]) -> bool:
        """Render and write a frame if the video needs it, checking the fps at sim time ``t``."""
        if t is not None:
            dt = t - self.t if self.t is not None else 0.0
            assert dt <= 0 or np.isclose(dt, 1 / EXAMPLE_FPS, atol=1e-4), (
                f"Examples render at {EXAMPLE_FPS} fps, not {1 / dt:.1f} fps"
            )
            self.t = t
        self.n_frames += 1
        if (self.n_frames - 1) % (EXAMPLE_FPS // FPS):
            return False
        frame = np.ascontiguousarray(render())
        if self.writer is None:
            self.writer = imageio_ffmpeg.write_frames(
                str(self.path),
                (frame.shape[1], frame.shape[0]),
                fps=FPS,
                codec="libx264",
                quality=None,
                macro_block_size=2,
                output_params=["-crf", CRF, "-preset", "slow", "-movflags", "+faststart"],
            )
            self.writer.send(None)
        self.writer.send(frame)
        return True

    def close(self):
        """Finish writing the video."""
        if self.writer is not None:
            self.writer.close()


def save_figure(fig: plt.Figure, path: Path, anim: animation.FuncAnimation | None):
    """Save an animation as mp4 and a figure as svg, or as png if the svg is too large."""
    if anim is not None:
        anim.save(
            path.with_suffix(".mp4"),
            writer="ffmpeg",
            codec="h264",
            extra_args=["-crf", CRF, "-movflags", "+faststart"],
        )
        return
    svg = io.BytesIO()
    fig.savefig(svg, format="svg")
    if svg.tell() <= MAX_SVG_BYTES:
        path.with_suffix(".svg").write_bytes(svg.getvalue())
    else:
        fig.savefig(path.with_suffix(".png"), dpi=150)


def parse_cam_config(text: str) -> dict[str, float | list[float]]:
    """Parse a camera config like ``"distance=3 elevation=-20 lookat=0,0,1"``."""
    config = {}
    for item in text.split():
        key, value = item.split("=")
        values = [float(v) for v in value.split(",")]
        config[key] = values if len(values) > 1 else values[0]
    return config


def capture(
    example: Path,
    kinds: list[str],
    out: Path,
    docs_camera: str | None = None,
    docs_cam_config: dict | None = None,
    docs_geomgroup: list[int] | None = None,
):
    """Run the example and save the requested kinds of media to ``out``."""
    plt.switch_backend("Agg")
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    sim_render = Sim.render
    func_animation_init = animation.FuncAnimation.__init__
    videos: list[Video] = []
    figure_videos: dict[plt.Figure, Video] = {}
    animations = {}
    plot_ids = itertools.count(1)

    def render(
        sim: Sim,
        mode: str | None = "human",
        world: int = 0,
        camera: int | str = -1,
        cam_config: dict | None = None,
        width: int = 1920,
        height: int = 1080,
    ) -> np.ndarray | None:
        if mode != "human":
            return sim_render(sim, mode, world, camera, cam_config, width, height)
        if "video" not in kinds:
            return None
        if docs_camera is not None:
            camera = docs_camera
        elif docs_cam_config:
            camera = -1
        cam_config = CAM_CONFIG | (cam_config or {}) | (docs_cam_config or {})
        if sim.viewer is None:
            videos.append(Video(out / f"video-{len(videos) + 1:02d}.mp4"))
            if docs_geomgroup is not None:
                sim_render(sim, "rgb_array", world, camera, cam_config, WIDTH, HEIGHT)
                sim.viewer.viewer.vopt.geomgroup[:] = [
                    group in docs_geomgroup for group in range(mujoco.mjNGROUP)
                ]
        rendered = videos[-1].add(
            float(sim.time[world, 0]),
            lambda: sim_render(sim, "rgb_array", world, camera, cam_config, WIDTH, HEIGHT),
        )
        if not rendered:  # Only a render clears the markers, so drop those of skipped frames
            sim.viewer.viewer._markers.clear()

    def func_animation(self: animation.FuncAnimation, fig: plt.Figure, *args: Any, **kwargs: Any):
        func_animation_init(self, fig, *args, **kwargs)
        animations[fig] = self

    def show(*args: Any, block: bool | None = None, **kwargs: Any):
        if block is False:  # A live plot, recorded by pause
            return
        if "plot" in kinds:
            for num in plt.get_fignums():
                fig = plt.figure(num)
                save_figure(fig, out / f"plot-{next(plot_ids):02d}", animations.get(fig))
        plt.close("all")

    def pause(interval: float):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if fig not in figure_videos:
                figure_videos[fig] = Video(out / f"plot-{next(plot_ids):02d}.mp4")
            fig.canvas.draw()
            figure_videos[fig].add(None, lambda: np.asarray(fig.canvas.buffer_rgba())[..., :3])

    sys.argv = [str(example), *(["--plot"] if "plot" in kinds else [])]
    spec = importlib.util.spec_from_file_location("__main__", example)
    module = importlib.util.module_from_spec(spec)
    sys.modules["__main__"] = module

    jax.devices()  # Initialize the backends before capturing their log messages
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    for name in ("jax", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        with (
            contextlib.redirect_stdout(output),
            patch.object(Sim, "render", render),
            patch.object(animation.FuncAnimation, "__init__", func_animation),
            patch.object(plt, "show", show),
            patch.object(plt, "pause", pause),
        ):
            spec.loader.exec_module(module)
    finally:
        for video in [*videos, *figure_videos.values()]:
            video.close()
    if "output" in kinds and output.getvalue().strip():
        (out / "output.txt").write_text(output.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("example", type=Path)
    parser.add_argument("kinds", type=lambda kinds: kinds.split(","))
    parser.add_argument("out", type=Path)
    parser.add_argument("--camera")
    parser.add_argument("--cam_config", type=parse_cam_config)
    parser.add_argument("--geomgroup", type=lambda groups: [int(g) for g in groups.split(",")])
    args = parser.parse_args()
    capture(args.example, args.kinds, args.out, args.camera, args.cam_config, args.geomgroup)
