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
import webbrowser
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import imageio_ffmpeg
import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.text import Text
from playwright.sync_api import sync_playwright

from crazyflow.sim import Sim

WIDTH, HEIGHT = 960, 540
EXAMPLE_FPS, FPS = 50, 25
CRF = "28"
# The initial free camera of the interactive viewer
CAM_CONFIG = {"lookat": [0.0, 0.0, 0.0], "distance": 2.0, "azimuth": 90.0, "elevation": -45.0}
MAX_SVG_BYTES = 500_000
# Hardware WebGL where available, software WebGL on runners without a GPU
BROWSER_ARGS = [
    "--use-angle=vulkan",
    "--enable-features=Vulkan",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-swiftshader",
]
# Colors of the dark docs theme. Videos cannot be transparent, so they get the page background
DARK_BACKGROUND, DARK_FOREGROUND = "#1e2129", "#d5d7e2"


class Video:
    """H.264 video of every second frame of an example.

    Examples are expected to run at 50 fps. Finishing the video also saves its middle frame as
    ``still-<name>.png`` for the user guide.
    """

    def __init__(self, path: Path):
        """Create the video, the file is written on the first frame."""
        self.path = path
        self.writer = None
        self.n_frames = 0
        self.t = None

    def due(self, t: float | None = None) -> bool:
        """Count a frame of the example and return whether the video records it."""
        if t is not None:
            dt = t - self.t if self.t is not None else 0.0
            assert dt <= 0 or np.isclose(dt, 1 / EXAMPLE_FPS, atol=1e-4), (
                f"Examples render at {EXAMPLE_FPS} fps, not {1 / dt:.1f} fps"
            )
            self.t = t
        self.n_frames += 1
        return (self.n_frames - 1) % (EXAMPLE_FPS // FPS) == 0

    def write(self, frame: np.ndarray):
        """Write a frame, opening the file on the first one."""
        frame = np.ascontiguousarray(frame)
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

    def finish(self):
        """Finish writing the video and save its still."""
        if self.writer is None:
            return
        self.writer.close()
        n_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(self.path))
        reader = imageio_ffmpeg.read_frames(str(self.path))
        width, height = next(reader)["size"]
        frame = next(itertools.islice(reader, n_frames // 2, None))
        reader.close()
        still = np.frombuffer(frame, np.uint8).reshape(height, width, 3)
        plt.imsave(self.path.with_name(f"still-{self.path.stem}.png"), still)


@contextlib.contextmanager
def dark_theme(fig: plt.Figure) -> Generator[None]:
    """Restyle a figure for the dark docs theme while the context is active."""
    texts = {text: text.get_color() for text in fig.findobj(Text)}
    spines = {spine: spine.get_edgecolor() for ax in fig.axes for spine in ax.spines.values()}
    faces = {artist: artist.get_facecolor() for artist in (fig, *fig.axes)}
    for text in texts:
        text.set_color(DARK_FOREGROUND)
    for spine in spines:
        spine.set_edgecolor(DARK_FOREGROUND)
    for artist in faces:
        artist.set_facecolor(DARK_BACKGROUND)
    for ax in fig.axes:
        ax.tick_params(colors=DARK_FOREGROUND)
    yield
    for text, color in texts.items():
        text.set_color(color)
    for spine, color in spines.items():
        spine.set_edgecolor(color)
    for artist, color in faces.items():
        artist.set_facecolor(color)
    for ax in fig.axes:
        ax.tick_params(axis="x", colors=plt.rcParams["xtick.color"])
        ax.tick_params(axis="y", colors=plt.rcParams["ytick.color"])


class FigureVideo(animation.AbstractMovieWriter):
    """A movie writer that saves a figure as both light and dark videos."""

    def __init__(self, fig: plt.Figure, path: Path):
        """Initialize the video writers."""
        super().__init__()
        self.fig = fig
        self.light = Video(path.with_suffix(".mp4"))
        self.dark = Video(path.with_name(f"{path.name}-dark.mp4"))

    def setup(self, fig: plt.Figure, outfile: str, dpi: float | None = None):
        """Do nothing, the videos open on their first frame."""

    def grab_frame(self, **savefig_kwargs: Any):
        """Record the current state of the figure in both themes."""
        if self.light.due():
            self.light.write(self._draw())
        if self.dark.due():
            self.dark.write(self._draw_dark())

    def finish(self):
        """Finish writing both videos."""
        self.light.finish()
        self.dark.finish()

    def _draw(self) -> np.ndarray:
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3]

    def _draw_dark(self) -> np.ndarray:
        with dark_theme(self.fig):
            return self._draw()


def save_figure(fig: plt.Figure, path: Path, anim: animation.FuncAnimation | None):
    """Save an animation as mp4 and a figure as svg, or png if the svg is too large.

    Also saves a copy with a ``-dark`` suffix for the dark docs theme.
    """
    if anim is not None:
        anim.save(path.with_suffix(".mp4"), writer=FigureVideo(fig, path))
        return
    themes = {path.name: contextlib.nullcontext(), f"{path.name}-dark": dark_theme(fig)}
    for name, theme in themes.items():
        with theme:
            svg = io.BytesIO()
            fig.savefig(svg, format="svg", transparent=True)
            if svg.tell() <= MAX_SVG_BYTES:
                path.with_name(f"{name}.svg").write_bytes(svg.getvalue())
            else:
                fig.savefig(path.with_name(f"{name}.png"), dpi=150, transparent=True)


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
    sim_render = Sim.render
    func_animation_init = animation.FuncAnimation.__init__
    videos: list[Video] = []
    figure_videos: dict[plt.Figure, FigureVideo] = {}
    animations = {}
    plot_ids = itertools.count(1)

    def render(
        sim: Sim,
        world: int | None = None,
        camera: int | str | None = None,
        cam_config: dict | None = None,
        capture: bool = False,
        width: int = 1920,
        height: int = 1080,
    ) -> np.ndarray | None:
        if capture:  # The example uses the images itself
            return sim_render(sim, world, camera, cam_config, capture, width, height)
        if "video" not in kinds:
            return None
        if docs_camera is not None:
            camera = docs_camera
        elif docs_cam_config:
            camera = -1
        if camera in (None, -1):
            cam_config = CAM_CONFIG | (cam_config or {}) | (docs_cam_config or {})
        if sim.viewer is None:
            videos.append(Video(out / f"video-{len(videos) + 1:02d}.mp4"))
            sim_render(sim, world, camera, cam_config)
            sim.viewer._speed.value = 1e9  # Record as fast as possible instead of in real time
            if docs_geomgroup is not None:
                for group, checkbox in sim.viewer.scene._geom_group_checkboxes.items():
                    checkbox.value = group in docs_geomgroup
        record = videos[-1].due(float(sim.time[0, 0]))
        frame = sim_render(sim, world, camera, cam_config, record, WIDTH, HEIGHT)
        if record:
            videos[-1].write(frame)

    def func_animation(self: animation.FuncAnimation, fig: plt.Figure, *args: Any, **kwargs: Any):
        func_animation_init(self, fig, *args, **kwargs)
        animations[fig] = self

    def show(*args: Any, block: bool = True, **kwargs: Any):
        if not block:  # A live plot, recorded by pause
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
                figure_videos[fig] = FigureVideo(fig, out / f"plot-{next(plot_ids):02d}")
            figure_videos[fig].grab_frame()

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
    with sync_playwright() as playwright:
        page = playwright.chromium.launch(args=BROWSER_ARGS).new_page(
            viewport={"width": WIDTH, "height": HEIGHT}
        )
        with (
            contextlib.redirect_stdout(output),
            patch.object(webbrowser, "open", page.goto),  # Open the viewer in the headless page
            patch.object(Sim, "render", render),
            patch.object(animation.FuncAnimation, "__init__", func_animation),
            patch.object(plt, "show", show),
            patch.object(plt, "pause", pause),
        ):
            spec.loader.exec_module(module)
    for video in [*videos, *figure_videos.values()]:
        video.finish()
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
