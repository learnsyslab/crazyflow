"""Unit tests for the gaussian splat module."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest
from conftest import available_backends
from scipy.spatial.transform import Rotation as R

splax = pytest.importorskip("splax", reason="the splat modules require the optional splats extra")

from crazyflow.sim import Sim  # noqa: E402
from crazyflow.sim.sensors.splat import (  # noqa: E402
    build_render_splat_fn,
    build_render_splat_rgbd_fn,
    camera_intrinsics,
    render_splat_rgb,
    render_splat_rgbd,
    viewmats,
)
from crazyflow.sim.splat import SPLATS_KEY, SplatViewer, attach_splats  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

requires_gpu = pytest.mark.skipif("gpu" not in available_backends(), reason="splax requires CUDA")


def _write_splat(path: Path, n: int = 64, extent: float = 0.5, n_coeffs: int = 1):
    """Write a small synthetic splat with ``n_coeffs`` SH coefficients per gaussian."""
    rng = np.random.default_rng(0)
    means = rng.uniform(-extent, extent, (n, 3)).astype(np.float32)
    log_scales = np.full((n, 3), np.log(0.05), np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    sh_colors = np.zeros((n, n_coeffs, 3), np.float32)
    sh_colors[:, 0] = splax.io.rgb_to_sh(rng.uniform(0.2, 0.8, (n, 3)))
    sh_colors[:, 1:] = rng.uniform(-0.5, 0.5, (n, n_coeffs - 1, 3))
    logit_opacities = np.full((n,), 2.0, np.float32)
    splax.io.write_ply(path, means, log_scales, quats, sh_colors, logit_opacities)


@pytest.mark.unit
def test_viewmats():
    # A camera at the origin with identity cam_xmat looks along -z (MuJoCo/OpenGL convention). In
    # the OpenCV convention splax expects, that point must land at +z, and world +y (up in the GL
    # camera frame) at -y.
    vm = viewmats(np.zeros((1, 3)), np.eye(3)[None])
    assert vm.shape == (1, 4, 4)
    assert np.allclose(vm[0] @ np.array([0.0, 0.0, -2.0, 1.0]), [0.0, 0.0, 2.0, 1.0], atol=1e-6)
    assert np.allclose(vm[0] @ np.array([0.0, 1.0, -2.0, 1.0]), [0.0, -1.0, 2.0, 1.0], atol=1e-6)
    # Random camera pose: rotation stays orthonormal with det +1, camera center maps to the origin
    xpos, xmat = np.random.default_rng(1).normal(size=(1, 3)), R.random().as_matrix()[None]
    vm = viewmats(xpos, xmat)
    rot = vm[0, :3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(rot), 1.0)
    assert np.allclose(vm[0] @ np.append(xpos[0], 1.0), [0.0, 0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.unit
def test_camera_intrinsics():
    sim = Sim()
    width, height = 320, 240
    f, c = camera_intrinsics(sim.mj_model, 0, (width, height))
    fov_y = np.deg2rad(sim.mj_model.cam_fovy[0])
    assert np.isclose(f[0], f[1]), "Pixels must be square"
    assert np.isclose(f[1], height / 2 / np.tan(fov_y / 2))
    assert c == (width / 2, height / 2)


@pytest.mark.unit
@pytest.mark.parametrize("n_coeffs", [1, 4, 16])
def test_attach_splats(tmp_path: Path, n_coeffs: int):
    n_splats = 64
    _write_splat(tmp_path / "splat.ply", n=n_splats, n_coeffs=n_coeffs)
    sim = Sim(n_worlds=2, n_drones=2)
    attach_splats(sim, scene=tmp_path / "splat.ply", drone=tmp_path / "splat.ply")
    # The scene and both drone copies concatenate into one buffer per parameter array
    n = 3 * n_splats
    shapes = ((n, 3), (n, 3), (n, 4), (n, n_coeffs, 3), (n,))
    splats = sim.data.plugins[SPLATS_KEY]
    assert tuple(x.shape for x in splats.params) == shapes
    assert all(x.device == sim.device for x in splats.params)
    assert splats.slices == ((n_splats, 2 * n_splats), (2 * n_splats, 3 * n_splats))
    # Splat data must survive resets
    sim.reset()
    splats = sim.data.plugins[SPLATS_KEY]
    assert tuple(x.shape for x in splats.params) == shapes
    assert splats.slices == ((n_splats, 2 * n_splats), (2 * n_splats, 3 * n_splats))


@pytest.mark.unit
def test_attach_splats_scene_only(tmp_path: Path):
    n_splats = 64
    _write_splat(tmp_path / "splat.ply", n=n_splats)
    sim = Sim(n_drones=2)
    attach_splats(sim, scene=tmp_path / "splat.ply")
    assert sim.data.plugins[SPLATS_KEY].means.shape == (n_splats, 3)
    assert sim.data.plugins[SPLATS_KEY].slices == ()


@pytest.mark.unit
def test_attach_splats_sh_mismatch(tmp_path: Path):
    _write_splat(tmp_path / "scene.ply", n_coeffs=1)
    _write_splat(tmp_path / "drone.ply", n_coeffs=4)
    sim = Sim()
    with pytest.raises(AssertionError, match="SH degree"):
        attach_splats(sim, scene=tmp_path / "scene.ply", drone=tmp_path / "drone.ply")


@pytest.mark.unit
def test_attach_splats_no_input():
    sim = Sim()
    with pytest.raises(ValueError, match="scene or drone"):
        attach_splats(sim)


@pytest.mark.unit
def test_render_splat_before_attach():
    sim = Sim()
    with pytest.raises(RuntimeError, match="attach_splats"):
        render_splat_rgb(sim)
    with pytest.raises(RuntimeError, match="attach_splats"):
        build_render_splat_fn(sim)
    with pytest.raises(RuntimeError, match="attach_splats"):
        render_splat_rgbd(sim)
    with pytest.raises(RuntimeError, match="attach_splats"):
        build_render_splat_rgbd_fn(sim)
    with pytest.raises(RuntimeError, match="attach_splats"):
        SplatViewer(sim)


@pytest.mark.unit
def test_render_splat_requires_gpu(tmp_path: Path):
    _write_splat(tmp_path / "splat.ply")
    sim = Sim(device="cpu")
    attach_splats(sim, drone=tmp_path / "splat.ply")
    with pytest.raises(RuntimeError, match="GPU"):
        render_splat_rgb(sim)
    with pytest.raises(RuntimeError, match="GPU"):
        render_splat_rgbd(sim)


@pytest.mark.unit
@requires_gpu
def test_render_splat_rgb(tmp_path: Path):
    _write_splat(tmp_path / "splat.ply", extent=1.0)
    sim = Sim(n_worlds=2, n_drones=2, device="gpu")
    attach_splats(sim, scene=tmp_path / "splat.ply", drone=tmp_path / "splat.ply")
    # Every drone's fpv camera renders into a (n_worlds, n_drones, H, W, 3) stack
    img = render_splat_rgb(sim, resolution=(32, 24))
    assert img.shape == (2, 2, 24, 32, 3)
    assert np.all(np.isfinite(img))
    assert img.max() > 0.0, "Nothing is visible in the image"
    # Selecting a single drone matches that slice of the full stack
    one = render_splat_rgb(sim, drones=1, resolution=(32, 24))
    assert one.shape == (2, 1, 24, 32, 3)
    assert np.allclose(one[:, 0], img[:, 1], atol=1e-5)
    # The compiled variant renders the same images
    render_fn = build_render_splat_fn(sim, resolution=(32, 24))
    assert np.allclose(img, render_fn(sim.data), atol=1e-5)
    # Hiding each drone from its own camera changes the images
    excl = render_splat_rgb(sim, resolution=(32, 24), exclude_self=True)
    assert not np.allclose(img, excl)


@pytest.mark.unit
@requires_gpu
def test_render_splat_clips(tmp_path: Path):
    n = 64
    rng = np.random.default_rng(0)
    means = rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)
    log_scales = np.full((n, 3), np.log(0.2), np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    sh_colors = np.asarray(splax.io.rgb_to_sh(np.full((n, 3), 3.0, np.float32)))[:, None]
    opacities = np.full((n,), 10.0, np.float32)
    splax.io.write_ply(tmp_path / "bright.ply", means, log_scales, quats, sh_colors, opacities)
    sim = Sim(device="gpu")
    attach_splats(sim, scene=tmp_path / "bright.ply")
    img = render_splat_rgb(sim, resolution=(32, 24), background=(0.0, 0.0, 0.0))
    assert img.max() == 1.0, "Colors above the sensor range must saturate, not exceed it"


@pytest.mark.unit
@requires_gpu
def test_render_splat_camera_prefix(tmp_path: Path):
    _write_splat(tmp_path / "splat.ply", extent=1.0)
    sim = Sim(n_worlds=2, n_drones=2, device="gpu")
    attach_splats(sim, scene=tmp_path / "splat.ply", drone=tmp_path / "splat.ply")
    fpv = render_splat_rgb(sim, resolution=(32, 24))
    track = render_splat_rgb(sim, resolution=(32, 24), camera_prefix="track_cam")
    assert track.shape == fpv.shape
    assert not np.allclose(fpv, track), "track_cam and fpv_cam must give different views"
    # The builder fuses the prefix in and matches the direct call
    render_fn = build_render_splat_fn(sim, resolution=(32, 24), camera_prefix="track_cam")
    assert np.allclose(track, render_fn(sim.data), atol=1e-5)
    # An unknown prefix raises
    with pytest.raises(ValueError, match="not found"):
        render_splat_rgb(sim, resolution=(32, 24), camera_prefix="does_not_exist")


@pytest.mark.unit
@requires_gpu
def test_render_splat_rgbd(tmp_path: Path):
    _write_splat(tmp_path / "splat.ply", extent=1.0)
    sim = Sim(n_worlds=2, n_drones=2, device="gpu")
    attach_splats(sim, scene=tmp_path / "splat.ply", drone=tmp_path / "splat.ply")
    # Every drone's fpv camera renders into a (n_worlds, n_drones, H, W, 4) rgbd stack
    img = render_splat_rgbd(sim, resolution=(32, 24), max_range=7.0)
    assert img.shape == (2, 2, 24, 32, 4)
    assert np.all(np.isfinite(img))
    depth = img[..., 3]
    assert depth.max() <= 7.0, "Depth reaches past the sensor range"
    assert depth.min() < 7.0, "Nothing is visible in the depth image"
    # Depth is metric, so it leaves the unit interval the color channels are confined to
    assert depth[depth < 7.0].max() > 1.0
    # The color channels match the dedicated rgb sensor approximately
    assert np.allclose(img[..., :3], render_splat_rgb(sim, resolution=(32, 24)), atol=1e-5)
    # Selecting a single drone keeps the drone axis and matches that slice of the full stack
    one = render_splat_rgbd(sim, drones=1, resolution=(32, 24), max_range=7.0)
    assert one.shape == (2, 1, 24, 32, 4)
    assert np.allclose(one[:, 0], img[:, 1], atol=1e-5)
    # The compiled variant renders the same images
    render_fn = build_render_splat_rgbd_fn(sim, resolution=(32, 24), max_range=7.0)
    assert np.allclose(img, render_fn(sim.data), atol=1e-5)
    # Hiding each drone from its own camera changes the images
    excl = render_splat_rgbd(sim, resolution=(32, 24), max_range=7.0, exclude_self=True)
    assert not np.allclose(img, excl)


@pytest.mark.unit
@requires_gpu
def test_render_splat_rgbd_range(tmp_path: Path):
    _write_splat(tmp_path / "splat.ply", extent=1.0)
    sim = Sim(device="gpu")
    attach_splats(sim, scene=tmp_path / "splat.ply", drone=tmp_path / "splat.ply")
    render = partial(render_splat_rgbd, sim, resolution=(32, 24), max_range=7.0)
    # Demanding more coverage can only reject hits, never create them, so nothing moves closer
    lenient, strict = render(alpha_threshold=0.1)[..., 3], render(alpha_threshold=0.9)[..., 3]
    assert np.all(strict >= lenient - 1e-5)
    assert np.any(strict > lenient), "The threshold never rejected a pixel"
    # No coverage can reach a threshold above one, leaving an empty image at the sensor range
    assert np.all(render(alpha_threshold=1.1)[..., 3] == 7.0)
    # Hits are clipped to the range as well, not just the pixels that miss
    near = render_splat_rgbd(sim, resolution=(32, 24), max_range=1e-3)
    assert np.all(near[..., 3] == 1e-3)
    # Only the depth channel is gated, the colors are untouched by the sensor range
    assert np.allclose(near[..., :3], render()[..., :3], atol=1e-5)


@pytest.mark.unit
@requires_gpu
def test_build_render_splat_fn(tmp_path: Path):
    _write_splat(tmp_path / "splat.ply", extent=1.0)
    sim = Sim(n_worlds=2, n_drones=2, device="gpu")
    attach_splats(sim, scene=tmp_path / "splat.ply", drone=tmp_path / "splat.ply")
    # The renderer is a pure function of the simulation data, so it traces and jits
    render_fn = build_render_splat_fn(sim, resolution=(32, 24))
    assert np.allclose(render_fn(sim.data), jax.jit(render_fn)(sim.data), atol=1e-5)
    # Rebuilding is only an optimization, the images match the one-shot renderer
    assert np.allclose(render_fn(sim.data), render_splat_rgb(sim, resolution=(32, 24)), atol=1e-5)
    # The renderer reads the poses per call instead of baking in the poses it was built with
    before = render_fn(sim.data)
    sim.step(sim.freq // 10)
    assert not np.allclose(before, render_fn(sim.data))
    # The drone selection and self-exclusion carry through the builder
    one = build_render_splat_fn(sim, resolution=(32, 24), drones=1)(sim.data)
    assert one.shape == (2, 1, 24, 32, 3)
    assert np.allclose(one, render_splat_rgb(sim, drones=1, resolution=(32, 24)), atol=1e-5)
    excl = build_render_splat_fn(sim, resolution=(32, 24), exclude_self=True)(sim.data)
    assert np.allclose(excl, render_splat_rgb(sim, resolution=(32, 24), exclude_self=True), 1e-5)
    assert not np.allclose(excl, render_fn(sim.data))


@pytest.mark.unit
@requires_gpu
def test_build_render_splat_rgbd_fn(tmp_path: Path):
    _write_splat(tmp_path / "splat.ply", extent=1.0)
    sim = Sim(n_worlds=2, n_drones=2, device="gpu")
    attach_splats(sim, scene=tmp_path / "splat.ply", drone=tmp_path / "splat.ply")
    render_fn = build_render_splat_rgbd_fn(sim, resolution=(32, 24), max_range=7.0)
    assert np.allclose(render_fn(sim.data), jax.jit(render_fn)(sim.data), atol=1e-5)
    direct = partial(render_splat_rgbd, sim, resolution=(32, 24), max_range=7.0)
    assert np.allclose(render_fn(sim.data), direct(), atol=1e-5)
    before = render_fn(sim.data)
    sim.step(sim.freq // 10)
    assert not np.allclose(before, render_fn(sim.data))
    # The sensor range and the coverage threshold carry through the builder
    render_fn = build_render_splat_rgbd_fn(
        sim, resolution=(32, 24), max_range=7.0, alpha_threshold=1.1
    )
    assert np.all(render_fn(sim.data)[..., 3] == 7.0)
    render_fn = build_render_splat_rgbd_fn(
        sim, resolution=(32, 24), max_range=7.0, drones=1, exclude_self=True
    )
    one = render_fn(sim.data)
    assert one.shape == (2, 1, 24, 32, 4)
    assert np.allclose(one, direct(drones=1, exclude_self=True), atol=1e-5)


@pytest.mark.unit
def test_splat_viewer(tmp_path: Path):
    pytest.importorskip("viser")

    _write_splat(tmp_path / "splat.ply")
    sim = Sim(n_drones=2)
    attach_splats(sim, scene=tmp_path / "splat.ply", drone=tmp_path / "splat.ply")
    viewer = SplatViewer(sim)
    sim.step()
    viewer.update(sim)
    viewer.close()
    sim.close()
