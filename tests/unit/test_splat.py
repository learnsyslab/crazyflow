"""Unit tests for the gaussian splat module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import available_backends
from scipy.spatial.transform import Rotation as R

splax = pytest.importorskip("splax", reason="the splat modules require the optional splats extra")

from crazyflow.sim import Sim  # noqa: E402
from crazyflow.sim.sensors.splat import (  # noqa: E402
    build_render_splat_fn,
    camera_intrinsics,
    render_splat_rgb,
    viewmats,
)
from crazyflow.sim.splat import (  # noqa: E402
    SPLAT_KEYS,
    SPLAT_SLICES_KEY,
    SplatViewer,
    attach_splats,
)

if TYPE_CHECKING:
    from pathlib import Path

requires_gpu = pytest.mark.skipif("gpu" not in available_backends(), reason="splax requires CUDA")


def _write_splat(path: Path, n: int = 64, extent: float = 0.5) -> int:
    """Write a small synthetic render-space splat to a .ply file."""
    rng = np.random.default_rng(0)
    means = rng.uniform(-extent, extent, (n, 3)).astype(np.float32)
    scales = np.full((n, 3), 0.05, np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    colors = rng.uniform(0.2, 0.8, (n, 3)).astype(np.float32)
    opacities = np.full((n, 1), 0.9, np.float32)
    splax.io.write_ply(path, means, scales, quats, colors, opacities)


@pytest.mark.unit
def test_viewmats():
    # A camera at the origin with identity cam_xmat looks along -z (MuJoCo/OpenGL convention). In
    # the OpenCV convention splax expects, that point must land at +z, and world +y (up in the GL
    # camera frame) at -y.
    vm = np.asarray(viewmats(np.zeros((1, 3)), np.eye(3)[None]))
    assert vm.shape == (1, 4, 4)
    assert np.allclose(vm[0] @ [0.0, 0.0, -2.0, 1.0], [0.0, 0.0, 2.0, 1.0], atol=1e-6)
    assert np.allclose(vm[0] @ [0.0, 1.0, -2.0, 1.0], [0.0, -1.0, 2.0, 1.0], atol=1e-6)
    # Random camera pose: rotation stays orthonormal with det +1, camera center maps to the origin
    xpos, xmat = np.random.default_rng(1).normal(size=(1, 3)), R.random().as_matrix()[None]
    vm = np.asarray(viewmats(xpos, xmat))
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
def test_attach_splats(tmp_path: Path):
    n_splats = 64
    _write_splat(tmp_path / "splat.ply", n=n_splats)
    sim = Sim(n_worlds=2, n_drones=2)
    attach_splats(sim, scene=tmp_path / "splat.ply", drone=tmp_path / "splat.ply")
    for key, dim in zip(SPLAT_KEYS, (3, 3, 4, 3, 1)):
        assert key in sim.data.plugins, f"Missing plugin key {key}"
        assert sim.data.plugins[key].shape == (3 * n_splats, dim)
        assert sim.data.plugins[key].device == sim.device
    slices = np.asarray(sim.data.plugins[SPLAT_SLICES_KEY])
    assert np.array_equal(slices, [[n_splats, 2 * n_splats], [2 * n_splats, 3 * n_splats]])
    # Splat data must survive resets
    sim.reset()
    assert all(key in sim.data.plugins for key in SPLAT_KEYS)
    for key, dim in zip(SPLAT_KEYS, (3, 3, 4, 3, 1)):
        assert sim.data.plugins[key].shape == (3 * n_splats, dim)
    assert np.asarray(sim.data.plugins[SPLAT_SLICES_KEY]).shape == (2, 2)


@pytest.mark.unit
def test_attach_splats_scene_only(tmp_path: Path):
    n_splats = 64
    _write_splat(tmp_path / "splat.ply", n=n_splats)
    sim = Sim(n_drones=2)
    attach_splats(sim, scene=tmp_path / "splat.ply")
    assert sim.data.plugins[SPLAT_KEYS[0]].shape == (n_splats, 3)
    assert np.asarray(sim.data.plugins[SPLAT_SLICES_KEY]).shape == (0, 2)


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
        SplatViewer(sim)


@pytest.mark.unit
def test_render_splat_requires_gpu(tmp_path: Path):
    _write_splat(tmp_path / "splat.ply")
    sim = Sim(device="cpu")
    attach_splats(sim, drone=tmp_path / "splat.ply")
    with pytest.raises(RuntimeError, match="GPU"):
        render_splat_rgb(sim)


@pytest.mark.unit
@requires_gpu
def test_render_splat_rgb(tmp_path: Path):
    _write_splat(tmp_path / "splat.ply", extent=1.0)
    sim = Sim(n_worlds=2, n_drones=2, device="gpu")
    attach_splats(sim, scene=tmp_path / "splat.ply", drone=tmp_path / "splat.ply")
    # Every drone's fpv camera renders into a (n_worlds, n_drones, H, W, 3) stack
    img = np.asarray(render_splat_rgb(sim, resolution=(32, 24)))
    assert img.shape == (2, 2, 24, 32, 3)
    assert np.all(np.isfinite(img))
    assert img.max() > 0.0, "Nothing is visible in the image"
    # Selecting a single drone drops the drone axis and matches that slice of the full stack
    one = np.asarray(render_splat_rgb(sim, drones=1, resolution=(32, 24)))
    assert one.shape == (2, 24, 32, 3)
    assert np.allclose(one, img[:, 1], atol=1e-5)
    # The compiled variant renders the same images
    render_fn = build_render_splat_fn(sim, resolution=(32, 24))
    assert np.allclose(img, np.asarray(render_fn(sim)), atol=1e-5)
    # Hiding each drone from its own camera changes the images
    excl = np.asarray(render_splat_rgb(sim, resolution=(32, 24), exclude_self=True))
    assert not np.allclose(img, excl)


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
