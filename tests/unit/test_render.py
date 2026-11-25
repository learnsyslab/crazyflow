import numpy as np
import pytest
from conftest import skip_if_headless

from crazyflow import Sim


@pytest.mark.unit
@pytest.mark.parametrize("cam_name", ["0:fpv_cam", "0:track_cam", "1:fpv_cam", "1:track_cam"])
@pytest.mark.render
@skip_if_headless
def test_render_camera_selection_from_name(cam_name: str):
    sim = Sim(drone="cf21B_500", n_drones=2)
    sim.render(camera=cam_name)
    camera = sim.viewer._cameras.value
    assert camera == cam_name, f"Expected {cam_name}, got {camera}"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("cam_id", [0, 1, 2, 3])
@pytest.mark.render
@skip_if_headless
def test_render_camera_selection_from_id(cam_id: int):
    sim = Sim(drone="cf21B_500", n_drones=2)
    sim.render(camera=cam_id)
    cam_name = sim.mj_model.cam(cam_id).name
    camera = sim.viewer._cameras.value
    assert camera == cam_name, f"Expected {cam_name}, got {camera}"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("cam_name", ["0:fpv_cam", "0:track_cam"])
@pytest.mark.render
@skip_if_headless
def test_drone_camera_follows_drone(cam_name: str):
    sim = Sim(drone="cf21B_500", n_worlds=1, n_drones=1)
    sim.render(camera=cam_name)
    camera = next(iter(sim.viewer.server.get_clients().values())).camera
    cam_pos_before = camera.position.copy()
    # Teleport the drone and force a re-sync of the mjx data on the next render
    offset = np.array([1.0, 2.0, 3.0])
    states = sim.data.states.replace(pos=sim.data.states.pos + offset)
    sim.data = sim.data.replace(states=states, core=sim.data.core.replace(mjx_synced=False))
    sim.render()
    cam_pos_after = camera.position.copy()
    sim.close()
    assert np.allclose(cam_pos_after - cam_pos_before, offset, atol=1e-6), (
        f"Camera {cam_name} did not follow the drone: moved {cam_pos_after - cam_pos_before}, "
        f"expected {offset}"
    )


@pytest.mark.unit
@pytest.mark.render
@skip_if_headless
def test_render_free_camera():
    sim = Sim(drone="cf21B_500", n_drones=2)
    sim.render()
    assert sim.viewer._cameras.value == "Free", "Viewer did not start with the free camera"
    sim.close()
