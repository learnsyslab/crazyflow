import mujoco
import numpy as np
import pytest
from conftest import skip_if_headless
from scipy.spatial.transform import Rotation as R

from crazyflow.sim import Dynamics, Sim


@pytest.mark.unit
@pytest.mark.parametrize("cam_name", ["0:fpv_cam", "0:track_cam", "1:fpv_cam", "1:track_cam"])
@pytest.mark.render
@skip_if_headless
def test_render_camera_selection_from_name(cam_name: str):
    sim = Sim(drone="cf21B_500", n_drones=2)
    cam_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    sim.render(mode="human", camera=cam_name)
    viewer_cam = sim.viewer.viewer.cam
    assert viewer_cam.type == mujoco.mjtCamera.mjCAMERA_FIXED, "Camera type was not set to FIXED"
    assert viewer_cam.fixedcamid == cam_id, f"Expected cam ID {cam_id}, got {viewer_cam.fixedcamid}"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("cam_id", [0, 1, 2, 3])
@pytest.mark.render
@skip_if_headless
def test_render_camera_selection_from_id(cam_id: int):
    sim = Sim(drone="cf21B_500", n_drones=2)
    sim.render(mode="human", camera=cam_id)
    viewer_cam = sim.viewer.viewer.cam
    assert viewer_cam.type == mujoco.mjtCamera.mjCAMERA_FIXED, "Camera type was not set to FIXED"
    assert viewer_cam.fixedcamid == cam_id, f"Expected cam ID {cam_id}, got {viewer_cam.fixedcamid}"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("cam_name", ["0:fpv_cam", "0:track_cam"])
@pytest.mark.render
@skip_if_headless
def test_drone_camera_follows_drone(cam_name: str):
    sim = Sim(drone="cf21B_500", n_worlds=1, n_drones=1)
    cam_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    sim.render(mode="rgb_array", camera=cam_name)
    cam_pos_before = sim.mj_data.cam_xpos[cam_id].copy()
    # Teleport the drone and force a re-sync of the mjx data on the next render
    offset = np.array([1.0, 2.0, 3.0])
    states = sim.data.states.replace(pos=sim.data.states.pos + offset)
    sim.data = sim.data.replace(states=states, core=sim.data.core.replace(mjx_synced=False))
    sim.render(mode="rgb_array", camera=cam_name)
    cam_pos_after = sim.mj_data.cam_xpos[cam_id].copy()
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
    sim.render(mode="human")
    viewer_cam = sim.viewer.viewer.cam
    assert viewer_cam.type == mujoco.mjtCamera.mjCAMERA_FREE, "Camera type was not set to FREE"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("dynamics", Dynamics)
@pytest.mark.render
@skip_if_headless
def test_render_spins_props(dynamics: Dynamics):
    sim = Sim(dynamics=dynamics)
    ids = [sim.mj_model.body(f"0:prop{i}").id for i in range(4)]
    sim.render(mode="rgb_array")
    quat_before = sim.mj_data.xquat[ids].copy()
    # 7.5 RPM for one second turns each prop by 45 degrees
    states = sim.data.states.replace(rotor_vel=sim.data.states.rotor_vel + 7.5)
    core = sim.data.core.replace(steps=sim.data.core.steps + sim.freq)
    sim.data = sim.data.replace(states=states, core=core)
    sim.render(mode="rgb_array")
    quat_after = sim.mj_data.xquat[ids].copy()
    sim.close()
    rot_before = R.from_quat(quat_before, scalar_first=True)
    rot = rot_before.inv() * R.from_quat(quat_after, scalar_first=True)
    expected = np.zeros((4, 3))
    if dynamics == Dynamics.first_principles:  # Only first principles dynamics track rotor RPM
        # Rotors spin opposite to the yaw torque they exert on the body
        expected[:, 2] = -np.asarray(sim.data.params.mixing_matrix[2]) * np.pi / 4
    assert np.allclose(rot.as_rotvec(), expected, atol=1e-6), f"Prop rotations {rot.as_rotvec()}"
