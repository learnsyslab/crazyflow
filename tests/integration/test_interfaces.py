import jax
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from crazyflow.control import Control, load_params, parametrize
from crazyflow.control.mellinger import force_torque2rotor_vel, state2attitude
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim import Dynamics, Sim


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
def test_state_interface(dynamics: Dynamics):
    sim = Sim(dynamics=dynamics, control=Control.state)

    # Simple P controller for attitude to reach target height
    target_height = 0.5
    cmd = np.zeros((1, 1, 13), dtype=np.float32)
    cmd[0, 0, 2] = target_height
    steps = int(2 * sim.control_freq)  # Run simulation for 2 seconds

    for i in range(steps):  # Run simulation for 2 seconds
        cmd[..., 2] = target_height * i / steps  # Linearly interpolate target height
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)

    # Check if drone reached target position
    distance = np.linalg.norm(sim.data.states.pos[0, 0] - np.array([0.0, 0.0, target_height]))
    assert distance < 0.1, f"Failed to reach target height with {dynamics} dynamics"


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
def test_attitude_interface(dynamics: Dynamics):
    sim = Sim(dynamics=dynamics, control=Control.attitude)
    target_pos = np.array([0.0, 0.0, 1.0])
    jit_state2attitude = jax.jit(parametrize(state2attitude, drone=sim.drone))

    pos_err_i = np.zeros((1, 1, 3))
    cmd = np.zeros((1, 1, 13))
    cmd[0, 0, 2] = 1.0
    steps = int(3 * sim.control_freq)

    for i in range(steps):
        cmd[..., :3] = target_pos * i / steps  # Linearly interpolate target position
        pos, vel, quat = sim.data.states.pos, sim.data.states.vel, sim.data.states.quat
        rpyt, pos_err_i = jit_state2attitude(pos, quat, vel, cmd, pos_err_i, ctrl_freq=100)
        sim.attitude_control(rpyt)
        sim.step(sim.freq // sim.control_freq)

    # Check if drone maintained hover position
    dpos = sim.data.states.pos[0, 0] - target_pos
    distance = np.linalg.norm(dpos)
    assert distance < 0.05, f"Failed to maintain hover with {dynamics} ({dpos})"


@pytest.mark.integration
def test_rotor_vel_interface():
    sim = Sim(dynamics=Dynamics.first_principles, control=Control.rotor_vel)
    thrust_max = load_params(state2attitude, sim.drone)["thrust_max"]
    rpm2thrust = load_params(force_torque2rotor_vel, sim.drone)["rpm2thrust"]
    max_rpm = motor_force2rotor_vel(np.array([thrust_max]), rpm2thrust)[0]

    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(0.5))
    )
    cmd = np.full((1, 1, 4), max_rpm, dtype=np.float32)  # More RPMs than required for hover
    sim.rotor_vel_control(cmd)
    sim.step(sim.freq * 2)  # Run simulation for 2 seconds

    # Check if drone is not tilted
    assert R.from_quat(sim.data.states.quat[0, 0]).magnitude() < 0.1, "Drone is tilted"
    assert sim.data.states.pos[0, 0, 2] > 0.5, "Failed to accelerate with rotor velocity control"


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
def test_swarm_control(dynamics: Dynamics):
    n_worlds, n_drones = 2, 3
    sim = Sim(n_worlds=n_worlds, n_drones=n_drones, dynamics=dynamics, control=Control.state)
    start_pos = np.asarray(sim.data.states.pos)
    target_pos = sim.data.states.pos + np.array([0.3, 0.3, 0.3])
    cmd = np.zeros((n_worlds, n_drones, 13))
    steps = int(3 * sim.control_freq)

    for i in range(steps):
        alpha = i / (steps)
        cmd[..., :3] = start_pos * (1 - alpha) + target_pos * alpha
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)

    max_dist = np.max(np.linalg.norm(sim.data.states.pos - target_pos, axis=-1))
    assert max_dist < 0.08, f"Failed to reach target, max dist: {max_dist}"


@pytest.mark.integration
@pytest.mark.parametrize("dynamics", Dynamics)
def test_yaw_rotation(dynamics: Dynamics):
    if dynamics != Dynamics.first_principles:
        pytest.skip(f"Dynamics mode {dynamics} currently does not support yaw rotation")

    sim = Sim(dynamics=dynamics, control=Control.state, state_freq=100)
    sim.reset()

    cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
    cmd[..., :3] = 0.2
    cmd[..., 9] = np.pi / 2  # Test if the drone can rotate in yaw

    sim.state_control(cmd)
    sim.step(200 * sim.freq // sim.control_freq)  # Run simulation for 2 seconds
    pos = sim.data.states.pos[0, 0]
    rot = R.from_quat(sim.data.states.quat[0, 0])
    distance = np.linalg.norm(pos - np.array([0.2, 0.2, 0.2]))
    assert distance < 0.1, f"Failed to reach target, distance: {distance}"
    angle = rot.as_euler("xyz")[2]
    assert np.abs(angle - np.pi / 2) < 0.1, f"Failed to rotate in yaw, angle: {angle}"
