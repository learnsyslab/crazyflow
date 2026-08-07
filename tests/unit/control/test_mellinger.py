from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from crazyflow.control import load_params, parametrize
from crazyflow.control.mellinger import (
    attitude2force_torque,
    force_torque2rotor_vel,
    state2attitude,
)
from crazyflow.drones import available_drones

if TYPE_CHECKING:
    from crazyflow._typing import Array  # To be changed to array_api_typing later


def create_rnd_states(shape: tuple[int, ...] = ()) -> tuple[Array, Array, Array, Array]:
    x = np.random.randn(*shape, 3 + 4 + 3 + 3)
    return x[..., :3], x[..., 3:7], x[..., 7:10], x[..., 10:13]


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_state2attitude(drone: str) -> None:
    controller = parametrize(state2attitude, drone)
    # Single input
    pos, quat, vel, ang_vel = create_rnd_states()
    rpyt, pos_err_i = controller(pos, quat, vel, np.ones(13), ctrl_freq=100)
    assert rpyt.shape == (4,)
    assert pos_err_i.shape == (3,)
    # Batch input
    pos, quat, vel, ang_vel = create_rnd_states((5, 4))
    rpyt, pos_err_i = controller(pos, quat, vel, np.ones((5, 4, 13)), ctrl_freq=100)
    assert rpyt.shape == (5, 4, 4)
    assert pos_err_i.shape == (5, 4, 3)


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_attitude2force_torque(drone: str) -> None:
    controller = parametrize(attitude2force_torque, drone)
    # Single input
    pos, quat, vel, ang_vel = create_rnd_states()
    rpyt_cmd = np.array([0.1, 0.1, 0.1, 1.0])  # roll, pitch, yaw, thrust command
    force_des, torque_des, r_int_error = controller(quat, ang_vel, rpyt_cmd)
    assert force_des.shape == (1,)
    assert torque_des.shape == (3,)
    assert r_int_error.shape == (3,)
    # Batch input
    pos, quat, vel, ang_vel = create_rnd_states((5, 4))
    rpyt_cmd = np.random.randn(5, 4, 4)
    rpyt_cmd[..., 3] = np.abs(rpyt_cmd[..., 3])  # Ensure positive thrust
    force_des, torque_des, r_int_error = controller(quat, ang_vel, rpyt_cmd)
    assert force_des.shape == (5, 4, 1)
    assert torque_des.shape == (5, 4, 3)
    assert r_int_error.shape == (5, 4, 3)


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_force_torque2rotor_vel(drone: str) -> None:
    controller = parametrize(force_torque2rotor_vel, drone)
    # Single input
    force = np.array([1.0])
    torque = np.array([0.1, 0.1, 0.1])
    rotor_vel = controller(force, torque)
    assert rotor_vel.shape == (4,)
    # Batch input
    force = np.ones((5, 4, 1))
    torque = np.random.randn(5, 4, 3) * 0.1
    rotor_vel = controller(force, torque)
    assert rotor_vel.shape == (5, 4, 4)


# Correctness / physics


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_state2attitude_at_setpoint(drone: str) -> None:
    # At setpoint with identity orientation and zero acc, RPY command should be
    # [0, 0, 0] and thrust must be positive (hovering against gravity).
    controller = parametrize(state2attitude, drone)
    pos = np.zeros(3)
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    vel = np.zeros(3)
    cmd = np.zeros(13)  # setpoint at origin, zero vel/acc, yaw=0
    rpyt, _ = controller(pos, quat, vel, cmd)
    assert np.allclose(rpyt[:3], 0.0, atol=1e-6), f"RPY at setpoint should be ~0, got {rpyt[:3]}"
    assert rpyt[3] > 0.0, "Hovering thrust must be positive"


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_state2attitude_integral_error_accumulation(drone: str) -> None:
    # A constant position error must cause the integral error to accumulate
    # linearly until it would exceed int_err_max (clipped by the controller).
    controller = parametrize(state2attitude, drone)
    params = load_params(state2attitude, drone)
    pos = np.zeros(3)
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    vel = np.zeros(3)
    cmd = np.zeros(13)
    cmd[0] = 1.0  # 1 m setpoint error in x
    ctrl_freq = 100.0
    dt = 1.0 / ctrl_freq
    steps = 5

    pos_err_i = None
    for _ in range(steps):
        _, pos_err_i = controller(pos, quat, vel, cmd, pos_err_i=pos_err_i, ctrl_freq=ctrl_freq)

    expected = np.clip(
        np.array([steps * dt, 0.0, 0.0]), -params["int_err_max"], params["int_err_max"]
    )
    assert np.allclose(pos_err_i, expected, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_attitude2force_torque_at_setpoint(drone: str) -> None:
    # Identity orientation commanded → zero attitude error → zero corrective torque.
    controller = parametrize(attitude2force_torque, drone)
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    ang_vel = np.zeros(3)
    cmd = np.array([0.0, 0.0, 0.0, 0.5])  # RPY=0, positive thrust
    force_des, torque_des, _ = controller(quat, ang_vel, cmd)
    assert np.allclose(torque_des, 0.0, atol=1e-6), (
        f"Torque at setpoint should be ~0, got {torque_des}"
    )
    assert force_des[0] > 0.0, "Force must be positive for positive thrust command"


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_attitude2force_torque_zero_thrust(drone: str):
    # Zero thrust command → firmware zeros torque; outputs are all zero.
    controller = parametrize(attitude2force_torque, drone)
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    ang_vel = np.zeros(3)
    cmd = np.array([0.1, 0.1, 0.1, 0.0])  # non-zero RPY but zero thrust
    force_des, torque_des, _ = controller(quat, ang_vel, cmd)
    assert np.allclose(force_des, 0.0, atol=1e-6)
    assert np.allclose(torque_des, 0.0, atol=1e-6)


# Batch consistency (batch result == sequential result)


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_state2attitude_batch_consistency(drone: str):
    controller = parametrize(state2attitude, drone)
    batch = (3, 2)
    pos, quat, vel, _ = create_rnd_states(batch)
    cmd = np.random.randn(*batch, 13)
    rpyt_batch, err_batch = controller(pos, quat, vel, cmd)
    for i in range(batch[0]):
        for j in range(batch[1]):
            rpyt_s, err_s = controller(pos[i, j], quat[i, j], vel[i, j], cmd[i, j])
            assert np.allclose(rpyt_batch[i, j], rpyt_s, atol=1e-5)
            assert np.allclose(err_batch[i, j], err_s, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_attitude2force_torque_batch_consistency(drone: str):
    controller = parametrize(attitude2force_torque, drone)
    batch = (3, 2)
    _, quat, _, ang_vel = create_rnd_states(batch)
    cmd = np.random.randn(*batch, 4)
    cmd[..., 3] = np.abs(cmd[..., 3])
    force_batch, torque_batch, err_batch = controller(quat, ang_vel, cmd)
    for i in range(batch[0]):
        for j in range(batch[1]):
            force_s, torque_s, err_s = controller(quat[i, j], ang_vel[i, j], cmd[i, j])
            assert np.allclose(force_batch[i, j], force_s, atol=1e-5)
            assert np.allclose(torque_batch[i, j], torque_s, atol=1e-5)
            assert np.allclose(err_batch[i, j], err_s, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_force_torque2rotor_vel_batch_consistency(drone: str):
    controller = parametrize(force_torque2rotor_vel, drone)
    batch = (3, 2)
    force = np.abs(np.random.randn(*batch, 1)) * 0.05 + 0.05
    torque = np.random.randn(*batch, 3) * 0.001
    rpm_batch = controller(force, torque)
    for i in range(batch[0]):
        for j in range(batch[1]):
            rpm_s = controller(force[i, j], torque[i, j])
            assert np.allclose(rpm_batch[i, j], rpm_s, atol=1e-5)


# Symmetric force check


@pytest.mark.unit
@pytest.mark.parametrize("drone", available_drones)
def test_force_torque2rotor_vel_symmetric(drone: str):
    # Pure vertical force with zero torque → X-frame symmetry → all 4 RPMs equal.
    controller = parametrize(force_torque2rotor_vel, drone)
    force = np.array([0.2])  # total thrust, split equally across 4 motors
    torque = np.zeros(3)
    rotor_vel = controller(force, torque)
    assert np.allclose(rotor_vel, rotor_vel[0], rtol=1e-5), f"RPMs not equal: {rotor_vel}"
