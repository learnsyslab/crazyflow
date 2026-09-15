"""Regression tests for #104: motor count must be derived from ``mixing_matrix.shape[-1]``.

These tests exercise the changed pure functions directly with a synthetic 6-column mixing matrix,
without registering a full hexacopter drone (MuJoCo asset + fitted dynamics coefficients for all
four dynamics models), since none of that exists for a real non-quadcopter platform yet.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from crazyflow.control import Control, load_params
from crazyflow.control.mellinger import force_torque2rotor_vel, state2attitude
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.dynamics import Dynamics
from crazyflow.envs.drone_env import action_space
from crazyflow.sim.data import SimControls, SimCore, SimData, SimParams, SimState, SimStateDeriv
from crazyflow.sim.functional import rotor_vel_control
from crazyflow.sim.sim import rotor_vel_limits

N_MOTORS = 6

# Arbitrary, physically meaningless mixing matrix: only the motor count (last dimension) matters.
HEXA_MIXING_MATRIX = np.array(
    [
        [-1.0, -1.0, 0.0, 1.0, 1.0, 0.0],
        [-0.5, 0.5, 1.0, 0.5, -0.5, -1.0],
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    ]
)


@pytest.mark.unit
def test_sim_data_buffers_scale_with_n_motors():
    device = jax.devices("cpu")[0]
    states = SimState.create(n_worlds=2, n_drones=3, n_motors=N_MOTORS, device=device)
    assert states.rotor_vel.shape == (2, 3, N_MOTORS)
    assert states.quat.shape == (2, 3, 4)  # Unaffected: quaternion, not motor count

    states_deriv = SimStateDeriv.create(n_worlds=2, n_drones=3, n_motors=N_MOTORS, device=device)
    assert states_deriv.rotor_acc.shape == (2, 3, N_MOTORS)

    controls = SimControls.create(
        n_worlds=2,
        n_drones=3,
        n_motors=N_MOTORS,
        control=Control.rotor_vel,
        drone="cf2x_L250",
        state_freq=None,
        attitude_freq=None,
        body_rate_freq=None,
        force_torque_freq=None,
        device=device,
    )
    assert controls.rotor_vel.shape == (2, 3, N_MOTORS)


def _build_rotor_vel_sim_data(n_worlds: int, n_drones: int, n_motors: int) -> SimData:
    """Minimal SimData in Control.rotor_vel mode, with rotor buffers sized for n_motors."""
    device = jax.devices("cpu")[0]
    return SimData(
        states=SimState.create(n_worlds, n_drones, n_motors, device),
        states_deriv=SimStateDeriv.create(n_worlds, n_drones, n_motors, device),
        controls=SimControls.create(
            n_worlds,
            n_drones,
            n_motors,
            Control.rotor_vel,
            "cf2x_L250",
            None,
            None,
            None,
            None,
            device,
        ),
        params=SimParams.create(Dynamics.first_principles, "cf2x_L250", device),
        core=SimCore.create(500, n_worlds, n_drones, list(range(n_drones)), 0, device),
    )


@pytest.mark.unit
def test_rotor_vel_control_accepts_n_motors_shape():
    n_worlds, n_drones = 2, 1
    data = _build_rotor_vel_sim_data(n_worlds, n_drones, N_MOTORS)

    controls = np.full((n_worlds, n_drones, N_MOTORS), 1000.0)
    updated = rotor_vel_control(data, controls)
    assert updated.controls.rotor_vel.shape == (n_worlds, n_drones, N_MOTORS)

    # A control array shaped for the old hardcoded 4-motor assumption must be rejected.
    wrong_shape_controls = np.full((n_worlds, n_drones, 4), 1000.0)
    with pytest.raises(AssertionError):
        rotor_vel_control(data, wrong_shape_controls)


@pytest.mark.unit
def test_force_torque2rotor_vel_scales_with_mixing_matrix():
    params = load_params(force_torque2rotor_vel, "cf2x_L250")
    params["mixing_matrix"] = HEXA_MIXING_MATRIX
    # Zero torque: thrust must split evenly across all N_MOTORS motors, not divided by 4. Keep the
    # per-motor share (force / N_MOTORS) within [thrust_min, thrust_max] so it isn't clipped.
    force = np.array([0.3])
    torque = np.zeros(3)
    rotor_vel = force_torque2rotor_vel(force, torque, **params)
    assert rotor_vel.shape == (N_MOTORS,)
    expected = motor_force2rotor_vel(force / N_MOTORS, params["rpm2thrust"])
    assert rotor_vel == pytest.approx(np.full(N_MOTORS, expected.item()))


@pytest.mark.unit
def test_state2attitude_scales_with_mixing_matrix():
    params = load_params(state2attitude, "cf2x_L250")
    params["mixing_matrix"] = HEXA_MIXING_MATRIX
    pos, quat, vel = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3)
    cmd = np.zeros(13)
    rpyt, pos_err_i = state2attitude(pos, quat, vel, cmd, ctrl_freq=100, **params)
    # Collective thrust command stays 4D ([roll, pitch, yaw, thrust]) regardless of motor count.
    assert rpyt.shape == (4,)
    assert pos_err_i.shape == (3,)


@pytest.mark.unit
def test_rotor_vel_limits_scale_with_n_motors(monkeypatch: pytest.MonkeyPatch):
    import crazyflow.sim.sim as sim_module

    real_params = sim_module.load_drone_params("cf2x_L250")
    hexa_params = real_params | {"mixing_matrix": HEXA_MIXING_MATRIX.tolist()}
    monkeypatch.setattr(sim_module, "load_drone_params", lambda drone: hexa_params)

    lower, upper = rotor_vel_limits(Dynamics.so_rpy, "cf2x_L250")
    assert lower == pytest.approx(N_MOTORS * real_params["thrust_min"])
    assert upper == pytest.approx(N_MOTORS * real_params["thrust_max"])


@pytest.mark.unit
def test_action_space_thrust_bounds_scale_with_n_motors(monkeypatch: pytest.MonkeyPatch):
    import crazyflow.envs.drone_env as drone_env_module

    real_params = drone_env_module.load_params("cf2x_L250")
    hexa_params = real_params | {"mixing_matrix": HEXA_MIXING_MATRIX.tolist()}
    monkeypatch.setattr(drone_env_module, "load_params", lambda drone: hexa_params)

    space = action_space(Control.attitude, "cf2x_L250")
    assert space.low[-1] == pytest.approx(N_MOTORS * real_params["thrust_min"])
    assert space.high[-1] == pytest.approx(N_MOTORS * real_params["thrust_max"])
