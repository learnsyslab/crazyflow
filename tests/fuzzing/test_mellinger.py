"""Compare the Mellinger controller against the Crazyflie firmware."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from unittest.mock import patch

import cffirmware as cf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy.spatial.transform import Rotation as R

from crazyflow.control import load_fn_params, parametrize
from crazyflow.control.mellinger import control
from crazyflow.control.transform import motor_force2rotor_vel, pwm2force
from crazyflow.drones import Drone

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture(scope="module")
def ref() -> cf.controllerMellinger_t:
    """Create the firmware controller with the cf2x_L250 gains."""
    ctrl = cf.controllerMellinger_t()
    cf.controllerMellingerInit(ctrl)
    ctrl.kd_z = 0.5  # TODO: Remove once cf2x_L250 uses the firmware kd_z, see test_params
    return ctrl


def _vector(bound: float, dim: int = 3) -> st.SearchStrategy[NDArray]:
    """Draw float32 vectors of size dim within the bound."""
    return hnp.arrays(np.float32, dim, elements={"min_value": -bound, "max_value": bound})


@st.composite
def _samples(draw: st.DrawFn) -> tuple[NDArray, ...]:
    """Draw a state, setpoint and controller memory combining special and random values."""
    special = st.sampled_from([0.0, np.pi / 2, -np.pi / 2, np.pi, -np.pi])
    angle = st.one_of(special, st.floats(-0.1, 0.1), st.floats(-np.pi, np.pi))
    # float32 so both controllers get identical inputs
    quat = R.from_euler("xyz", draw(st.tuples(angle, angle, angle))).as_quat().astype(np.float32)
    pos, vel = draw(_vector(10)), draw(_vector(5))
    ang_vel = draw(_vector(35))
    prev_ang_vel = ang_vel + draw(_vector(35))
    # Beyond the integrator limits to cover clamping
    pos_err_i, r_int_error = draw(_vector(3)), draw(_vector(2000))
    pos_des, vel_des = pos + draw(_vector(4)), vel + draw(_vector(2))
    acc_des = draw(_vector(20))  # Includes free fall
    quat_des = R.from_euler("z", draw(angle)).as_quat().astype(np.float32)
    rate_des = draw(_vector(35))
    prev_rate_des = rate_des + draw(_vector(35))
    cmd = np.concatenate((pos_des, vel_des, acc_des, quat_des, rate_des))
    integrators = np.concatenate((pos_err_i, r_int_error))
    return pos, vel, quat, ang_vel, cmd, prev_ang_vel, integrators, prev_rate_des


def _reference_step(
    ctrl: cf.controllerMellinger_t,
    pos: NDArray,
    vel: NDArray,
    quat: NDArray,
    ang_vel: NDArray,
    cmd: NDArray,
    prev_ang_vel: NDArray,
    integrators: NDArray,
    prev_rate_des: NDArray,
) -> cf.control_t:
    """Run one firmware controller step from the given controller memory."""
    ctrl.i_error_x, ctrl.i_error_y, ctrl.i_error_z = integrators[:3].tolist()
    # The firmware runs in the legacy Crazyflie frame with a flipped pitch axis
    ctrl.i_error_m_x, ctrl.i_error_m_y, ctrl.i_error_m_z = (integrators[3:] * [1, -1, 1]).tolist()
    ctrl.prev_omega_roll, ctrl.prev_omega_pitch = (prev_ang_vel[:2] * [1, -1]).tolist()
    ctrl.prev_setpoint_omega_roll, ctrl.prev_setpoint_omega_pitch = prev_rate_des[:2].tolist()
    state, sensors, setpoint = cf.state_t(), cf.sensorData_t(), cf.setpoint_t()
    state.position.x, state.position.y, state.position.z = pos.tolist()
    state.velocity.x, state.velocity.y, state.velocity.z = vel.tolist()
    q = state.attitudeQuaternion
    q.x, q.y, q.z, q.w = quat.tolist()
    sensors.gyro.x, sensors.gyro.y, sensors.gyro.z = np.degrees(ang_vel).tolist()
    setpoint.position.x, setpoint.position.y, setpoint.position.z = cmd[:3].tolist()
    setpoint.velocity.x, setpoint.velocity.y, setpoint.velocity.z = cmd[3:6].tolist()
    acc = setpoint.acceleration
    acc.x, acc.y, acc.z = cmd[6:9].tolist()
    setpoint.attitude.yaw = float(np.degrees(R.from_quat(cmd[9:13]).as_euler("xyz")[2]))
    rate = setpoint.attitudeRate
    rate.roll, rate.pitch, rate.yaw = np.degrees(cmd[13:16]).tolist()
    setpoint.mode.x = setpoint.mode.y = setpoint.mode.z = setpoint.mode.yaw = cf.modeAbs
    out = cf.control_t()
    cf.controllerMellinger(ctrl, out, setpoint, sensors, state, 0)
    return out


@pytest.mark.xfail(strict=True, reason="TODO: cf2x_L250 uses kd_z = 0.5, the firmware 0.4")
def test_params():
    """Compare the gains with the firmware defaults."""
    ctrl = cf.controllerMellinger_t()
    cf.controllerMellingerInit(ctrl)
    state_params = load_fn_params(control.state2attitude, Drone.cf2x_L250, xp=jnp)
    attitude_params = load_fn_params(control.attitude2force_torque, Drone.cf2x_L250, xp=jnp)
    xy_z = {
        "kp": (ctrl.kp_xy, ctrl.kp_z),
        "kd": (ctrl.kd_xy, ctrl.kd_z),
        "ki": (ctrl.ki_xy, ctrl.ki_z),
        "kR": (ctrl.kR_xy, ctrl.kR_z),
        "kw": (ctrl.kw_xy, ctrl.kw_z),
        "ki_m": (ctrl.ki_m_xy, ctrl.ki_m_z),
        "kd_omega": (ctrl.kd_omega_rp, 0.0),
    }
    params = state_params | attitude_params
    for name, (xy, z) in xy_z.items():
        np.testing.assert_allclose(params[name], [xy, xy, z], rtol=1e-6, err_msg=name)
    i_range = [ctrl.i_range_xy, ctrl.i_range_xy, ctrl.i_range_z]
    np.testing.assert_allclose(state_params["int_err_max"], i_range, rtol=1e-6)
    i_range_m = [ctrl.i_range_m_xy, ctrl.i_range_m_xy, ctrl.i_range_m_z]
    np.testing.assert_allclose(attitude_params["int_err_max"], i_range_m, rtol=1e-6)
    np.testing.assert_allclose(state_params["mass"], ctrl.mass, rtol=1e-6)
    np.testing.assert_allclose(state_params["mass_thrust"], ctrl.massThrust, rtol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="TODO: Use the current and previous body rate setpoint and reset the integrators at "
    "zero thrust as the firmware",
)
@given(sample=_samples())
def test_controller(ref: cf.controllerMellinger_t, sample: tuple[NDArray, ...]):
    """Compare thrust, torques and integrators."""
    pos, vel, quat, ang_vel, cmd, prev_ang_vel, integrators, _ = sample
    state2attitude = parametrize(control.state2attitude, Drone.cf2x_L250)
    attitude2force_torque = parametrize(control.attitude2force_torque, Drone.cf2x_L250)
    # The firmware outputs thrust and torques in PWM, whereas crazyflow's controller returns them
    # after mixing and clipping in SI units. To get access to the intermediate PWM values in
    # crazyflow's controllers, we patch the intermediate functions that get called with the PWM
    # values and compare their arguments instead of the final output. Monkey-patching is
    # incompatible with jit, so we use numpy here
    with patch.object(control, "pwm2force", wraps=control.pwm2force) as spy:
        rpyt, pos_err_i = state2attitude(pos, quat, vel, cmd, integrators[:3], cf.ATTITUDE_RATE)
    thrust = spy.call_args.args[0]
    mixer = control.force_torque_pwms2pwms
    with patch.object(control, "force_torque_pwms2pwms", wraps=mixer) as spy:
        _, _, r_int_error = attitude2force_torque(
            quat, ang_vel, rpyt, prev_ang_vel, integrators[3:]
        )
    # The firmware runs in the legacy Crazyflie frame with flipped pitch and yaw torques
    torque = spy.call_args.args[1] * [1, -1, -1]
    out = _reference_step(ref, *sample)
    np.testing.assert_allclose(thrust, out.thrust, atol=1)
    # Zero thrust turns off torques. If the firmware and crazyflow controller disagree within
    # numerical noise, the torque contribution can still lead to significant differences. We hence
    # only compare torques if we have at least 1 PWM of thrust
    assume(abs(out.thrust) > 1)
    np.testing.assert_allclose(torque, [out.roll, out.pitch, out.yaw], atol=2)
    np.testing.assert_allclose(pos_err_i, [ref.i_error_x, ref.i_error_y, ref.i_error_z], atol=1e-6)
    ref_r_int_error = [ref.i_error_m_x, -ref.i_error_m_y, ref.i_error_m_z]
    np.testing.assert_allclose(r_int_error, ref_r_int_error, atol=1e-6)


@given(sample=_samples())
def test_mixer(ref: cf.controllerMellinger_t, sample: tuple[NDArray, ...]):
    """Compare the motor PWMs for identical controller outputs."""
    out, ref_out = _reference_step(ref, *sample), cf.motors_thrust_uncapped_t()
    cf.powerDistribution(out, ref_out)
    mixing_matrix = load_fn_params(control.attitude2force_torque, Drone.cf2x_L250, xp=jnp)[
        "mixing_matrix"
    ]
    # The firmware runs in the legacy Crazyflie frame with flipped pitch and yaw torques
    torque = jnp.array([out.roll, -out.pitch, -out.yaw])
    motors = jax.jit(control.force_torque_pwms2pwms)(out.thrust, torque, mixing_matrix)
    m = ref_out.motors
    np.testing.assert_allclose(motors, [m.m1, m.m2, m.m3, m.m4], atol=3)


@pytest.mark.xfail(
    strict=True,
    reason="TODO: Clip as the firmware, uniform above the maximum and at 0, and use the current "
    "and previous body rate setpoint",
)
@given(sample=_samples())
def test_end_to_end(ref: cf.controllerMellinger_t, sample: tuple[NDArray, ...]):
    """Compare the rotor velocities from the returned force and torque."""
    pos, vel, quat, ang_vel, cmd, prev_ang_vel, integrators, _ = map(jnp.asarray, sample)
    # Jitting the module functions keeps JAX's compilation cache across examples
    state_params = load_fn_params(control.state2attitude, Drone.cf2x_L250, xp=jnp)
    attitude_params = load_fn_params(control.attitude2force_torque, Drone.cf2x_L250, xp=jnp)
    state2attitude = partial(jax.jit(control.state2attitude), **state_params)
    attitude2force_torque = partial(jax.jit(control.attitude2force_torque), **attitude_params)
    rotor_params = load_fn_params(control.force_torque2rotor_vel, Drone.cf2x_L250, xp=jnp)
    force_torque2rotor_vel = partial(jax.jit(control.force_torque2rotor_vel), **rotor_params)
    rpyt, _ = state2attitude(pos, quat, vel, cmd, integrators[:3], cf.ATTITUDE_RATE)
    force, torque, _ = attitude2force_torque(quat, ang_vel, rpyt, prev_ang_vel, integrators[3:])
    rotor_vel = force_torque2rotor_vel(force, torque)
    out = _reference_step(ref, *sample)
    # See test_controller for why we assume a minimum thrust for torque comparison
    assume(abs(out.thrust) > 1)
    uncapped, capped = cf.motors_thrust_uncapped_t(), cf.motors_thrust_pwm_t()
    cf.powerDistribution(out, uncapped)
    cf.powerDistributionCap(uncapped, capped)  # We always assume perfect battery compensation
    m = capped.motors
    pwms = np.array([m.m1, m.m2, m.m3, m.m4])
    motor_forces = pwm2force(pwms, rotor_params["thrust_max"], attitude_params["pwm_max"])
    ref_rotor_vel = motor_force2rotor_vel(motor_forces, rotor_params["rpm2thrust"])
    np.testing.assert_allclose(rotor_vel, ref_rotor_vel, atol=2)  # Tolerance in RPM
