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
from crazyflow.control.transform import force2pwm, motor_force2rotor_vel, pwm2force
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
def _rpy(draw: st.DrawFn) -> NDArray:
    """Draw roll, pitch and yaw combining special and random angles."""
    special = st.sampled_from([0.0, np.pi / 2, -np.pi / 2, np.pi, -np.pi])
    angle = st.one_of(special, st.floats(-0.1, 0.1), st.floats(-np.pi, np.pi))
    return np.array(draw(st.tuples(angle, angle, angle)), np.float32)


@st.composite
def _states(draw: st.DrawFn) -> tuple[NDArray, ...]:
    """Draw position, velocity, attitude quaternion and body rates."""
    # float32 so both controllers get identical inputs
    quat = R.from_euler("xyz", draw(_rpy())).as_quat().astype(np.float32)
    return draw(_vector(10)), draw(_vector(5)), quat, draw(_vector(35))


def _carry_overs() -> st.SearchStrategy[tuple[NDArray, ...]]:
    """Draw integrators, previous body rates and previous body rate setpoints."""
    # Beyond the integrator limits to cover clamping
    return st.tuples(_vector(3), _vector(2000), _vector(35), _vector(35))


@st.composite
def _state_cmds(draw: st.DrawFn) -> NDArray:
    """Draw full state commands."""
    quat_des = R.from_euler("xyz", draw(_rpy())).as_quat().astype(np.float32)
    acc_des = draw(_vector(20))  # Includes free fall
    return np.concatenate(
        (draw(_vector(10)), draw(_vector(5)), acc_des, quat_des, draw(_vector(35)))
    )


@st.composite
def _attitude_cmds(draw: st.DrawFn) -> NDArray:
    """Draw attitude commands of roll, pitch, yaw and thrust."""
    return np.concatenate((draw(_rpy()), draw(_vector(0.5, 1))))


@st.composite
def _body_rate_cmds(draw: st.DrawFn) -> NDArray:
    """Draw body rate commands of body rates and thrust."""
    return np.concatenate((draw(_vector(35)), draw(_vector(0.5, 1))))


def _reference_step(
    ctrl: cf.controllerMellinger_t,
    setpoint: cf.setpoint_t,
    pos: NDArray,
    vel: NDArray,
    quat: NDArray,
    ang_vel: NDArray,
    pos_err_i: NDArray,
    r_int_error: NDArray,
    prev_ang_vel: NDArray,
    prev_rate_des: NDArray,
) -> cf.control_t:
    """Run one firmware controller step from the given state and carry-over."""
    ctrl.i_error_x, ctrl.i_error_y, ctrl.i_error_z = pos_err_i.tolist()
    # The firmware runs in the legacy Crazyflie frame with a flipped pitch axis
    ctrl.i_error_m_x, ctrl.i_error_m_y, ctrl.i_error_m_z = (r_int_error * [1, -1, 1]).tolist()
    ctrl.prev_omega_roll, ctrl.prev_omega_pitch = (prev_ang_vel[:2] * [1, -1]).tolist()
    ctrl.prev_setpoint_omega_roll, ctrl.prev_setpoint_omega_pitch = prev_rate_des[:2].tolist()
    state, sensors = cf.state_t(), cf.sensorData_t()
    state.position.x, state.position.y, state.position.z = pos.tolist()
    state.velocity.x, state.velocity.y, state.velocity.z = vel.tolist()
    q = state.attitudeQuaternion
    q.x, q.y, q.z, q.w = quat.tolist()
    # The state estimator converts the quaternion with the firmware's own Euler angle convention
    state.attitude.yaw = float(np.degrees(cf.quat2rpy(cf.mkquat(*quat.tolist())).z))
    sensors.gyro.x, sensors.gyro.y, sensors.gyro.z = np.degrees(ang_vel).tolist()
    out = cf.control_t()
    cf.controllerMellinger(ctrl, out, setpoint, sensors, state, 0)
    return out


@pytest.mark.xfail(strict=True, reason="TODO: The Crazyflies use kd_z = 0.5, the firmware 0.4")
@pytest.mark.parametrize(
    "drone", [Drone.cf2x_L250, Drone.cf2x_P250, Drone.cf2x_T350, Drone.cf21B_500]
)
def test_params(drone: Drone):
    """Compare the gains with the firmware defaults."""
    ctrl = cf.controllerMellinger_t()
    cf.controllerMellingerInit(ctrl)
    state_params = load_fn_params(control.state2attitude, drone, xp=jnp)
    attitude_params = load_fn_params(control.attitude2force_torque, drone, xp=jnp)
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
    np.testing.assert_allclose(state_params["mass_thrust"], ctrl.massThrust, rtol=1e-6)
    # The firmware is built for the Crazyflie 2 without the thrust upgrade kit
    if drone in (Drone.cf2x_L250, Drone.cf2x_P250):
        np.testing.assert_allclose(state_params["mass"], ctrl.mass, rtol=1e-6)


@pytest.mark.xfail(strict=True, reason="TODO: Reset the integrators at zero thrust as the firmware")
@given(state=_states(), carry_over=_carry_overs(), cmd=_state_cmds())
def test_state2attitude(
    ref: cf.controllerMellinger_t,
    state: tuple[NDArray, ...],
    carry_over: tuple[NDArray, ...],
    cmd: NDArray,
):
    """Compare thrust, desired z axis and position integrators for state commands."""
    pos, vel, quat, _ = state
    state2attitude = parametrize(control.state2attitude, Drone.cf2x_L250)
    # The firmware outputs thrust and torques in PWM, whereas crazyflow's controller returns them
    # after mixing and clipping in SI units. To get access to the intermediate PWM values in
    # crazyflow's controllers, we patch the intermediate functions that get called with the PWM
    # values and compare their arguments instead of the final output. Monkey-patching is
    # incompatible with jit, so we use numpy here
    with patch.object(control, "pwm2force", wraps=control.pwm2force) as spy:
        rpyt, pos_err_i = state2attitude(pos, quat, vel, cmd, carry_over[0], cf.ATTITUDE_RATE)
    thrust = spy.call_args.args[0]
    setpoint = cf.setpoint_t()
    setpoint.position.x, setpoint.position.y, setpoint.position.z = cmd[:3].tolist()
    setpoint.velocity.x, setpoint.velocity.y, setpoint.velocity.z = cmd[3:6].tolist()
    acc = setpoint.acceleration
    acc.x, acc.y, acc.z = cmd[6:9].tolist()
    yaw = R.from_quat(cmd[9:13]).as_euler("xyz", suppress_warnings=True)[2]
    setpoint.attitude.yaw = float(np.degrees(yaw))
    rate = setpoint.attitudeRate
    rate.roll, rate.pitch, rate.yaw = np.degrees(cmd[13:16]).tolist()
    setpoint.mode.x = setpoint.mode.y = setpoint.mode.z = setpoint.mode.yaw = cf.modeAbs
    out = _reference_step(ref, setpoint, *state, *carry_over)
    np.testing.assert_allclose(thrust, out.thrust, atol=1)
    z_axis, ref_z_axis = R.from_euler("xyz", rpyt[:3]).as_matrix()[:, 2], ref.z_axis_desired
    np.testing.assert_allclose(z_axis, [ref_z_axis.x, ref_z_axis.y, ref_z_axis.z], atol=1e-6)
    # Integrators reset on zero thrust, but crazyflow can be off by ~1 PWM. Resets are thus tested
    # only for thrusts with margin towards zero.
    assume(abs(out.thrust) > 1)
    np.testing.assert_allclose(pos_err_i, [ref.i_error_x, ref.i_error_y, ref.i_error_z], atol=1e-6)


@pytest.mark.xfail(strict=True, reason="TODO: Reset the integrators at zero thrust as the firmware")
@given(state=_states(), carry_over=_carry_overs(), cmd=_attitude_cmds())
def test_attitude2force_torque(
    ref: cf.controllerMellinger_t,
    state: tuple[NDArray, ...],
    carry_over: tuple[NDArray, ...],
    cmd: NDArray,
):
    """Compare thrust, torques and attitude integrators for attitude commands."""
    _, _, quat, ang_vel = state
    _, r_int_error, prev_ang_vel, _ = carry_over
    params = load_fn_params(control.attitude2force_torque, Drone.cf2x_L250)
    attitude2force_torque = parametrize(control.attitude2force_torque, Drone.cf2x_L250)
    # See test_state2attitude for why we spy on the intermediate PWM values
    mixer = control.force_torque_pwms2pwms
    with patch.object(control, "force_torque_pwms2pwms", wraps=mixer) as spy:
        _, _, r_int_error = attitude2force_torque(quat, ang_vel, cmd, prev_ang_vel, r_int_error)
    thrust = spy.call_args.args[0]
    # The firmware runs in the legacy Crazyflie frame with flipped pitch and yaw torques
    torque = spy.call_args.args[1] * [1, -1, -1]
    setpoint = cf.setpoint_t()
    # The firmware takes the pitch setpoint in the legacy Crazyflie frame
    att = setpoint.attitude
    att.roll, att.pitch, att.yaw = (np.degrees(cmd[:3]) * [1, -1, 1]).tolist()
    setpoint.thrust = float(force2pwm(cmd[3] / 4, params["thrust_max"], params["pwm_max"]))
    setpoint.mode.roll = setpoint.mode.pitch = setpoint.mode.yaw = cf.modeAbs
    # Crazyflow does not allow controllers to switch mode in between, so attitude_rate must be zero
    out = _reference_step(ref, setpoint, *state, *carry_over[:3], np.zeros(3, np.float32))
    np.testing.assert_allclose(thrust, out.thrust, atol=1)
    # Zero thrust turns off torques. If the firmware and crazyflow controller disagree within
    # numerical noise, the torque contribution can still lead to significant differences. We hence
    # only compare torques if we have at least 1 PWM of thrust
    assume(abs(out.thrust) > 1)
    np.testing.assert_allclose(torque, [out.roll, out.pitch, out.yaw], atol=2)
    ref_r_int_error = [ref.i_error_m_x, -ref.i_error_m_y, ref.i_error_m_z]
    np.testing.assert_allclose(r_int_error, ref_r_int_error, atol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="TODO: Report the firmware levelling upside down in body rate mode once its z thrust "
    "target turns negative, and reset the integrators at zero thrust as the firmware",
)
@given(state=_states(), carry_over=_carry_overs(), cmd=_body_rate_cmds())
def test_body_rate2force_torque(
    ref: cf.controllerMellinger_t,
    state: tuple[NDArray, ...],
    carry_over: tuple[NDArray, ...],
    cmd: NDArray,
):
    """Compare thrust, torques and attitude integrators for body rate commands."""
    _, _, quat, ang_vel = state
    _, r_int_error, prev_ang_vel, prev_rate_des = carry_over
    params = load_fn_params(control.body_rate2force_torque, Drone.cf2x_L250)
    body_rate2force_torque = parametrize(control.body_rate2force_torque, Drone.cf2x_L250)
    prev_cmd = np.concatenate((prev_rate_des, cmd[3:]))
    # See test_state2attitude for why we spy on the intermediate PWM values
    mixer = control.force_torque_pwms2pwms
    with patch.object(control, "force_torque_pwms2pwms", wraps=mixer) as spy:
        _, _, r_int_error = body_rate2force_torque(
            quat, ang_vel, cmd, prev_ang_vel, prev_cmd, r_int_error
        )
    thrust = spy.call_args.args[0]
    # The firmware runs in the legacy Crazyflie frame with flipped pitch and yaw torques
    torque = spy.call_args.args[1] * [1, -1, -1]
    setpoint = cf.setpoint_t()
    rate = setpoint.attitudeRate
    rate.roll, rate.pitch, rate.yaw = np.degrees(cmd[:3]).tolist()
    setpoint.thrust = float(force2pwm(cmd[3] / 4, params["thrust_max"], params["pwm_max"]))
    setpoint.mode.roll = setpoint.mode.pitch = setpoint.mode.yaw = cf.modeVelocity
    out = _reference_step(ref, setpoint, *state, *carry_over)
    np.testing.assert_allclose(thrust, out.thrust, atol=1)
    # See test_attitude2force_torque for why we assume a minimum thrust for torque comparison
    assume(abs(out.thrust) > 1)
    # The yaw is undefined at 90 degrees pitch, and both controllers level at the current yaw
    assume(abs(R.from_quat(quat).as_matrix()[2, 0]) < 1 - 1e-6)
    np.testing.assert_allclose(torque, [out.roll, out.pitch, out.yaw], atol=2)
    ref_r_int_error = [ref.i_error_m_x, -ref.i_error_m_y, ref.i_error_m_z]
    np.testing.assert_allclose(r_int_error, ref_r_int_error, atol=1e-6)


@given(thrust=_vector(70000, 1), torque=hnp.arrays(np.int16, 3))
def test_mixer(thrust: NDArray, torque: NDArray):
    """Compare the motor PWMs for identical controller outputs."""
    out, ref_out = cf.control_t(), cf.motors_thrust_uncapped_t()
    out.thrust = float(thrust[0])
    out.roll, out.pitch, out.yaw = torque.tolist()
    cf.powerDistribution(out, ref_out)
    params = load_fn_params(control.attitude2force_torque, Drone.cf2x_L250, xp=jnp)
    # The firmware runs in the legacy Crazyflie frame with flipped pitch and yaw torques
    torque_pwm = jnp.asarray(torque * [1, -1, -1])
    motors = jax.jit(control.force_torque_pwms2pwms)(
        out.thrust, torque_pwm, params["mixing_matrix"]
    )
    m = ref_out.motors
    np.testing.assert_allclose(motors, [m.m1, m.m2, m.m3, m.m4], atol=3)


@pytest.mark.xfail(
    strict=True,
    reason="TODO: Clip as the firmware, uniform above the maximum and at 0, and use the current "
    "and previous body rate setpoint",
)
@given(state=_states(), carry_over=_carry_overs(), cmd=_state_cmds())
def test_end_to_end(
    ref: cf.controllerMellinger_t,
    state: tuple[NDArray, ...],
    carry_over: tuple[NDArray, ...],
    cmd: NDArray,
):
    """Compare the rotor velocities for state commands."""
    pos, vel, quat, ang_vel = map(jnp.asarray, state)
    pos_err_i, r_int_error, prev_ang_vel, _ = map(jnp.asarray, carry_over)
    # Jitting the module functions keeps JAX's compilation cache across examples
    state_params = load_fn_params(control.state2attitude, Drone.cf2x_L250, xp=jnp)
    attitude_params = load_fn_params(control.attitude2force_torque, Drone.cf2x_L250, xp=jnp)
    state2attitude = partial(jax.jit(control.state2attitude), **state_params)
    attitude2force_torque = partial(jax.jit(control.attitude2force_torque), **attitude_params)
    rotor_params = load_fn_params(control.force_torque2rotor_vel, Drone.cf2x_L250, xp=jnp)
    force_torque2rotor_vel = partial(jax.jit(control.force_torque2rotor_vel), **rotor_params)
    rpyt, _ = state2attitude(pos, quat, vel, jnp.asarray(cmd), pos_err_i, cf.ATTITUDE_RATE)
    force, torque, _ = attitude2force_torque(quat, ang_vel, rpyt, prev_ang_vel, r_int_error)
    rotor_vel = force_torque2rotor_vel(force, torque)
    setpoint = cf.setpoint_t()
    setpoint.position.x, setpoint.position.y, setpoint.position.z = cmd[:3].tolist()
    setpoint.velocity.x, setpoint.velocity.y, setpoint.velocity.z = cmd[3:6].tolist()
    acc = setpoint.acceleration
    acc.x, acc.y, acc.z = cmd[6:9].tolist()
    yaw = R.from_quat(cmd[9:13]).as_euler("xyz", suppress_warnings=True)[2]
    setpoint.attitude.yaw = float(np.degrees(yaw))
    rate = setpoint.attitudeRate
    rate.roll, rate.pitch, rate.yaw = np.degrees(cmd[13:16]).tolist()
    setpoint.mode.x = setpoint.mode.y = setpoint.mode.z = setpoint.mode.yaw = cf.modeAbs
    out = _reference_step(ref, setpoint, *state, *carry_over)
    # See test_attitude2force_torque for why we assume a minimum thrust for torque comparison
    assume(abs(out.thrust) > 1)
    uncapped, capped = cf.motors_thrust_uncapped_t(), cf.motors_thrust_pwm_t()
    cf.powerDistribution(out, uncapped)
    cf.powerDistributionCap(uncapped, capped)  # We always assume perfect battery compensation
    m = capped.motors
    pwms = np.array([m.m1, m.m2, m.m3, m.m4])
    motor_forces = pwm2force(pwms, rotor_params["thrust_max"], attitude_params["pwm_max"])
    ref_rotor_vel = motor_force2rotor_vel(motor_forces, rotor_params["rpm2thrust"])
    np.testing.assert_allclose(rotor_vel, ref_rotor_vel, atol=2)  # Tolerance in RPM
