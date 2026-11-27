"""Compare the Mellinger controller against the Crazyflie firmware."""

from __future__ import annotations

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

from crazyflow.control import Control, load_fn_params, load_params, parametrize
from crazyflow.control.mellinger import control
from crazyflow.control.transform import force2pwm, motor_force2rotor_vel, pwm2force
from crazyflow.drones import Drone
from crazyflow.sim import Sim

if TYPE_CHECKING:
    from numpy.typing import NDArray


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


def _ctrl_states() -> st.SearchStrategy[tuple[NDArray, ...]]:
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
    setpoint: cf.setpoint_t,
    pos: NDArray,
    vel: NDArray,
    quat: NDArray,
    ang_vel: NDArray,
    pos_err_i: NDArray,
    rot_err_i: NDArray,
    prev_ang_vel: NDArray,
    prev_ang_vel_des: NDArray,
) -> tuple[cf.control_t, cf.controllerMellinger_t]:
    """Run one step of a new firmware controller from the given state and integrators."""
    ctrl = cf.controllerMellinger_t()
    cf.controllerMellingerInit(ctrl)
    # The firmware defaults to the mass of a bare drone, crazyflow to the mass of the model
    ctrl.mass = load_params("mellinger", Drone.cf2x_L250)["core"]["mass"]
    ctrl.i_error_x, ctrl.i_error_y, ctrl.i_error_z = pos_err_i.tolist()
    # The firmware runs in the legacy Crazyflie frame with a flipped pitch axis
    ctrl.i_error_m_x, ctrl.i_error_m_y, ctrl.i_error_m_z = (rot_err_i * [1, -1, 1]).tolist()
    ctrl.prev_omega_roll, ctrl.prev_omega_pitch = (prev_ang_vel[:2] * [1, -1]).tolist()
    ctrl.prev_setpoint_omega_roll, ctrl.prev_setpoint_omega_pitch = prev_ang_vel_des[:2].tolist()
    state, sensors = cf.state_t(), cf.sensorData_t()
    state.position.x, state.position.y, state.position.z = pos.tolist()
    state.velocity.x, state.velocity.y, state.velocity.z = vel.tolist()
    q = state.attitudeQuaternion
    q.x, q.y, q.z, q.w = quat.tolist()
    # The state estimator converts the quaternion with the firmware's own Euler angle convention
    state.attitude.yaw = float(np.rad2deg(cf.quat2rpy(cf.mkquat(*quat.tolist())).z))
    sensors.gyro.x, sensors.gyro.y, sensors.gyro.z = np.rad2deg(ang_vel).tolist()
    out = cf.control_t()
    cf.controllerMellinger(ctrl, out, setpoint, sensors, state, 0)  # 0 to step the controller
    return out, ctrl


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


@given(state=_states(), ctrl_state=_ctrl_states(), cmd=_state_cmds())
def test_state2attitude(state: tuple[NDArray, ...], ctrl_state: tuple[NDArray, ...], cmd: NDArray):
    """Compare thrust, desired z axis and position integrators for state commands."""
    pos, vel, quat, _ = state
    state2attitude = parametrize(control.state2attitude, Drone.cf2x_L250)
    rpyt, pos_err_i = state2attitude(pos, quat, vel, cmd, ctrl_state[0], cf.ATTITUDE_RATE)
    # The firmware outputs the thrust in PWM, as its fraction of the maximum collective thrust
    params = load_params("mellinger", Drone.cf2x_L250)["core"]
    thrust = force2pwm(rpyt[3], 4 * params["thrust_max"], params["pwm_max"])
    setpoint = cf.setpoint_t()
    sp_pos = setpoint.position
    sp_pos.x, sp_pos.y, sp_pos.z = cmd[:3].tolist()
    sp_vel = setpoint.velocity
    sp_vel.x, sp_vel.y, sp_vel.z = cmd[3:6].tolist()
    sp_acc = setpoint.acceleration
    sp_acc.x, sp_acc.y, sp_acc.z = cmd[6:9].tolist()
    sp_att = setpoint.attitude
    sp_att.yaw = R.from_quat(cmd[9:13]).as_euler("xyz", degrees=True, suppress_warnings=True)[2]
    rate = setpoint.attitudeRate
    rate.roll, rate.pitch, rate.yaw = np.rad2deg(cmd[13:16]).tolist()
    setpoint.mode.x = setpoint.mode.y = setpoint.mode.z = setpoint.mode.yaw = cf.modeAbs
    out, ref = _reference_step(setpoint, *state, *ctrl_state)
    np.testing.assert_allclose(thrust, out.thrust, atol=1)
    z_axis, ref_z_axis = R.from_euler("xyz", rpyt[:3]).as_matrix()[:, 2], ref.z_axis_desired
    np.testing.assert_allclose(z_axis, [ref_z_axis.x, ref_z_axis.y, ref_z_axis.z], atol=1e-6)
    # Integrators reset on zero thrust, but crazyflow can be off by ~1 PWM. Resets are thus tested
    # only for thrusts with margin towards zero.
    assume(abs(out.thrust) > 1)
    np.testing.assert_allclose(pos_err_i, [ref.i_error_x, ref.i_error_y, ref.i_error_z], atol=1e-6)


@given(state=_states(), ctrl_state=_ctrl_states(), cmd=_attitude_cmds())
def test_attitude2force_torque(
    state: tuple[NDArray, ...], ctrl_state: tuple[NDArray, ...], cmd: NDArray
):
    """Compare thrust, torques and attitude integrators for attitude commands."""
    _, _, quat, ang_vel = state
    _, rot_err_i, prev_ang_vel, _ = ctrl_state
    params = load_fn_params(control.attitude2force_torque, Drone.cf2x_L250)
    attitude2force_torque = parametrize(control.attitude2force_torque, Drone.cf2x_L250)
    # The firmware outputs thrust and torques in PWM, whereas crazyflow's controller returns them
    # after mixing and clipping in SI units. To get access to the intermediate PWM values in
    # crazyflow's controllers, we patch the intermediate functions that get called with the PWM
    # values and compare their arguments instead of the final output. Monkey-patching is
    # incompatible with jit, so we use numpy here
    mixer = control.force_torque_pwms2pwms
    with patch.object(control, "force_torque_pwms2pwms", wraps=mixer) as spy:
        _, _, rot_err_i = attitude2force_torque(quat, ang_vel, cmd, prev_ang_vel, rot_err_i)
    thrust = spy.call_args.args[0]
    # The firmware runs in the legacy Crazyflie frame with flipped pitch and yaw torques. Pitch uses
    # the legacy frame, yaw inverts later in the controller. See
    # https://github.com/bitcraze/crazyflie-firmware/blob/d794c5ca081d19aa1e1bbcb6fd4a3711784fb8b1/src/modules/src/controller/controller_mellinger.c#L309
    torque = spy.call_args.args[1] * [1, -1, -1]
    setpoint = cf.setpoint_t()
    # The firmware takes the pitch setpoint in the legacy Crazyflie frame
    att = setpoint.attitude
    att.roll, att.pitch, att.yaw = (np.rad2deg(cmd[:3]) * [1, -1, 1]).tolist()
    setpoint.thrust = float(force2pwm(cmd[3] / 4, params["thrust_max"], params["pwm_max"]))
    setpoint.mode.roll = setpoint.mode.pitch = setpoint.mode.yaw = cf.modeAbs
    # Crazyflow does not allow controllers to switch mode in between, so attitude_rate must be zero
    out, ref = _reference_step(setpoint, *state, *ctrl_state[:3], np.zeros(3, np.float32))
    np.testing.assert_allclose(thrust, out.thrust, atol=1)
    # Zero thrust turns off torques. If the firmware and crazyflow controller disagree within
    # numerical noise, the torque contribution can still lead to significant differences. We hence
    # only compare torques if we have at least 1 PWM of thrust
    assume(abs(out.thrust) > 1)
    np.testing.assert_allclose(torque, [out.roll, out.pitch, out.yaw], atol=2)
    ref_rot_err_i = [ref.i_error_m_x, -ref.i_error_m_y, ref.i_error_m_z]
    np.testing.assert_allclose(rot_err_i, ref_rot_err_i, atol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="TODO: Report the firmware levelling upside down in body rate mode once its z thrust "
    "target turns negative",
)
@given(state=_states(), ctrl_state=_ctrl_states(), cmd=_body_rate_cmds())
def test_body_rate2force_torque(
    state: tuple[NDArray, ...], ctrl_state: tuple[NDArray, ...], cmd: NDArray
):
    """Compare thrust, torques and attitude integrators for body rate commands."""
    _, _, quat, ang_vel = state
    _, rot_err_i, prev_ang_vel, prev_ang_vel_des = ctrl_state
    params = load_fn_params(control.body_rate2force_torque, Drone.cf2x_L250)
    body_rate2force_torque = parametrize(control.body_rate2force_torque, Drone.cf2x_L250)
    prev_cmd = np.concatenate((prev_ang_vel_des, cmd[3:]))
    # See test_attitude2force_torque for why we spy on the intermediate PWM values
    mixer = control.force_torque_pwms2pwms
    with patch.object(control, "force_torque_pwms2pwms", wraps=mixer) as spy:
        _, _, rot_err_i = body_rate2force_torque(
            quat, ang_vel, cmd, prev_ang_vel, prev_cmd, rot_err_i
        )
    thrust = spy.call_args.args[0]
    # The firmware runs in the legacy Crazyflie frame with flipped pitch and yaw torques
    torque = spy.call_args.args[1] * [1, -1, -1]
    setpoint = cf.setpoint_t()
    rate = setpoint.attitudeRate
    rate.roll, rate.pitch, rate.yaw = np.rad2deg(cmd[:3]).tolist()
    setpoint.thrust = float(force2pwm(cmd[3] / 4, params["thrust_max"], params["pwm_max"]))
    setpoint.mode.roll = setpoint.mode.pitch = setpoint.mode.yaw = cf.modeVelocity
    out, ref = _reference_step(setpoint, *state, *ctrl_state)
    np.testing.assert_allclose(thrust, out.thrust, atol=1)
    # See test_attitude2force_torque for why we assume a minimum thrust for torque comparison
    assume(abs(out.thrust) > 1)
    # The yaw is undefined at 90 degrees pitch
    pitch = R.from_quat(quat).as_euler("xyz", suppress_warnings=True)[1]
    assume(abs(pitch) < np.pi / 2 - 1e-3)
    np.testing.assert_allclose(torque, [out.roll, out.pitch, out.yaw], atol=2)
    ref_rot_err_i = [ref.i_error_m_x, -ref.i_error_m_y, ref.i_error_m_z]
    np.testing.assert_allclose(rot_err_i, ref_rot_err_i, atol=1e-6)


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


@pytest.fixture(scope="module")
def sim() -> Sim:
    """Simulate one cf2x_L250 with the state controller at the firmware's controller rate."""
    sim = Sim(drone=Drone.cf2x_L250, control=Control.state, state_freq=cf.ATTITUDE_RATE)
    sim.step()  # Compile outside the examples to stay within the hypothesis deadline
    return sim


@given(state=_states(), ctrl_state=_ctrl_states(), cmd=_state_cmds())
def test_end_to_end(
    sim: Sim, state: tuple[NDArray, ...], ctrl_state: tuple[NDArray, ...], cmd: NDArray
):
    """Compare the rotor velocities of a simulated state control step."""
    # The setpoint yaw is undefined at 90 degrees pitch
    pitch_des = R.from_quat(cmd[9:13]).as_euler("xyz", suppress_warnings=True)[1]
    assume(abs(pitch_des) < np.pi / 2 - 1e-3)
    pos, vel, quat, ang_vel = jax.device_put([x[None, None] for x in state], sim.device)
    pos_err_i, rot_err_i, prev_ang_vel, prev_ang_vel_des = jax.device_put(
        [x[None, None] for x in ctrl_state], sim.device
    )
    sim.reset()
    data = sim.data
    controls = sim.data.controls
    data = data.replace(states=data.states.replace(pos=pos, vel=vel, quat=quat, ang_vel=ang_vel))
    controls = controls.replace(state=data.controls.state.replace(pos_err_i=pos_err_i))
    controls = controls.replace(
        attitude=data.controls.attitude.replace(
            rot_err_i=rot_err_i, prev_ang_vel=prev_ang_vel, ang_vel_des=prev_ang_vel_des
        )
    )
    sim.data = data.replace(controls=controls)
    sim.state_control(jax.device_put(cmd[None, None], sim.device))
    sim.step()
    rotor_vel = sim.data.controls.rotor_vel[0, 0]
    params = load_params("mellinger", Drone.cf2x_L250)["core"]
    setpoint = cf.setpoint_t()
    sp_pos = setpoint.position
    sp_pos.x, sp_pos.y, sp_pos.z = cmd[:3].tolist()
    sp_vel = setpoint.velocity
    sp_vel.x, sp_vel.y, sp_vel.z = cmd[3:6].tolist()
    sp_acc = setpoint.acceleration
    sp_acc.x, sp_acc.y, sp_acc.z = cmd[6:9].tolist()
    sp_att = setpoint.attitude
    sp_att.yaw = R.from_quat(cmd[9:13]).as_euler("xyz", degrees=True, suppress_warnings=True)[2]
    rate = setpoint.attitudeRate
    rate.roll, rate.pitch, rate.yaw = np.rad2deg(cmd[13:16]).tolist()
    setpoint.mode.x = setpoint.mode.y = setpoint.mode.z = setpoint.mode.yaw = cf.modeAbs
    out, _ = _reference_step(setpoint, *state, *ctrl_state)
    # See test_attitude2force_torque for why we assume a minimum thrust for torque comparison
    assume(abs(out.thrust) > 1)
    uncapped, capped = cf.motors_thrust_uncapped_t(), cf.motors_thrust_pwm_t()
    cf.powerDistribution(out, uncapped)
    # cffirmware does not run the battery compensation. The battery compensation turns off motors
    # below the minimum thrust. We assume perfect compensation, so we clip manually
    m = uncapped.motors
    pwms = [m.m1, m.m2, m.m3, m.m4]
    m.m1, m.m2, m.m3, m.m4 = [0 if pwm < params["pwm_min"] else pwm for pwm in (pwms)]
    cf.powerDistributionCap(uncapped, capped)
    m = capped.motors
    pwms = np.array([m.m1, m.m2, m.m3, m.m4])
    motor_forces = pwm2force(pwms, params["thrust_max"], params["pwm_max"])
    ref_rotor_vel = motor_force2rotor_vel(motor_forces, params["rpm2thrust"])
    np.testing.assert_allclose(rotor_vel, ref_rotor_vel, atol=2)  # Tolerance in RPM
