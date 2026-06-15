"""Second-order fitted RPY dynamics model (no rotor dynamics).

This module implements a simplified quadrotor dynamics model where the rotational
dynamics are modelled as a fitted second-order linear system driven by roll, pitch,
and yaw (RPY) commands, and the translational dynamics are driven by the collective
thrust command.  No motor spin-up dynamics are modelled.

The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``.

Both a numeric implementation ([dynamics][crazyflow.models.so_rpy.dynamics]) and symbolic CasADi
implementations ([symbolic_dynamics][crazyflow.models.so_rpy.symbolic_dynamics],
[symbolic_dynamics_euler][crazyflow.models.so_rpy.symbolic_dynamics_euler]) are provided.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import casadi as cs
from array_api_compat import array_namespace, device
from scipy.spatial.transform import Rotation as R

import crazyflow.models.symbols as symbols
from crazyflow.models.core import supports
from crazyflow.models.utils import rotation, to_xp

if TYPE_CHECKING:
    from crazyflow.models._typing import Array  # To be changed to array_api_typing later


@supports(rotor_dynamics=False)
def dynamics(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    cmd: Array,
    rotor_vel: Array | None = None,
    dist_f: Array | None = None,
    dist_t: Array | None = None,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
) -> tuple[Array, Array, Array, Array, Array | None]:
    """The fitted model with linear, second order rpy dynamics.

    Args:
        pos: Position of the drone (m).
        quat: Quaternion of the drone (xyzw).
        vel: Velocity of the drone (m/s).
        ang_vel: Angular velocity of the drone (rad/s).
        cmd: Roll pitch yaw (rad) and collective thrust (N) command.
        rotor_vel: Speed of the 4 motors (RPMs). Kept for compatibility with the model signature.
        dist_f: Disturbance force (N) in the world frame acting on the CoM.
        dist_t: Disturbance torque (Nm) in the world frame acting on the CoM.

        mass: Mass of the drone (kg).
        gravity_vec: Gravity vector (m/s^2). We assume the gravity vector points downwards, e.g.
            [0, 0, -9.81].
        J: Inertia matrix (kg m^2).
        J_inv: Inverse inertia matrix (1/kg m^2).
        acc_coef: Coefficient for the acceleration (1/s^2).
        cmd_f_coef: Coefficient for the collective thrust (N/rad^2).
        rpy_coef: Coefficient for the roll pitch yaw dynamics (1/s).
        rpy_rates_coef: Coefficient for the roll pitch yaw rates dynamics (1/s^2).
        cmd_rpy_coef: Coefficient for the roll pitch yaw command dynamics (1/s).

    Returns:
        The derivatives of all state variables.
    """
    xp = array_namespace(pos)
    # Convert parameters to correct xp framework
    mass, gravity_vec, J, J_inv = to_xp(mass, gravity_vec, J, J_inv, xp=xp, device=device(pos))
    acc_coef, cmd_f_coef, rpy_coef, rpy_rates_coef, cmd_rpy_coef = to_xp(
        acc_coef, cmd_f_coef, rpy_coef, rpy_rates_coef, cmd_rpy_coef, xp=xp, device=device(pos)
    )
    cmd_f = cmd[..., -1]
    cmd_rpy = cmd[..., 0:3]
    rot = R.from_quat(quat)
    euler_angles = rot.as_euler("xyz")

    rotor_vel_dot = None
    thrust = acc_coef + cmd_f_coef * cmd_f

    drone_z_axis = rot.as_matrix()[..., -1]

    pos_dot = vel
    vel_dot = 1.0 / mass * thrust[..., None] * drone_z_axis + gravity_vec

    if dist_f is not None:
        vel_dot = vel_dot + dist_f / mass  # Adding force disturbances to the state
    vel_dot = xp.asarray(vel_dot)

    # Rotational equation of motion
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    rpy_rates = rotation.ang_vel2rpy_rates(quat, ang_vel)
    rpy_rates_dot = rpy_coef * euler_angles + rpy_rates_coef * rpy_rates + cmd_rpy_coef * cmd_rpy
    ang_vel_dot = rotation.rpy_rates_deriv2ang_vel_deriv(quat, rpy_rates, rpy_rates_dot)
    if dist_t is not None:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque given the inertia matrix
        torque = (J @ ang_vel_dot[..., None])[..., 0]
        torque = torque + xp.linalg.cross(ang_vel, (J @ ang_vel[..., None])[..., 0])
        # adding torque. TODO: This should be a linear transformation. Can't we just transform the
        # disturbance torque to an ang_vel_dot summand directly?
        torque = torque + rot.apply(dist_t, inverse=True)
        # back to angular acceleration
        torque = torque - xp.linalg.cross(ang_vel, (J @ ang_vel[..., None])[..., 0])
        ang_vel_dot = (J_inv @ torque[..., None])[..., 0]

    return pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot


def symbolic_dynamics(
    model_rotor_vel: bool = False,
    model_dist_f: bool = False,
    model_dist_t: bool = False,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """Return CasADi symbolic expressions for the so_rpy model in quaternion form.

    Internally delegates to
    [symbolic_dynamics_euler][crazyflow.models.so_rpy.symbolic_dynamics_euler] and converts the
    Euler-angle state to quaternion + angular-velocity state so that the interface matches that of
    [symbolic_dynamics][crazyflow.models.first_principles.symbolic_dynamics].

    Args:
        model_rotor_vel: If ``True``, a scalar rotor-velocity state is appended
            to ``X`` (for interface compatibility only — ``so_rpy`` has no thrust
            dynamics and will log a warning).  Defaults to ``False``.
        model_dist_f: If ``True``, a 3-D force disturbance is appended to ``X``.
        model_dist_t: If ``True``, a 3-D torque disturbance is appended to ``X``.
        mass: Drone mass in kg.
        gravity_vec: Gravity vector, shape ``(3,)``.
        J: Inertia matrix, shape ``(3, 3)``.
        J_inv: Inverse inertia matrix, shape ``(3, 3)``.
        acc_coef: Scalar acceleration offset coefficient.
        cmd_f_coef: Collective-thrust-to-acceleration coefficient.
        rpy_coef: RPY state feedback coefficient, shape ``(3,)``.
        rpy_rates_coef: RPY-rate feedback coefficient, shape ``(3,)``.
        cmd_rpy_coef: RPY command feedforward coefficient, shape ``(3,)``.

    Returns:
        Tuple ``(X_dot, X, U, Y)`` of CasADi ``MX`` expressions:

        * ``X_dot``: State derivative, length 13 (or more with disturbance states).
        * ``X``: State vector ``[pos(3), quat(4), vel(3), ang_vel(3)]``.
        * ``U``: Input vector ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``.
        * ``Y``: Output ``[pos(3), quat(4)]``.
    """
    # We need to set the rpy and drpy symbols before building the euler model
    _saved_rpy = symbols.rpy
    _saved_drpy = symbols.drpy
    _rpy_quat = rotation.cs_quat2euler(symbols.quat)
    _drpy_quat = rotation.cs_ang_vel2rpy_rates(symbols.quat, symbols.ang_vel)
    symbols.rpy = _rpy_quat
    symbols.drpy = _drpy_quat
    X_dot_euler, X_euler, U_euler, Y_euler = symbolic_dynamics_euler(
        model_rotor_vel=model_rotor_vel,
        mass=mass,
        gravity_vec=gravity_vec,
        J=J,
        J_inv=J_inv,
        acc_coef=acc_coef,
        cmd_f_coef=cmd_f_coef,
        rpy_coef=rpy_coef,
        rpy_rates_coef=rpy_rates_coef,
        cmd_rpy_coef=cmd_rpy_coef,
    )
    symbols.rpy = _saved_rpy
    symbols.drpy = _saved_drpy

    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.quat, symbols.vel, symbols.ang_vel)
    if model_rotor_vel:
        logging.getLogger(__name__).warning("The so_rpy model does not support thrust dynamics")
        X = cs.vertcat(X, symbols.rotor_vel)
    if model_dist_f:
        X = cs.vertcat(X, symbols.dist_f)
    if model_dist_t:
        X = cs.vertcat(X, symbols.dist_t)
    U = U_euler

    # Linear equation of motion
    pos_dot = X_dot_euler[0:3]
    vel_dot = X_dot_euler[6:9]
    if model_dist_f:
        # Adding force disturbances to the state
        vel_dot = vel_dot + symbols.dist_f / mass

    # Rotational equation of motion
    xi = cs.vertcat(
        cs.horzcat(0, -symbols.ang_vel.T), cs.horzcat(symbols.ang_vel, -cs.skew(symbols.ang_vel))
    )
    quat_dot = 0.5 * (xi @ symbols.quat)
    ang_vel_dot = rotation.cs_rpy_rates_deriv2ang_vel_deriv(
        symbols.quat, _drpy_quat, X_dot_euler[9:12]
    )
    if model_dist_t:
        # adding torque disturbances to the state
        # angular acceleration can be converted to total torque
        torque = J @ ang_vel_dot + cs.cross(symbols.ang_vel, J @ symbols.ang_vel)
        # adding torque
        torque = torque + symbols.rot.T @ symbols.dist_t
        # back to angular acceleration
        ang_vel_dot = J_inv @ (torque - cs.cross(symbols.ang_vel, J @ symbols.ang_vel))

    X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(symbols.pos, symbols.quat)

    return X_dot, X, U, Y


def symbolic_dynamics_euler(
    model_rotor_vel: bool = False,
    *,
    mass: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    acc_coef: Array,
    cmd_f_coef: Array,
    rpy_coef: Array,
    rpy_rates_coef: Array,
    cmd_rpy_coef: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """Return CasADi symbolic expressions for the so_rpy model in Euler-angle form.

    This is the native representation of the ``so_rpy`` model.  The state uses
    roll/pitch/yaw and their rates rather than quaternion + angular velocity,
    which avoids trigonometric overhead inside CasADi-based solvers.

    Args:
        model_rotor_vel: If ``True``, a scalar rotor-velocity state is appended
            to ``X`` for interface compatibility (no dynamics are modelled).
        mass: Drone mass in kg.
        gravity_vec: Gravity vector, shape ``(3,)``.
        J: Inertia matrix, shape ``(3, 3)``.
        J_inv: Inverse inertia matrix, shape ``(3, 3)``.
        acc_coef: Scalar acceleration offset coefficient.
        cmd_f_coef: Collective-thrust-to-acceleration coefficient.
        rpy_coef: RPY state feedback coefficient, shape ``(3,)``.
        rpy_rates_coef: RPY-rate feedback coefficient, shape ``(3,)``.
        cmd_rpy_coef: RPY command feedforward coefficient, shape ``(3,)``.

    Returns:
        Tuple ``(X_dot, X, U, Y)`` of CasADi ``MX`` expressions:

        * ``X_dot``: State derivative, length 12.
        * ``X``: State vector ``[pos(3), rpy(3), vel(3), drpy(3)]``.
        * ``U``: Input vector ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``.
        * ``Y``: Output ``[pos(3), rpy(3)]``.
    """
    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.rpy, symbols.vel, symbols.drpy)
    if model_rotor_vel:
        logging.getLogger(__name__).warning("The so_rpy model does not support thrust dynamics")
        X = cs.vertcat(X, symbols.rotor_vel)
    U = symbols.cmd_rpyt
    cmd_rpy = U[:3]
    cmd_thrust = U[-1]
    rot = rotation.cs_rpy2matrix(symbols.rpy)

    # Defining the dynamics function
    forces_motor = cmd_thrust

    # Creating force vector
    forces_motor_vec = cs.vertcat(0, 0, acc_coef + cmd_f_coef * forces_motor)

    # Linear equation of motion
    pos_dot = symbols.vel
    vel_dot = rot @ forces_motor_vec / mass + gravity_vec

    ddrpy = rpy_coef * symbols.rpy + rpy_rates_coef * symbols.drpy + cmd_rpy_coef * cmd_rpy

    X_dot = cs.vertcat(pos_dot, symbols.drpy, vel_dot, ddrpy)
    Y = cs.vertcat(symbols.pos, symbols.rpy)

    return X_dot, X, U, Y
