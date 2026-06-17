"""First-principles dynamics-based quadrotor dynamics.

This module implements full rigid-body dynamics for a quadrotor based on Newton-Euler equations. The
dynamics are parameterised with physical constants (mass, inertia, thrust and torque curves, motor
arm length, drag coefficients) and require no data fitting. Propeller gyroscopic effects are
included.

The command interface is four motor angular velocities in RPM.

Both a numeric implementation ([dynamics][crazyflow.dynamics.first_principles.dynamics]) and a
symbolic CasADi implementation
([symbolic_dynamics][crazyflow.dynamics.first_principles.symbolic_dynamics]) are provided.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import casadi as cs
import jax
import jax.numpy as jnp
from array_api_compat import array_namespace, device
from flax.struct import dataclass
from scipy.spatial.transform import Rotation as R

import crazyflow.dynamics.symbols as symbols
from crazyflow.dynamics.core import load_params, supports
from crazyflow.dynamics.utils import rotation, to_xp

if TYPE_CHECKING:
    from jax import Device

    from crazyflow._typing import Array  # To be changed to array_api_typing later
    from crazyflow.sim.data import SimData


@supports(rotor_dynamics=True)
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
    L: float,
    prop_inertia: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    rpm2thrust: Array,
    rpm2torque: Array,
    mixing_matrix: Array,
    drag_matrix: Array,
    rotor_dyn_coef: Array,
) -> tuple[Array, Array, Array, Array, Array | None]:
    r"""First principles dynamics for a quatrotor.

    The command is four motor angular velocities in RPM. Forces and torques are
    computed internally using quadratic thrust and torque curves, the mixing matrix,
    and the motor arm length.

    Based on the quaternion dynamics from <https://www.dynsyslab.org/wp-content/papercite-data/pdf/mckinnon-robot20.pdf>

    Args:
        pos: Position of the drone (m).
        quat: Quaternion of the drone (xyzw).
        vel: Velocity of the drone (m/s).
        ang_vel: Angular velocity of the drone (rad/s).
        cmd: Motor speeds (RPMs).
        rotor_vel: Angular velocity of the 4 motors (RPMs). If None, the commanded thrust is
            directly applied. If value is given, thrust dynamics are calculated.
        dist_f: Disturbance force (N) in the world frame acting on the CoM.
        dist_t: Disturbance torque (Nm) in the world frame acting on the CoM.

        mass: Mass of the drone (kg).
        L: Distance from the CoM to the motor (m).
        prop_inertia: Inertia of one propeller in z direction (kg m^2).
        gravity_vec: Gravity vector (m/s^2). We assume the gravity vector points downwards, e.g.
            [0, 0, -9.81].
        J: Inertia matrix (kg m^2).
        J_inv: Inverse inertia matrix (1/kg m^2).
        rpm2thrust: Propeller force constant (N min^2).
        rpm2torque: Propeller torque constant (Nm min^2).
        mixing_matrix: Mixing matrix denoting the turn direction of the motors (4x3).
        drag_matrix: Drag matrix containing the linear drag coefficients (3x3).
        rotor_dyn_coef: Rotor dynamics coefficients.


    Warning:
        Do not use quat_dot directly for integration! Only usage of ang_vel is mathematically
        correct. If you still decide to use quat_dot to integrate, ensure unit length!
        More information: <https://ahrs.readthedocs.io/en/latest/filters/angular.html>
    """
    xp = array_namespace(pos)
    mass, L, prop_inertia, gravity_vec, rpm2thrust, rpm2torque = to_xp(
        mass, L, prop_inertia, gravity_vec, rpm2thrust, rpm2torque, xp=xp, device=device(pos)
    )
    J, J_inv, mixing_matrix, rotor_dyn_coef, drag_matrix = to_xp(
        J, J_inv, mixing_matrix, rotor_dyn_coef, drag_matrix, xp=xp, device=device(pos)
    )
    rot = R.from_quat(quat)  # from body to world
    rot_mat = rot.inv().as_matrix()  # from world to body
    # Rotor dynamics
    if rotor_vel is None:
        warnings.warn("Rotor velocity not provided, using commanded rotor velocity.")
        rotor_vel, rotor_vel_dot = cmd, None
    else:
        rotor_vel_dot = xp.where(
            cmd > rotor_vel,
            rotor_dyn_coef[0] * (cmd - rotor_vel) + rotor_dyn_coef[1] * (cmd**2 - rotor_vel**2),
            rotor_dyn_coef[2] * (cmd - rotor_vel) + rotor_dyn_coef[3] * (cmd**2 - rotor_vel**2),
        )
    # Creating force and torque vector
    forces_motor = rpm2thrust[0] + rpm2thrust[1] * rotor_vel + rpm2thrust[2] * rotor_vel**2
    forces_motor_tot = xp.sum(forces_motor, axis=-1)
    zeros = xp.zeros_like(forces_motor_tot)
    forces_motor_vec = xp.stack((zeros, zeros, forces_motor_tot), axis=-1)
    forces_motor_vec_world = rot.apply(forces_motor_vec)
    force_gravity = gravity_vec * mass
    force_drag = (rot_mat.mT @ (drag_matrix @ (rot_mat @ vel[..., None])))[..., 0]

    torques_motor = rpm2torque[0] + rpm2torque[1] * rotor_vel + rpm2torque[2] * rotor_vel**2
    torque_thrust = (mixing_matrix @ (forces_motor)[..., None])[..., 0] * xp.stack(
        [L, L, xp.asarray(0.0)]
    )
    torque_drag = (mixing_matrix @ (torques_motor)[..., None])[..., 0] * xp.stack(
        [xp.asarray(0.0), xp.asarray(0.0), xp.asarray(1.0)]
    )
    # convert rotor speed from RPM to rad/s for physical calculations
    rpm_to_rad = 2 * xp.pi / 60
    rotor_vel_rads = rotor_vel * rpm_to_rad
    rotor_vel_dot_rads = (
        rotor_vel_dot * rpm_to_rad if rotor_vel_dot is not None else xp.zeros_like(rotor_vel)
    )
    torque_inertia = prop_inertia * xp.stack(
        [
            -ang_vel[..., 1] * xp.sum(mixing_matrix[..., -1, :] * rotor_vel_rads, axis=-1),
            -ang_vel[..., 0] * xp.sum(mixing_matrix[..., -1, :] * rotor_vel_rads, axis=-1),
            xp.sum(mixing_matrix[..., -1, :] * rotor_vel_dot_rads, axis=-1),
        ],
        axis=-1,
    )
    torque_vec = torque_thrust + torque_drag + torque_inertia

    # Linear equation of motion
    forces_sum = forces_motor_vec_world + force_gravity + force_drag
    if dist_f is not None:
        forces_sum = forces_sum + dist_f

    pos_dot = vel
    vel_dot = forces_sum / mass

    # Rotational equation of motion
    if dist_t is not None:
        torque_vec = torque_vec + rot.apply(dist_t, inverse=True)
    quat_dot = rotation.ang_vel2quat_dot(quat, ang_vel)
    torque_vec = torque_vec - xp.linalg.cross(ang_vel, (J @ ang_vel[..., None])[..., 0])
    ang_vel_dot = (J_inv @ torque_vec[..., None])[..., 0]
    return pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot


def symbolic_dynamics(
    model_rotor_vel: bool = True,
    model_dist_f: bool = False,
    model_dist_t: bool = False,
    *,
    mass: float,
    L: float,
    prop_inertia: float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array,
    rpm2thrust: Array,
    rpm2torque: Array,
    mixing_matrix: Array,
    rotor_dyn_coef: Array,
    drag_matrix: Array,
) -> tuple[cs.MX, cs.MX, cs.MX, cs.MX]:
    """Return CasADi symbolic expressions for the first-principles dynamics.

    Implements the same dynamics as [dynamics][crazyflow.dynamics.first_principles.dynamics] using
    CasADi ``MX`` symbolic expressions, validated to be numerically equivalent.

    Args:
        model_rotor_vel: If ``True``, the four motor RPM states are included in ``X`` and rotor
            dynamics are modelled.  If ``False``, the commanded RPMs are used directly. Defaults to
            ``True``.
        model_dist_f: If ``True``, a 3-D force disturbance is appended to ``X``.
        model_dist_t: If ``True``, a 3-D torque disturbance is appended to ``X``.
        mass: Drone mass in kg.
        L: Distance from centre of mass to motor in metres.
        prop_inertia: Moment of inertia of one propeller about its spin axis in kg m².
        gravity_vec: Gravity vector, shape ``(3,)``.
        J: Inertia matrix, shape ``(3, 3)``.
        J_inv: Inverse inertia matrix, shape ``(3, 3)``.
        rpm2thrust: Polynomial coefficients ``[a, b, c]`` for the thrust curve
            ``f = a + b * rpm + c * rpm²``.
        rpm2torque: Polynomial coefficients ``[a, b, c]`` for the drag-torque curve
            ``τ = a + b * rpm + c * rpm²``.
        mixing_matrix: Matrix of shape ``(3, 4)`` mapping per-motor forces to body torques.
        rotor_dyn_coef: Four rotor dynamics coefficients ``[k_acc1, k_acc2, k_dec1, k_dec2]`` used
            in the piecewise-linear spin-up/down model.
        drag_matrix: Diagonal ``(3, 3)`` matrix of linear drag coefficients.

    Returns:
        Tuple ``(X_dot, X, U, Y)`` of CasADi ``MX`` expressions:

        * ``X_dot``: State derivative, length 17 when ``model_rotor_vel=True`` (13 otherwise), plus
          3 per enabled disturbance.
        * ``X``: State vector ``[pos(3), quat(4), vel(3), ang_vel(3)]``, with ``rotor_vel(4)``
          appended if ``model_rotor_vel=True``.
        * ``U``: Input vector ``[rpm_1, rpm_2, rpm_3, rpm_4]``.
        * ``Y``: Output ``[pos(3), quat(4)]``.
    """
    # States and Inputs
    X = cs.vertcat(symbols.pos, symbols.quat, symbols.vel, symbols.ang_vel)
    if model_rotor_vel:
        X = cs.vertcat(X, symbols.rotor_vel)
    if model_dist_f:
        X = cs.vertcat(X, symbols.dist_f)
    if model_dist_t:
        X = cs.vertcat(X, symbols.dist_t)
    U = symbols.cmd_rotor_vel

    # Defining the dynamics function
    if model_rotor_vel:
        # Rotor dynamics
        rotor_vel_dot = cs.if_else(
            U > symbols.rotor_vel,
            rotor_dyn_coef[0] * (U - symbols.rotor_vel)
            + rotor_dyn_coef[1] * (U**2 - symbols.rotor_vel**2),
            rotor_dyn_coef[2] * (U - symbols.rotor_vel)
            + rotor_dyn_coef[3] * (U**2 - symbols.rotor_vel**2),
        )
    else:
        _saved_rotor_vel = symbols.rotor_vel
        symbols.rotor_vel = U
    # Creating force and torque vector
    forces_motor = (
        rpm2thrust[0] + rpm2thrust[1] * symbols.rotor_vel + rpm2thrust[2] * symbols.rotor_vel**2
    )
    forces_motor_vec = cs.vertcat(0.0, 0.0, cs.sum1(forces_motor))
    forces_motor_vec_world = symbols.rot @ forces_motor_vec
    force_gravity = gravity_vec * mass
    force_drag = symbols.rot @ (drag_matrix @ (symbols.rot.T @ symbols.vel))

    torques_motor = (
        rpm2torque[0] + rpm2torque[1] * symbols.rotor_vel + rpm2torque[2] * symbols.rotor_vel**2
    )
    torques_thrust = mixing_matrix @ forces_motor * cs.vertcat(L, L, 0.0)
    torques_drag = mixing_matrix @ torques_motor * cs.vertcat(0.0, 0.0, 1.0)
    # convert rotor speed from RPM to rad/s for physical calculations
    rpm_to_rad = 2 * cs.pi / 60
    rotor_vel_rads = symbols.rotor_vel * rpm_to_rad
    rotor_vel_dot_rads = rotor_vel_dot * rpm_to_rad if model_rotor_vel else symbols.rotor_vel * 0.0
    torque_inertia = prop_inertia * cs.vertcat(
        -symbols.ang_vel[1] * cs.sum(mixing_matrix[-1, :] * rotor_vel_rads),
        -symbols.ang_vel[0] * cs.sum(mixing_matrix[-1, :] * rotor_vel_rads),
        cs.sum(mixing_matrix[-1, :] * rotor_vel_dot_rads),
    )
    torques_motor_vec = torques_thrust + torques_drag + torque_inertia

    # Linear equation of motion
    forces_sum = forces_motor_vec_world + force_gravity + force_drag
    if model_dist_f:
        forces_sum = forces_sum + symbols.dist_f

    pos_dot = symbols.vel
    vel_dot = forces_sum / mass

    # Rotational equation of motion
    xi = cs.vertcat(
        cs.horzcat(0, -symbols.ang_vel.T), cs.horzcat(symbols.ang_vel, -cs.skew(symbols.ang_vel))
    )
    quat_dot = 0.5 * (xi @ symbols.quat)
    torques_sum = torques_motor_vec
    if model_dist_t:
        torques_sum = torques_sum + symbols.rot.T @ symbols.dist_t
    ang_vel_dot = J_inv @ (torques_sum - cs.cross(symbols.ang_vel, J @ symbols.ang_vel))

    if model_rotor_vel:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot)
    else:
        X_dot = cs.vertcat(pos_dot, quat_dot, vel_dot, ang_vel_dot)
    Y = cs.vertcat(symbols.pos, symbols.quat)

    if not model_rotor_vel:
        symbols.rotor_vel = _saved_rotor_vel
    return X_dot, X, U, Y


@dataclass
class Params:
    mass: Array  # (N, M, 1)
    """Mass of the drone."""
    L: Array  # (N, M, 1)
    """Arm length of the drone."""
    prop_inertia: Array  # (N, M, 1)
    """Inertia of the propeller."""
    gravity_vec: Array  # (N, M, 3)
    """Gravity vector of the drone."""
    J: Array  # (N, M, 3, 3)
    """Inertia matrix of the drone."""
    J_inv: Array  # (N, M, 3, 3)
    """Inverse of the inertia matrix of the drone."""
    rpm2thrust: Array  # (N, M, 1)
    """Force constant of the drone."""
    rpm2torque: Array  # (N, M, 1)
    """Torque constant of the drone."""
    mixing_matrix: Array  # (N, M, 3, 4)
    """Mixing matrix of the drone."""
    drag_matrix: Array  # (N, M, 3, 3)
    """Drag matrix of the drone."""
    rotor_dyn_coef: Array  # (N, M, 4)
    """Rotor speed dynamics time constant of the drone."""

    @staticmethod
    def create(n_worlds: int, n_drones: int, drone: str, device: Device) -> Params:
        """Create a default set of parameters for the simulation."""
        p = load_params("first_principles", drone)
        J = jax.device_put(jnp.tile(p["J"][None, None, :, :], (n_worlds, n_drones, 1, 1)), device)
        return Params(
            mass=jnp.full((n_worlds, n_drones, 1), p["mass"], device=device),
            L=jnp.asarray(p["L"], device=device),
            prop_inertia=jnp.asarray(p["prop_inertia"], device=device),
            gravity_vec=jnp.asarray(p["gravity_vec"], device=device),
            J=J,
            J_inv=jnp.linalg.inv(J),
            rpm2thrust=jnp.asarray(p["rpm2thrust"], device=device),
            rpm2torque=jnp.asarray(p["rpm2torque"], device=device),
            mixing_matrix=jnp.asarray(p["mixing_matrix"], device=device),
            drag_matrix=jnp.asarray(p["drag_matrix"], device=device),
            rotor_dyn_coef=jnp.asarray(p["rotor_dyn_coef"], device=device),
        )


def sim_dynamics(data: SimData) -> SimData:
    """Compute the forces and torques from the first principle dynamics."""
    params: Params = data.params
    vel, _, acc, ang_acc, rotor_acc = dynamics(
        pos=data.states.pos,
        quat=data.states.quat,
        vel=data.states.vel,
        ang_vel=data.states.ang_vel,
        cmd=data.controls.rotor_vel,
        rotor_vel=data.states.rotor_vel,
        dist_f=data.states.force,
        dist_t=data.states.torque,
        **params.__dict__,
    )
    states_deriv = data.states_deriv.replace(
        vel=vel, ang_vel=data.states.ang_vel, acc=acc, ang_acc=ang_acc, rotor_acc=rotor_acc
    )
    return data.replace(states_deriv=states_deriv)
