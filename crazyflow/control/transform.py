"""Transformations between physical parameters of the quadrotors.

Conversions such as from motor forces to rotor speeds, or from thrust to PWM, are bundled in this
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_extra as xpx
from array_api_compat import array_namespace

if TYPE_CHECKING:
    from crazyflow._typing import Array  # To be changed to array_api_typing later


def motor_force2rotor_vel(motor_forces: Array, rpm2thrust: Array) -> Array:
    """Convert motor forces to rotor velocities, where f=a*rpm^2+b*rpm+c.

    Args:
        motor_forces: Motor forces in SI units with shape (..., N).
        rpm2thrust: RPM to thrust conversion factors.

    Returns:
        Array of rotor velocities in rad/s with shape (..., N).
    """
    xp = array_namespace(motor_forces)
    return (
        -rpm2thrust[1]
        + xp.sqrt(rpm2thrust[1] ** 2 - 4 * rpm2thrust[2] * (rpm2thrust[0] - motor_forces))
    ) / (2 * rpm2thrust[2])


def rotor_vel2body_force(rotor_vel: Array, rpm2thrust: Array) -> Array:
    """Convert rotor velocities to motor forces."""
    xp = array_namespace(rotor_vel)
    body_force = xp.zeros(rotor_vel.shape[:-1] + (3,), dtype=rotor_vel.dtype)
    body_force = xpx.at(body_force)[..., 2].set(
        xp.sum(
            rpm2thrust[..., 0] + rpm2thrust[..., 1] * rotor_vel + rpm2thrust[..., 2] * rotor_vel**2,
            axis=-1,
        )
    )
    return body_force


def rotor_vel2body_torque(
    rotor_vel: Array, rpm2thrust: Array, rpm2torque: Array, L: float | Array, mixing_matrix: Array
) -> Array:
    """Convert rotor velocities to motor torques."""
    xp = array_namespace(rotor_vel)
    forces = rpm2thrust[..., 0] + rpm2thrust[..., 1] * rotor_vel + rpm2thrust[..., 2] * rotor_vel**2
    torques_xy = (
        xp.stack([xp.zeros_like(forces), xp.zeros_like(forces), forces])
        @ mixing_matrix
        * xp.stack([L, L, 0])
    )
    torques = (
        rpm2torque[..., 0] + rpm2torque[..., 1] * rotor_vel + rpm2torque[..., 2] * rotor_vel**2
    )
    torques_z = xp.stack([xp.zeros_like(torques), xp.zeros_like(torques), torques])
    body_torque = torques_xy + torques_z
    return body_torque


def force2pwm(thrust: Array | float, thrust_max: Array | float, pwm_max: Array | float) -> Array:
    """Convert thrust in N to thrust in PWM.

    Args:
        thrust: Array or float of the thrust in [N]
        thrust_max: Maximum thrust in [N]
        pwm_max: Maximum PWM value

    Returns:
        Thrust converted in PWM.
    """
    return thrust / thrust_max * pwm_max


def pwm2force(
    pwm: Array | float, thrust_max: Array | float, pwm_max: Array | float
) -> Array | float:
    """Convert pwm thrust command to actual thrust.

    Args:
        pwm: Array or float of the pwm value
        thrust_max: Maximum thrust in [N]
        pwm_max: Maximum PWM value

    Returns:
        thrust: Array or float thrust in [N]
    """
    return pwm / pwm_max * thrust_max
