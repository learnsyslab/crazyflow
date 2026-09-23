"""Minimal far-field downwash external-wrench plugin.

This models the downwash of level, hovering, identical Crazyflies using the
far-field jet from Bauersfeld et al. (arXiv:2403.13321) and the thrust-decay
model of Su et al. (arXiv:2207.09645).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim import Sim
from crazyflow.sim.pipeline import insert_fn_before

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

# Physical parameters for the cf21B_500
AIR_DENSITY = 1.225  # kg/m^3
PROPELLER_RADIUS = 27.5e-3  # m
MOTOR_DISTANCE = 0.035355 * 2  # m, distance between opposite motors
N_PROPELLERS = 4
GRAVITY = 9.81

# This must be fitted for the propeller/downwash setup.
THRUST_DECAY_COEFFICIENT = 0.05  # s/m

# Far-field fit in Eq. (9) of the Bauersfeld paper.
BD = 10.11
S = 0.07668
S0 = -5.817


def downwash_fn(data: SimData) -> SimData:
    """Apply downwash-induced thrust loss as a world-frame external wrench.

    The source flow originates at each drone centre, while the field is sampled
    at every target rotor.
    """
    rotation = R.from_quat(data.states.quat)
    rotor_offsets = data.params.L * jnp.array(
        [[1.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    rotor_offsets_world = jnp.swapaxes(rotation.as_matrix() @ rotor_offsets.T, -1, -2)
    rotor_positions = data.states.pos[..., None, :] + rotor_offsets_world

    # Axis 1 indexes the source drone, axis 2 the target, and axis 3 its rotor.
    source_to_target = data.states.pos[:, :, None, None, :] - rotor_positions[:, None, :, :, :]
    s = source_to_target[..., 2]  # Positive only for targets below a source.
    r = jnp.linalg.vector_norm(source_to_target[..., :2], axis=-1)
    s_normalized = s / MOTOR_DISTANCE

    mass = data.params.mass[0]
    u_hover = jnp.sqrt(
        mass * GRAVITY / (2.0 * AIR_DENSITY * jnp.pi * PROPELLER_RADIUS**2 * N_PROPELLERS)
    )
    half_width = S * (s_normalized - S0)
    centerline_velocity = u_hover * BD / (s_normalized - S0)
    xi = (r / MOTOR_DISTANCE) / half_width
    u_downwash = centerline_velocity / (1.0 + (jnp.sqrt(2.0) - 1.0) * xi**2) ** 2
    # u_downwash = jnp.where(s_normalized > 2.5, u_downwash, 0.0)
    u_downwash = jnp.sum(u_downwash, axis=1)  # Sum all sources at each target rotor.

    # Eq. (5) in Su et al.: each motor loses a fraction b_v * U_D of its
    # current thrust.  Clamp this extrapolation so effective thrust is never
    # negative outside the fitted range.
    loss_fraction = jnp.clip(THRUST_DECAY_COEFFICIENT * u_downwash, 0.0, 1.0)
    rotor_vel = data.states.rotor_vel
    k0, k1, k2 = (
        data.params.rpm2thrust[..., 0],
        data.params.rpm2thrust[..., 1],
        data.params.rpm2thrust[..., 2],
    )
    motor_thrust = k0 + k1 * rotor_vel + k2 * rotor_vel**2
    thrust_delta = -loss_fraction * motor_thrust

    # Map the per-motor force changes to a body-frame wrench, as in Eq. (7).
    total_thrust_delta = jnp.sum(thrust_delta, axis=-1)
    zeros = jnp.zeros_like(total_thrust_delta)
    force_body = jnp.stack((zeros, zeros, total_thrust_delta), axis=-1)

    lever = jnp.array([1.0, 1.0, 0.0])
    torque_body = (data.params.mixing_matrix @ (thrust_delta * data.params.L)[..., None])[
        ..., 0
    ] * lever

    # Account for the corresponding reaction-torque change about body z.
    effective_motor_thrust = jnp.maximum(motor_thrust + thrust_delta, 0.0)
    effective_rotor_vel = motor_force2rotor_vel(effective_motor_thrust, data.params.rpm2thrust)
    c0, c1, c2 = (
        data.params.rpm2torque[..., 0],
        data.params.rpm2torque[..., 1],
        data.params.rpm2torque[..., 2],
    )
    motor_torque = c0 + c1 * rotor_vel + c2 * rotor_vel**2
    effective_motor_torque = c0 + c1 * effective_rotor_vel + c2 * effective_rotor_vel**2
    reaction_torque_delta = effective_motor_torque - motor_torque
    torque_body = torque_body + (data.params.mixing_matrix @ reaction_torque_delta[..., None])[
        ..., 0
    ] * jnp.array([0.0, 0.0, 1.0])

    states = data.states.replace(
        force=rotation.apply(force_body), torque=rotation.apply(torque_body)
    )
    return data.replace(states=states)


def main(plot: bool = True) -> None:
    """Hover drone 0 while drone 1 flies straight through its downwash."""
    sim = Sim(n_drones=2, drone="cf21B_500", control="state")

    insert_fn_before(sim.step_pipeline, "integration", downwash_fn)
    sim.build_step_fn()

    upper_pos = np.array([0.0, 0.0, 1.0])
    lower_start = np.array([-0.5, 0.0, 0.5])
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=jnp.array([[upper_pos, lower_start]]))
    )
    sim.build_default_data()

    duration = 6.0
    speed = 1.0 / duration
    command = np.zeros((1, 2, 16))
    command[..., 9:13] = [0.0, 0.0, 0.0, 1.0]  # level quaternion (xyzw)
    command[0, 0, :3] = upper_pos
    z_positions = []
    downwash_force_z = []
    downwash_pitch_torque = []

    for step in range(int(duration * sim.control_freq)):
        t = step / sim.control_freq
        command[0, 1, :3] = [-0.5 + speed * t, 0.0, 0.5]
        command[0, 1, 3:6] = [speed, 0.0, 0.0]
        sim.state_control(command)
        sim.step(sim.freq // sim.control_freq)
        z_positions.append(np.asarray(sim.data.states.pos[0, :, 2]))
        downwash_force_z.append(np.asarray(sim.data.states.force[0, 1, 2]))
        downwash_pitch_torque.append(np.asarray(sim.data.states.torque[0, 1, 1]))
        sim.render()

    sim.close()
    if plot:
        import matplotlib.pyplot as plt

        t = np.arange(len(z_positions)) / sim.control_freq
        z_positions = np.asarray(z_positions)
        fig, axes = plt.subplots(3, 1, sharex=True)
        axes[0].plot(t, z_positions[:, 0], label="upper drone")
        axes[0].plot(t, z_positions[:, 1], label="lower drone")
        axes[0].set_ylabel("z position (m)")
        axes[0].legend()

        axes[1].plot(t, downwash_force_z, label="lower drone")
        axes[1].set_ylabel("downwash force z (N)")
        axes[1].legend()

        axes[2].plot(t, downwash_pitch_torque, label="lower drone")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("downwash pitch torque y (Nm)")
        axes[2].legend()
        plt.show()


if __name__ == "__main__":
    main()
