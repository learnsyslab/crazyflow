"""Minimal far-field downwash external-wrench plugin.

This models the downwash of identical Crazyflies using the far-field jet from
[1] and the thrust-decay model of [2]. 

[1] Bauersfeld et al. https://arxiv.org/abs/2403.13321
[2] Su et al. https://arxiv.org/abs/2207.09645
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.sim import Sim
from crazyflow.sim.pipeline import insert_fn_before

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

# Physical parameters for the cf21B_500
AIR_DENSITY = 1.225  # kg/m^3
PROPELLER_RADIUS = 27.5e-3  # m
MOTOR_DISTANCE = 0.1  # m, distance between opposite motors

# This must be fitted for the propeller/downwash setup.
THRUST_DECAY_COEFFICIENT = 0.07  # s/m

# Far-field fit in Eq. (9) of [1]
BD = 10.11
S = 0.07668
S0 = -5.817


def downwash_fn(data: SimData) -> SimData:
    """Apply downwash-induced thrust loss as a world-frame external wrench.

    The source flow originates at each drone centre, while the field is sampled
    at every target rotor in the source's body frame.
    """
    rotation = R.from_quat(data.states.quat)
    mixing_matrix = data.params.mixing_matrix

    offsets = data.params.L * jnp.stack(
        [-mixing_matrix[1], mixing_matrix[0], jnp.zeros_like(mixing_matrix[0])], 
        axis=0
    )
    rotor_offsets_body = offsets.T
    rotor_offsets_world = jnp.swapaxes(rotation.as_matrix() @ rotor_offsets_body.T, -1, -2)
    rotor_positions = data.states.pos[..., None, :] + rotor_offsets_world

    # Axis 1 indexes the source drone, axis 2 the target, and axis 3 its rotor.
    source_to_target = data.states.pos[:, :, None, None, :] - rotor_positions[:, None, :, :, :]
    # Rotate the source-minus-target displacement into each source's frame.
    world_to_body = rotation.as_matrix().mT
    # Broadcast each source rotation across all target drones and rotors.
    source_to_target_body = (
        world_to_body[:, :, None, None, :, :] @ source_to_target[..., None]
    )[..., 0]
    s = source_to_target_body[..., 2]
    r = jnp.linalg.vector_norm(source_to_target_body[..., :2], axis=-1)

    # Normalization according to [1] Eq. (8)
    s_normalized = s / MOTOR_DISTANCE
    r_normalized = r / MOTOR_DISTANCE

    mass = data.params.mass[0]
    gravity = -data.params.gravity_vec[2]
    n_propellers = mixing_matrix.shape[-1]

    u_hover = jnp.sqrt(
        mass * gravity / (2.0 * AIR_DENSITY * jnp.pi * PROPELLER_RADIUS**2 * n_propellers)
    )  # [1] Eq. (1)

    # Keep the fit finite upstream, where its contribution is masked below.
    axial_distance = jnp.maximum(s_normalized - S0, 1e-6)
    half_width = S * axial_distance  # [1] Eq. (6)

    centerline_velocity = u_hover * BD / axial_distance  # [1] Eq. (2)

    xi = (r_normalized) / half_width  # [1] Eq. (4)

    u_downwash = centerline_velocity / (1.0 + (jnp.sqrt(2.0) - 1.0) * xi**2) ** 2  # [1] Eq. (3)

    # This prevents "negative" downwash
    u_downwash = jnp.where(s_normalized > 0.1, u_downwash, 0.0)
    u_downwash = jnp.sum(u_downwash, axis=1)  # Sum all sources at each target rotor.

    # [2] Eq. (5): each motor loses a fraction b_v * U_D of its current thrust
    loss_fraction = THRUST_DECAY_COEFFICIENT * u_downwash
    rotor_vel = data.states.rotor_vel
    k0, k1, k2 = (
        data.params.rpm2thrust[..., 0],
        data.params.rpm2thrust[..., 1],
        data.params.rpm2thrust[..., 2],
    )
    motor_thrust = k0 + k1 * rotor_vel + k2 * rotor_vel**2
    thrust_delta = -loss_fraction * motor_thrust

    # Map the per-motor force changes to a body-frame wrench, as in [2] Eq. (7).
    total_thrust_delta = jnp.sum(thrust_delta, axis=-1)
    zeros = jnp.zeros_like(total_thrust_delta)
    force_body = jnp.stack((zeros, zeros, total_thrust_delta), axis=-1)

    lever = jnp.array([1.0, 1.0, 0.0])
    torque_body = (mixing_matrix @ (thrust_delta * data.params.L)[..., None])[
        ..., 0
    ] * lever

    states = data.states.replace(
        force=rotation.apply(force_body), torque=rotation.apply(torque_body)
    )
    return data.replace(states=states)


def plot_hover_velocity_field(source_positions: np.ndarray, data: SimData) -> None:
    """Plot the far-field downwash-speed magnitude in the y=0 plane."""
    import matplotlib.pyplot as plt

    x = np.linspace(-0.6, 0.6, 300)
    z = np.linspace(0.0, 1.15, 300)
    X, Z = np.meshgrid(x, z)

    # Every grid point lies in the y=0 plane.
    points = np.stack((X, np.zeros_like(X), Z), axis=-1)
    u_downwash = np.zeros_like(X)

    gravity = -data.params.gravity_vec[2]
    n_propellers = data.params.mixing_matrix.shape[-1]
    mass = data.params.mass[0]

    u_hover = np.sqrt(
        mass * gravity / (2.0 * AIR_DENSITY * np.pi * PROPELLER_RADIUS**2 * n_propellers)
    )

    for source_pos in source_positions:
        source_to_point = source_pos - points
        s = source_to_point[..., 2]
        r = np.linalg.vector_norm(source_to_point[..., :2], axis=-1)
        s_normalized = s / MOTOR_DISTANCE

        q = np.maximum(s_normalized - S0, 1e-6)
        half_width = S * q
        centerline_velocity = u_hover * BD / q
        xi = (r / MOTOR_DISTANCE) / half_width

        velocity = centerline_velocity / (1.0 + (np.sqrt(2.0) - 1.0) * xi**2) ** 2
        u_downwash += velocity

    fig, ax = plt.subplots()
    image = ax.pcolormesh(X, Z, u_downwash, shading="auto", cmap="viridis")
    ax.scatter(source_positions[:, 0], source_positions[:, 2], color="red", label="source drone")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Hovering-drone downwash speed")
    ax.legend()
    fig.colorbar(image, ax=ax, label="downward airspeed $U_D$ (m/s)")
    plt.show()


def main(plot: bool = True) -> None:
    """Hover drone 0 while drone 1 makes two downwash passes at different heights."""
    sim = Sim(n_drones=2, drone="cf21B_500", control="state")

    insert_fn_before(sim.step_pipeline, "integration", downwash_fn)
    sim.build_step_fn()

    upper_pos = np.array([0.0, 0.0, 1.2])
    outbound_height = 0.5
    return_height = 0.95
    lower_start = np.array([-0.5, 0.0, outbound_height])

    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=jnp.array([[upper_pos, lower_start]]))
    )
    sim.build_default_data()

    command = np.zeros((1, 2, 16))
    command[..., 9:13] = R.from_euler("z", 0).as_quat()
    command[0, 0, :3] = upper_pos

    waypoints = np.concatenate(
        (
            np.linspace(
                lower_start, [0.5, 0.0, outbound_height], 3 * sim.control_freq, endpoint=False
            ),
            np.linspace(
                [0.5, 0.0, outbound_height],
                [0.5, 0.0, return_height],
                sim.control_freq,
                endpoint=False,
            ),
            np.linspace(
                [0.5, 0.0, return_height], [-0.5, 0.0, return_height], 3 * sim.control_freq
            ),
        )
    )

    z_positions = []
    downwash_force_z = []
    downwash_pitch_torque = []

    for position in waypoints:
        command[0, 1, :3] = position

        sim.state_control(command)
        sim.step(sim.freq // sim.control_freq)

        z_positions.append(np.asarray(sim.data.states.pos[0, :, 2]))
        downwash_force_z.append(np.asarray(sim.data.states.force[0, 1, 2]))
        downwash_pitch_torque.append(np.asarray(sim.data.states.torque[0, 1, 1]))
        sim.render()

    sim.close()

    if plot:
        import matplotlib.pyplot as plt

        t = np.arange(len(waypoints)) / sim.control_freq
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
        axes[2].set_xlabel("time (s)")
        axes[2].set_ylabel("downwash pitch torque y (Nm)")
        axes[2].legend()

        for axis in axes:
            axis.axvline(3.0, color="black", linestyle="--", alpha=0.5)
            axis.axvline(4.0, color="black", linestyle="--", alpha=0.5)

        plot_hover_velocity_field(np.asarray([upper_pos]), sim.data)
        plt.show()


if __name__ == "__main__":
    main()
