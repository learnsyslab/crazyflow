"""Minimal far-field downwash thrust-loss plugin.

This models the downwash of level, hovering, identical Crazyflies using the
far-field jet from Bauersfeld et al. (arXiv:2403.13321). It belongs after
``force_torque_controller`` and before ``clip_rotor_vel_cmd``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.control.mellinger import force_torque2rotor_vel
from crazyflow.sim.pipeline import insert_fn_before

if TYPE_CHECKING:
    from crazyflow.sim import Sim
    from crazyflow.sim.data import SimData

# Physical parameters for the cf21B_500
AIR_DENSITY = 1.225  # kg/m^3
PROPELLER_RADIUS = 27.5e-3  # m
MOTOR_DISTANCE = 0.035355*2  # m, distance between opposite motors 
N_PROPELLERS = 4
GRAVITY = 9.81

# Far-field fit in Eq. (9) of the paper.
BD = 10.11
S = 0.07668
S0 = -5.817


def _thrust_loss(alpha: jax.Array) -> jax.Array:
    """Return T_effective / T_nominal at constant aerodynamic power.

    Equating the still-air and downwash momentum-theory powers yields the
    positive root of eta**3 + alpha * eta - 1 = 0, with alpha = U_D / U_H.
    """
    lower = jnp.zeros_like(alpha)
    upper = jnp.ones_like(alpha)

    def bisect(_: int, bounds: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        lower, upper = bounds
        eta = (lower + upper) / 2.0
        above_root = eta**3 + alpha * eta > 1.0
        return jnp.where(above_root, lower, eta), jnp.where(above_root, eta, upper)

    lower, upper = jax.lax.fori_loop(0, 24, bisect, (lower, upper))
    return (lower + upper) / 2.0


def downwash_fn(data: SimData) -> SimData:
    """Reduce every lower drone's nominal rotor force by its downwash loss."""
    # Axis 1 indexes the source drone; axis 2 indexes the target drone.
    source_to_target = data.states.pos[:, :, None, :] - data.states.pos[:, None, :, :]
    s = source_to_target[..., 2]  # Positive only for targets below a source.
    r = jnp.linalg.vector_norm(source_to_target[..., :2], axis=-1)
    s_normalized = s / MOTOR_DISTANCE

    mass = data.params.mass[0] 
    u_hover = jnp.sqrt(
        mass * GRAVITY
        / (2.0 * AIR_DENSITY * jnp.pi * PROPELLER_RADIUS**2 * N_PROPELLERS)
    )
    half_width = S * (s_normalized - S0)
    centerline_velocity = u_hover * BD / (s_normalized - S0)
    xi = (r / MOTOR_DISTANCE) / half_width
    u_downwash = centerline_velocity / (1.0 + (jnp.sqrt(2.0) - 1.0) * xi**2) ** 2
    u_downwash = jnp.where(s_normalized > 2.5, u_downwash, 0.0)
    u_downwash = jnp.sum(u_downwash, axis=1)  # Sum all sources at each target.

    eta = _thrust_loss(u_downwash / u_hover)
    force_torque = data.controls.force_torque

    # Scale only thrust
    effective_thrust = force_torque.cmd[..., 0] * eta

    rotor_vel = force_torque2rotor_vel(
        effective_thrust[..., None], force_torque.cmd[..., 1:], **force_torque.params
    )
    return data.replace(controls=data.controls.replace(rotor_vel=rotor_vel))


def install_downwash(sim: Sim) -> None:
    """Add downwash after allocation and rebuild the compiled step function."""
    insert_fn_before(sim.step_pipeline, "clip_rotor_vel_cmd", downwash_fn)
    sim.build_step_fn()


def main(plot: bool = True) -> None:
    """Hover drone 0 while drone 1 flies straight through its downwash."""
    from crazyflow.sim import Sim

    sim = Sim(n_drones=2, drone="cf21B_500", control="state")
    install_downwash(sim)

    upper_pos = np.array([0.0, 0.0, 1.2])
    lower_start = np.array([-0.5, 0.0, 1.0])
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=jnp.array([[upper_pos, lower_start]]))
    )
    sim.build_default_data()

    duration = 3.0
    speed = 1.0 / duration
    command = np.zeros((1, 2, 16))
    command[..., 9:13] = [0.0, 0.0, 0.0, 1.0]  # level quaternion (xyzw)
    command[0, 0, :3] = upper_pos
    z_positions = []

    for step in range(int(duration * sim.control_freq)):
        t = step / sim.control_freq
        command[0, 1, :3] = [-0.5 + speed * t, 0.0, 1.0]
        command[0, 1, 3:6] = [speed, 0.0, 0.0]
        sim.state_control(command)
        sim.step(sim.freq // sim.control_freq)
        z_positions.append(np.asarray(sim.data.states.pos[0, :, 2]))
        sim.render()

    sim.close()
    if plot:
        import matplotlib.pyplot as plt

        t = np.arange(len(z_positions)) / sim.control_freq
        z_positions = np.asarray(z_positions)
        plt.plot(t, z_positions[:, 0], label="upper drone")
        plt.plot(t, z_positions[:, 1], label="lower drone")
        plt.xlabel("Time (s)")
        plt.ylabel("z position (m)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
    
