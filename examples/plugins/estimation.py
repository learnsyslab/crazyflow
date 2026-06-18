"""State estimation example using UWB position measurements.

WARNING: This is an advanced example meant for advanced Crazyflow users.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow import Sim
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim.pipeline import insert_fn_after, prepend_fn
from crazyflow.sim.visualize import draw_line, draw_points

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

UWB_BASE_STATIONS = jnp.array(
    [
        [-2.5, -2.5, 0.0],
        [-2.5, 2.5, 0.0],
        [2.5, -2.5, 0.0],
        [2.5, 2.5, 0.0],
        [-2.5, -2.5, 3.0],
        [-2.5, 2.5, 3.0],
        [2.5, -2.5, 3.0],
        [2.5, 2.5, 3.0],
    ]
)
RANGE_STD = 0.03  # measurement std dev [m]
RANGE_BIAS_MAX = 0.08  # measurement bias, uniformly distributed in [0, RANGE_BIAS_MAX] [m]


def trajectory(t: float, t_total: float = 20.0) -> np.ndarray:
    """Return a slow figure-eight state command."""
    center = np.array([0.0, 0.0, 1.0])
    size = np.array([1.0, 0.75, 0.0])
    omega = 2 * np.pi / t_total
    cmd = np.zeros((1, 1, 13))
    pos = center + size * np.array([np.sin(omega * t), np.sin(2 * omega * t), 0.0])
    vel = size * omega * np.array([np.cos(omega * t), 2 * np.cos(2 * omega * t), 0.0])
    acc = size * omega**2 * np.array([-np.sin(omega * t), -4 * np.sin(2 * omega * t), 0.0])
    cmd[..., 0:3] = pos
    cmd[..., 3:6] = vel
    cmd[..., 6:9] = acc
    return cmd


def simulate_uwb(data: SimData) -> SimData:
    """Generate one range measurement for each UWB base station."""
    key, noise_key = jax.random.split(data.core.rng_key)
    ranges = jnp.linalg.norm(data.states.pos[..., None, :] - UWB_BASE_STATIONS, axis=-1)
    ranges += data.plugins["uwb_bias"]
    ranges += data.plugins["range_std"] * jax.random.normal(noise_key, ranges.shape)
    plugins = data.plugins | {"uwb_ranges": jnp.maximum(ranges, 0.0)}
    return data.replace(plugins=plugins, core=data.core.replace(rng_key=key))


def estimate_state(data: SimData) -> SimData:
    """Run a constant-velocity EKF update using the UWB ranges."""
    dt = 1.0 / data.core.freq

    # Predict: x_prior = F x, P_prior = F P F^T + Q
    state = data.plugins["estimate"]
    covariance = data.plugins["covariance"]
    # Constant velocity: p_next = p + v * dt and v_next = v.
    transition = jnp.eye(6).at[:3, 3:].set(jnp.eye(3) * dt)
    # Model unknown acceleration a as process noise:
    # x_next = F x + acceleration_to_state * a, where delta_p = 0.5*a*dt^2 and delta_v = a*dt.
    acceleration_to_state = jnp.concat((jnp.eye(3) * (0.5 * dt**2), jnp.eye(3) * dt), axis=0)
    process_covariance = 30.0**2 * acceleration_to_state @ acceleration_to_state.T
    # Standard deviation of 30 of the unknown acceleration driving the constant-velocity process.
    predicted_state = state @ transition.T
    predicted_covariance = transition @ covariance @ transition.T + process_covariance

    # Linearize ranges: z_prior = h(x_prior), H = dh/dx at x_prior
    anchor_offsets = predicted_state[..., None, :3] - UWB_BASE_STATIONS
    predicted_ranges = jnp.linalg.norm(anchor_offsets, axis=-1)
    range_directions = anchor_offsets / jnp.maximum(predicted_ranges[..., None], 1e-6)
    n_ranges = UWB_BASE_STATIONS.shape[0]
    measurement_jacobian = jnp.zeros((*predicted_state.shape[:-1], n_ranges, 6))
    measurement_jacobian = measurement_jacobian.at[..., :, :3].set(range_directions)
    measurement_covariance = jnp.eye(n_ranges) * data.plugins["range_std"] ** 2
    measurement_jacobian_transpose = jnp.swapaxes(measurement_jacobian, -1, -2)

    # Innovation and gain: y = z - z_prior, S = H P_prior H^T + R, K = P_prior H^T S^-1
    innovation = data.plugins["uwb_ranges"] - predicted_ranges
    innovation_covariance = (
        measurement_jacobian @ predicted_covariance @ measurement_jacobian_transpose
        + measurement_covariance
    )
    covariance_times_jacobian = predicted_covariance @ measurement_jacobian_transpose
    # K = P_prior H^T S^-1. Solve S K^T = (P_prior H^T)^T instead of forming S^-1;
    # a linear solve is more numerically stable and directly computes the same result.
    kalman_gain = jnp.swapaxes(
        jnp.linalg.solve(innovation_covariance, jnp.swapaxes(covariance_times_jacobian, -1, -2)),
        -1,
        -2,
    )

    # Correct: x = x_prior + K y
    corrected_state = predicted_state + (kalman_gain @ innovation[..., None])[..., 0]

    # Joseph form: P = (I - K H) P_prior (I - K H)^T + K R K^T
    identity = jnp.eye(6)
    residual_map = identity - kalman_gain @ measurement_jacobian
    corrected_covariance = residual_map @ predicted_covariance @ jnp.swapaxes(
        residual_map, -1, -2
    ) + kalman_gain @ measurement_covariance @ jnp.swapaxes(kalman_gain, -1, -2)
    corrected_covariance = 0.5 * (corrected_covariance + jnp.swapaxes(corrected_covariance, -1, -2))

    plugins = data.plugins | {"estimate": corrected_state, "covariance": corrected_covariance}
    return data.replace(plugins=plugins)


def use_estimate_for_control(data: SimData) -> SimData:
    """Save the physical state and expose the estimate to the controllers."""
    estimate = data.plugins["estimate"]
    plugins = data.plugins | {"ground_truth_state": data.states}
    states = data.states.replace(pos=estimate[..., :3], vel=estimate[..., 3:])
    return data.replace(states=states, plugins=plugins)


def restore_ground_truth(data: SimData) -> SimData:
    """Restore the physical state before evaluating and integrating the dynamics."""
    return data.replace(states=data.plugins["ground_truth_state"])


def main(noisy: bool = False, render: bool = True) -> None:
    """Run one perfect or noisy UWB estimation example."""
    name = "biased and noisy UWB measurements" if noisy else "perfect UWB measurements"
    print(f"Running with {name}")

    sim = Sim(control="state", integrator="rk4", rng_key=42)
    hover_force = sim.data.params.mass * -sim.data.params.gravity_vec[2] / 4
    motor_forces = jnp.broadcast_to(hover_force, sim.data.states.rotor_vel.shape)
    hover_rotor_vel = motor_force2rotor_vel(motor_forces, sim.data.params.rpm2thrust)
    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=trajectory(0.0)[..., :3], vel=trajectory(0.0)[..., 3:6], rotor_vel=hover_rotor_vel
        )
    )

    key, bias_key = jax.random.split(sim.data.core.rng_key)
    bias_shape = (sim.n_worlds, sim.n_drones, len(UWB_BASE_STATIONS))
    bias = (
        jax.random.uniform(bias_key, bias_shape, minval=0.0, maxval=RANGE_BIAS_MAX)
        if noisy
        else jnp.zeros(bias_shape)
    )
    range_std = RANGE_STD if noisy else 1e-6  # Nonzero for numerical stability
    estimate = jnp.concat((sim.data.states.pos, sim.data.states.vel), axis=-1)
    covariance = jnp.broadcast_to(jnp.eye(6) * 0.1, (sim.n_worlds, sim.n_drones, 6, 6))
    plugins = {
        "uwb_bias": bias,
        "uwb_ranges": jnp.zeros_like(bias),
        "range_std": jnp.asarray(range_std),
        "estimate": estimate,
        "covariance": covariance,
        "ground_truth_state": sim.data.states,
    }
    sim.data = sim.data.replace(
        plugins=sim.data.plugins | plugins, core=sim.data.core.replace(rng_key=key)
    )

    prepend_fn(sim.step_pipeline, simulate_uwb)
    insert_fn_after(sim.step_pipeline, "simulate_uwb", estimate_state)
    insert_fn_after(sim.step_pipeline, "estimate_state", use_estimate_for_control)
    insert_fn_after(sim.step_pipeline, "force_torque_controller", restore_ground_truth)
    sim.build_default_data()
    sim.build_step_fn()

    duration = 20.0
    reference_times = np.linspace(0.0, duration, 200)
    reference = np.array([trajectory(t, t_total=duration)[0, 0, :3] for t in reference_times])
    estimation_errors = []
    tracking_errors = []
    fps = 60

    for i in range(int(duration * sim.control_freq)):
        t = i / sim.control_freq
        cmd = trajectory(t, t_total=duration)
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)

        truth = np.asarray(sim.data.states.pos[0, 0])
        estimate = np.asarray(sim.data.plugins["estimate"][0, 0, :3])
        estimation_errors.append(np.linalg.norm(estimate - truth))
        tracking_errors.append(np.linalg.norm(truth - cmd[0, 0, :3]))

        if render and ((i * fps) % sim.control_freq) < fps:
            truth = np.asarray(sim.data.states.pos[0, 0])
            estimate = np.asarray(sim.data.plugins["estimate"][0, 0, :3])

            draw_line(
                sim, reference, rgba=np.array([0.4, 0.4, 0.4, 0.5]), start_size=0.5, end_size=0.5
            )
            for anchor in UWB_BASE_STATIONS:
                draw_line(
                    sim,
                    np.stack((truth, anchor)),
                    rgba=np.array([0.2, 0.5, 1.0, 0.25]),
                    start_size=0.3,
                    end_size=0.3,
                )
            draw_points(sim, UWB_BASE_STATIONS, rgba=np.array([0.1, 0.4, 1.0, 1.0]), size=0.05)
            draw_points(sim, cmd[0, 0, :3], rgba=np.array([0.1, 0.8, 0.2, 1.0]), size=0.04)
            draw_points(sim, estimate, rgba=np.array([1.0, 0.2, 0.1, 1.0]), size=0.04)
            sim.render(
                cam_config={
                    "distance": 3.0,
                    "elevation": -45.0,
                    "azimuth": 90.0,
                    "lookat": [0.0, 0.0, 1.0],
                }
            )

    sim.close()
    estimation_rms = np.sqrt(np.mean(np.asarray(estimation_errors) ** 2))
    tracking_rms = np.sqrt(np.mean(np.asarray(tracking_errors) ** 2))
    print(f"Tracking RMS: {tracking_rms:.3f} m, estimation RMS: {estimation_rms:.3f} m")


if __name__ == "__main__":
    main(noisy=False)
    main(noisy=True)
