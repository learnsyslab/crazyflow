"""State estimation example using UWB position measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from drone_models.transform import motor_force2rotor_vel

from crazyflow import Sim
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
HOVER_POSITION = jnp.array([0.0, 0.0, 1.0])
TRAJECTORY_DURATION = 20.0
RANGE_STD = 0.03
RANGE_BIAS_MAX = 0.08
# The constant-velocity model has no acceleration input, so it needs enough process noise to turn.
PROCESS_ACCEL_STD = 30.0
MIN_ESTIMATOR_RANGE_STD = 1e-6


def trajectory(t: float) -> np.ndarray:
    """Return a slow figure-eight state command."""
    omega = 2 * np.pi / TRAJECTORY_DURATION
    cmd = np.zeros((1, 1, 13))
    cmd[..., 0] = np.sin(omega * t)
    cmd[..., 1] = 0.75 * np.sin(2 * omega * t)
    cmd[..., 2] = 1.0
    cmd[..., 3] = omega * np.cos(omega * t)
    cmd[..., 4] = 1.5 * omega * np.cos(2 * omega * t)
    cmd[..., 6] = -(omega**2) * np.sin(omega * t)
    cmd[..., 7] = -3 * omega**2 * np.sin(2 * omega * t)
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
    estimate = data.plugins["estimate"]
    covariance = data.plugins["covariance"]

    transition = jnp.eye(6).at[:3, 3:].set(jnp.eye(3) * dt)
    accel_map = jnp.concat((jnp.eye(3) * (0.5 * dt**2), jnp.eye(3) * dt), axis=0)
    process_covariance = PROCESS_ACCEL_STD**2 * accel_map @ accel_map.T

    predicted = estimate @ transition.T
    predicted_covariance = transition @ covariance @ transition.T + process_covariance

    difference = predicted[..., None, :3] - UWB_BASE_STATIONS
    predicted_ranges = jnp.linalg.norm(difference, axis=-1)
    range_directions = difference / jnp.maximum(predicted_ranges[..., None], 1e-6)
    n_ranges = UWB_BASE_STATIONS.shape[0]
    measurement_jacobian = jnp.zeros((*predicted.shape[:-1], n_ranges, 6))
    measurement_jacobian = measurement_jacobian.at[..., :, :3].set(range_directions)
    range_std = jnp.maximum(data.plugins["range_std"], MIN_ESTIMATOR_RANGE_STD)
    measurement_variance = range_std**2
    measurement_covariance = jnp.eye(n_ranges) * measurement_variance

    innovation_covariance = (
        measurement_jacobian @ predicted_covariance @ jnp.swapaxes(measurement_jacobian, -1, -2)
        + measurement_covariance
    )
    covariance_times_jacobian = predicted_covariance @ jnp.swapaxes(measurement_jacobian, -1, -2)
    gain = jnp.swapaxes(
        jnp.linalg.solve(innovation_covariance, jnp.swapaxes(covariance_times_jacobian, -1, -2)),
        -1,
        -2,
    )
    innovation = data.plugins["uwb_ranges"] - predicted_ranges
    estimate = predicted + jnp.einsum("...ij,...j->...i", gain, innovation)

    identity = jnp.eye(6)
    residual_map = identity - gain @ measurement_jacobian
    covariance = residual_map @ predicted_covariance @ jnp.swapaxes(residual_map, -1, -2)
    covariance += gain @ measurement_covariance @ jnp.swapaxes(gain, -1, -2)
    covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))

    return data.replace(plugins=data.plugins | {"estimate": estimate, "covariance": covariance})


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
    sim.max_visual_geom = 1000

    hover_force = sim.data.params.mass * -sim.data.params.gravity_vec[2] / 4
    motor_forces = jnp.broadcast_to(hover_force, sim.data.states.rotor_vel.shape)
    hover_rotor_vel = motor_force2rotor_vel(motor_forces, sim.data.params.rpm2thrust)

    states = sim.data.states.replace(
        pos=trajectory(0.0)[..., :3], vel=trajectory(0.0)[..., 3:6], rotor_vel=hover_rotor_vel
    )
    sim.data = sim.data.replace(states=states)

    key, bias_key = jax.random.split(sim.data.core.rng_key)
    bias_shape = (sim.n_worlds, sim.n_drones, len(UWB_BASE_STATIONS))
    bias = (
        RANGE_BIAS_MAX * jax.random.uniform(bias_key, bias_shape)
        if noisy
        else jnp.zeros(bias_shape)
    )
    range_std = RANGE_STD if noisy else 0.0
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

    controllers = sim.step_pipeline[:-3]
    integration = sim.step_pipeline[-3:]
    sim.step_pipeline = (
        simulate_uwb,
        estimate_state,
        use_estimate_for_control,
        *controllers,
        restore_ground_truth,
        *integration,
    )
    sim.build_default_data()
    sim.build_step_fn()

    duration = TRAJECTORY_DURATION
    reference_times = np.linspace(0.0, duration, 200)
    reference = np.array([trajectory(t)[0, 0, :3] for t in reference_times])
    estimation_errors = []
    tracking_errors = []
    fps = 60

    for i in range(int(duration * sim.control_freq)):
        t = i / sim.control_freq
        cmd = trajectory(t)
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)

        truth = np.asarray(sim.data.states.pos[0, 0])
        estimate = np.asarray(sim.data.plugins["estimate"][0, 0, :3])
        estimation_errors.append(np.linalg.norm(estimate - truth))
        tracking_errors.append(np.linalg.norm(truth - cmd[0, 0, :3]))

        if render and ((i * fps) % sim.control_freq) < fps:
            truth = np.asarray(sim.data.states.pos[0, 0])
            estimate = np.asarray(sim.data.plugins["estimate"][0, 0, :3])
            anchors = np.asarray(UWB_BASE_STATIONS)

            draw_line(
                sim, reference, rgba=np.array([0.4, 0.4, 0.4, 0.5]), start_size=0.5, end_size=0.5
            )
            for anchor in anchors:
                draw_line(
                    sim,
                    np.stack((truth, anchor)),
                    rgba=np.array([0.2, 0.5, 1.0, 0.25]),
                    start_size=0.3,
                    end_size=0.3,
                )
            draw_points(sim, anchors, rgba=np.array([0.1, 0.4, 1.0, 1.0]), size=0.05)
            draw_points(sim, cmd[0, 0, :3], rgba=np.array([0.1, 0.8, 0.2, 1.0]), size=0.04)
            draw_points(sim, estimate, rgba=np.array([1.0, 0.2, 0.1, 1.0]), size=0.04)
            sim.render()

    sim.close()
    estimation_rms = np.sqrt(np.mean(np.asarray(estimation_errors) ** 2))
    tracking_rms = np.sqrt(np.mean(np.asarray(tracking_errors) ** 2))
    print(f"Tracking RMS: {tracking_rms:.3f} m, estimation RMS: {estimation_rms:.3f} m")


if __name__ == "__main__":
    main(noisy=False)
    main(noisy=True)
