"""Example of a simulated UWB tag as a plugin.

The tag measures the ranges to eight anchors with noise and a constant bias per anchor as in [1] and
solves them for a position. The drone flies a figure-eight once with the state controller reading
the true position and once with the UWB position. The plot compares both runs.

[1] Schuck et al. "UWB Meets Crazyflow: Simulating Degraded Feedback at Scale for Aerial Robotics",
    in 1st Workshop on Robot Meets GNSS and Ranging for Seamless Autonomy, ICRA 2026.
    https://arxiv.org/abs/2610.08225
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

os.environ["SCIPY_ARRAY_API"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass, field
from scipy.spatial.transform import Rotation as R

from crazyflow.control.core import controllable
from crazyflow.control.mellinger import control_state2attitude
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim import Sim
from crazyflow.sim.pipeline import append_fn, insert_fn_after, prepend_fn, replace_fn
from crazyflow.utils import CORE_NDIM_KEY, leaf_replace

if TYPE_CHECKING:
    from jax import Array, Device

    from crazyflow.sim.data import SimData

ANCHORS = np.array([(x, y, z) for x in (-2.5, 2.5) for y in (-2.5, 2.5) for z in (0.0, 3.0)])  # m
RANGE_STD = 0.1  # m
RANGE_BIAS_MAX = 0.15  # m, the bias of each anchor is uniform in [0, max]
UWB_FREQ = 100  # Hz
DURATION = 10.0  # s, one loop of the figure-eight


@dataclass
class UWBData:
    """UWB sensing data."""

    base_stations: Array  # (A, 3)
    """UWB base station positions in the world frame."""
    ranges: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, A)
    """Latest measured UWB ranges from each drone to each base station."""
    range_std: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 1)
    """Standard deviation of the range noise in meters."""
    range_bias: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, A)
    """Persistent positive bias added to each UWB range measurement."""
    range_bias_max: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, M, 1)
    """Upper bound of the uniformly sampled range bias."""
    steps: Array = field(metadata={CORE_NDIM_KEY: 1})  # (N, 1)
    """Last simulation steps that UWB ranges were updated."""
    freq: int = field(pytree_node=False)
    """Frequency of UWB communication."""

    @staticmethod
    def create(
        n_worlds: int,
        n_drones: int,
        base_stations: Array,
        range_std: float,
        range_bias_max: float,
        freq: int,
        device: Device,
    ) -> UWBData:
        """Create default UWB sensing data."""
        shape = (n_worlds, n_drones, len(base_stations))
        return UWBData(
            base_stations=jnp.asarray(base_stations, device=device),
            ranges=jnp.zeros(shape, device=device),
            range_std=jnp.full((n_worlds, n_drones, 1), range_std, device=device),
            range_bias=jnp.zeros(shape, device=device),
            range_bias_max=jnp.full((n_worlds, n_drones, 1), range_bias_max, device=device),
            steps=-jnp.ones((n_worlds, 1), dtype=jnp.int32, device=device),
            freq=freq,
        )


def simulate_uwb(data: SimData) -> SimData:
    """Simulate one UWB communication update.

    Each range is the true distance to the anchor plus a constant random bias of that anchor and
    zero-mean Gaussian noise, clipped at zero. The ranges only update at the UWB frequency.
    """
    uwb = data.plugins["uwb"]
    mask = controllable(data.core.steps, data.core.freq, uwb.steps, uwb.freq)

    key, subkey = jax.random.split(data.core.rng_key)
    ranges = jnp.linalg.norm(data.states.pos[..., None, :] - uwb.base_stations, axis=-1)
    ranges = ranges + uwb.range_bias + jax.random.normal(subkey, ranges.shape) * uwb.range_std
    ranges = jnp.maximum(ranges, 0.0)

    uwb = leaf_replace(uwb, mask, ranges=ranges, steps=data.core.steps)
    return data.replace(plugins=data.plugins | {"uwb": uwb}, core=data.core.replace(rng_key=key))


def solve_uwb_position(data: SimData) -> SimData:
    """Solve the latest UWB ranges for a position by linear least squares.

    Subtracting the squared range to the first anchor from the squared ranges to the others cancels
    the quadratic term in the position and leaves an overdetermined linear system. The position only
    updates when new ranges arrived in this step.
    """
    uwb = data.plugins["uwb"]
    anchors, ranges = uwb.base_stations, uwb.ranges
    lhs = 2 * (anchors[1:] - anchors[0])
    rhs = ranges[..., :1] ** 2 - ranges[..., 1:] ** 2
    rhs = rhs + jnp.sum(anchors[1:] ** 2 - anchors[0] ** 2, axis=-1)
    updated = uwb.steps == data.core.steps  # UWB ranges are updated in this step
    pos = jnp.where(updated[..., None], rhs @ jnp.linalg.pinv(lhs).T, data.plugins["uwb_pos"])
    return data.replace(plugins=data.plugins | {"uwb_pos": pos})


def uwb_state_controller(data: SimData) -> SimData:
    """Run the state controller on the UWB position instead of the true position."""
    uwb_states = data.states.replace(pos=data.plugins["uwb_pos"])
    return control_state2attitude(data.replace(states=uwb_states)).replace(states=data.states)


def trajectory(t: float) -> np.ndarray:
    """Return a figure-eight state command with velocity and acceleration feedforward."""
    omega = 2 * np.pi / DURATION * np.array([1, 2])  # rad/s, x and y
    amp = np.array([1.5, 1.0])  # m, x and y
    cmd = np.zeros((1, 1, 16))
    cmd[..., :2] = amp * np.sin(omega * t)
    cmd[..., 2] = 1.5
    cmd[..., 3:5] = amp * omega * np.cos(omega * t)
    cmd[..., 6:8] = -amp * omega**2 * np.sin(omega * t)
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    return cmd


def warm_start(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
    """Start on the trajectory at hover as in [1]."""
    hover_force = data.params.mass * -data.params.gravity_vec[2] / 4
    motor_forces = jnp.broadcast_to(hover_force, data.states.rotor_vel.shape)
    rotor_vel = motor_force2rotor_vel(motor_forces, data.params.rpm2thrust)
    init_state = trajectory(0.0)
    pos, vel = init_state[..., :3], init_state[..., 3:6]
    states = leaf_replace(data.states, mask, pos=pos, vel=vel, rotor_vel=rotor_vel)
    return data.replace(states=states)


def reset_uwb_bias(data: SimData, default_data: SimData, mask: Array | None = None) -> SimData:
    """Sample a persistent UWB bias for the selected worlds."""
    key, bias_key = jax.random.split(data.core.rng_key)
    uwb = data.plugins["uwb"]
    range_bias = jax.random.uniform(bias_key, uwb.range_bias.shape) * uwb.range_bias_max
    uwb = leaf_replace(uwb, mask, range_bias=range_bias)
    return data.replace(plugins=data.plugins | {"uwb": uwb}, core=data.core.replace(rng_key=key))


def fly(use_uwb: bool) -> dict[str, np.ndarray]:
    """Fly one loop with the true or the UWB position as feedback."""
    sim = Sim(control="state")
    append_fn(sim.reset_pipeline, warm_start)
    if use_uwb:
        uwb = UWBData.create(
            sim.n_worlds, sim.n_drones, ANCHORS, RANGE_STD, RANGE_BIAS_MAX, UWB_FREQ, sim.device
        )
        uwb_pos = jnp.zeros_like(sim.data.states.pos)
        sim.data = sim.data.replace(plugins=sim.data.plugins | {"uwb": uwb, "uwb_pos": uwb_pos})
        append_fn(sim.reset_pipeline, reset_uwb_bias)
        prepend_fn(sim.step_pipeline, simulate_uwb)
        insert_fn_after(sim.step_pipeline, "simulate_uwb", solve_uwb_position)
        replace_fn(sim.step_pipeline, uwb_state_controller, "state_controller")
    sim.build_default_data()
    sim.build_reset_fn()
    sim.build_step_fn()
    sim.reset()

    log = {key: [] for key in ("ref", "pos", "obs")}
    for i in range(int(DURATION * sim.control_freq)):
        cmd = trajectory(i / sim.control_freq)
        sim.state_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        obs = sim.data.plugins["uwb_pos"] if use_uwb else sim.data.states.pos
        log["ref"].append(cmd[0, 0, :3])
        log["pos"].append(np.asarray(sim.data.states.pos[0, 0]))
        log["obs"].append(np.asarray(obs[0, 0]))
    sim.close()
    return {key: np.array(value) for key, value in log.items()}


def main(plot: bool = True):
    logs = {"perfect": fly(use_uwb=False), "UWB": fly(use_uwb=True)}
    for name, log in logs.items():
        log["obs_error"] = np.linalg.norm(log["obs"] - log["pos"], axis=-1)
        log["tracking_error"] = np.linalg.norm(log["pos"] - log["ref"], axis=-1)
        print(
            f"{name} feedback: observation error RMS {np.sqrt(np.mean(log['obs_error'] ** 2)):.3f}"
            f" m, tracking error RMS {np.sqrt(np.mean(log['tracking_error'] ** 2)):.3f} m"
        )
    if plot:
        plot_results(logs)


def plot_results(logs: dict[str, dict[str, np.ndarray]]):
    import matplotlib.pyplot as plt

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(12, 4))
    ref = logs["perfect"]["ref"]
    t = np.linspace(0, DURATION, len(ref))
    colors = {"perfect": "C0", "UWB": "C1"}
    ax_traj.plot(ref[:, 0], ref[:, 1], "k--", label="reference")
    for name, log in logs.items():
        c = colors[name]
        ax_traj.plot(log["pos"][:, 0], log["pos"][:, 1], c, label=f"{name} feedback")
        ax_err.plot(t, log["obs_error"], c, lw=0.5, alpha=0.5, label=f"{name} observation error")
        ax_err.plot(t, log["tracking_error"], c, label=f"{name} tracking error")
    ax_traj.set(xlabel="x [m]", ylabel="y [m]")
    ax_err.set(xlabel="Time [s]", ylabel="Position error [m]")
    ax_traj.legend()
    ax_err.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
