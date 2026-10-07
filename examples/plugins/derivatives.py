"""Example of adding the state derivatives to the step pipeline with plugins.

Evaluating the dynamics gives the exact derivative at the current state. Finite differences give the
exact average derivative from the last to the current step. The simulation does not store either by
default because it costs performance and they are rarely needed.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

os.environ["SCIPY_ARRAY_API"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.dynamics.first_principles import sim_dynamics
from crazyflow.sim import Sim
from crazyflow.sim.data import SimStateDeriv
from crazyflow.sim.pipeline import append_fn

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

DURATION = 10.0  # s, one loop of the figure-eight


def dynamics_deriv(data: SimData) -> SimData:
    """Evaluate the dynamics at the current state."""
    return data.replace(plugins=data.plugins | {"states_deriv": sim_dynamics(data)})


def finite_diff_deriv(data: SimData) -> SimData:
    """Differentiate the states over the last step."""
    prev, states, freq = data.plugins["prev_states"], data.states, data.core.freq
    rot = R.from_quat(prev.quat).inv() * R.from_quat(states.quat)
    deriv = SimStateDeriv(
        vel=(states.pos - prev.pos) * freq,
        ang_vel=rot.as_rotvec() * freq,
        acc=(states.vel - prev.vel) * freq,
        ang_acc=(states.ang_vel - prev.ang_vel) * freq,
        rotor_acc=(states.rotor_vel - prev.rotor_vel) * freq,
    )
    return data.replace(plugins=data.plugins | {"fd_states_deriv": deriv, "prev_states": states})


def trajectory(t: float) -> np.ndarray:
    """Return a figure-eight state command."""
    omega = 2 * np.pi / DURATION
    cmd = np.zeros((1, 1, 16))
    cmd[..., :3] = [2 * np.sin(omega * t), np.sin(2 * omega * t), 1.0]
    cmd[..., 9:13] = R.from_euler("z", 0.0).as_quat()
    return cmd


def main(plot: bool = True):
    sim = Sim(dynamics="first_principles", control="state", integrator="rk4")
    pos = jnp.asarray(trajectory(0.0)[..., :3], device=sim.device)
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
    plugins = {
        "states_deriv": SimStateDeriv.create(sim.n_worlds, sim.n_drones, sim.device),
        "fd_states_deriv": SimStateDeriv.create(sim.n_worlds, sim.n_drones, sim.device),
        "prev_states": sim.data.states,
    }
    sim.data = sim.data.replace(plugins=sim.data.plugins | plugins)

    # Append after integration step
    append_fn(sim.step_pipeline, dynamics_deriv)
    append_fn(sim.step_pipeline, finite_diff_deriv)
    sim.build_default_data()
    sim.build_step_fn()

    log = {"states_deriv": [], "fd_states_deriv": []}
    for i in range(int(2 * DURATION * sim.control_freq)):
        sim.state_control(trajectory(i / sim.control_freq))
        sim.step(sim.freq // sim.control_freq)
        for key in log:
            log[key].append(jax.tree.map(lambda x: np.asarray(x[0, 0]), sim.data.plugins[key]))
    sim.close()

    dynamics, fd = (jax.tree.map(lambda *x: np.stack(x), *log[key]) for key in log)
    for name in ("vel", "ang_vel", "acc", "ang_acc", "rotor_acc"):
        x, x_fd = getattr(dynamics, name), getattr(fd, name)
        print(f"{name}: max relative difference {np.abs(x - x_fd).max() / np.abs(x).max():.1e}")

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 7))
        quantities = (
            ("acc", "Linear acceleration", "m/s$^2$"),
            ("ang_acc", "Angular acceleration", "rad/s$^2$"),
        )
        for col, (name, title, unit) in enumerate(quantities):
            x, x_fd = getattr(dynamics, name), getattr(fd, name)
            x, x_fd = x[len(x) // 2 :], x_fd[len(x_fd) // 2 :]
            t = np.arange(len(x)) / sim.control_freq
            for i, axis in enumerate("xyz"):
                label = f"{axis} finite differences"
                axes[0, col].plot(t, x_fd[:, i], f"C{i}", lw=2.5, alpha=0.6, label=label)
                axes[1, col].plot(t, x_fd[:, i] - x[:, i], f"C{i}", lw=0.8)
            axes[0, col].plot(t, x, "k--", lw=0.8)
            axes[0, col].set(title=title, ylabel=f"{title} [{unit}]")
            axes[1, col].set(xlabel="Time [s]", ylabel=f"Finite differences - dynamics [{unit}]")
        axes[0, 0].plot([], [], "k--", lw=0.8, label="dynamics")
        axes[0, 0].legend()
        for ax in axes.flat:
            ax.grid()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
