"""External-force implementation of the ground effect in Eq. (15) of Shi et al arXiv:1811.08027v."""
from __future__ import annotations
import os

os.environ["SCIPY_ARRAY_API"] = "1"



from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow.sim.pipeline import insert_fn_before

if TYPE_CHECKING:
    from crazyflow.sim import Sim
    from crazyflow.sim.data import SimData


# Parameters for cf21B_500. Tune MU for the actual airframe/propeller layout.
PROPELLER_DIAMETER = 55e-3  # m
MU = 2.0
MIN_HEIGHT = 0.02  # m; Eq. (15) is not valid arbitrarily close to the floor
MAX_GAIN = 2.0  # avoid the model's singularity near the floor


def ground_effect_fn(data: SimData) -> SimData:
    rpm = data.states.rotor_vel
    c, b, a = (
        data.params.rpm2thrust[..., 0],
        data.params.rpm2thrust[..., 1],
        data.params.rpm2thrust[..., 2],
    )
    nominal_thrust = jnp.sum(c + b * rpm + a * rpm**2, axis=-1)

    height = jnp.maximum(data.states.pos[..., 2], MIN_HEIGHT)
    gain = 1.0 / (1.0 - MU * (PROPELLER_DIAMETER / (8.0 * height)) ** 2)
    gain = jnp.minimum(gain, MAX_GAIN)
    extra_thrust = nominal_thrust * (gain - 1.0)

    # The force follows the body z thrust axis; ``states.force`` needs world coordinates.
    body_force = jnp.zeros_like(data.states.pos).at[..., 2].set(extra_thrust)
    ground_force = R.from_quat(data.states.quat).apply(body_force)
    return data.replace(states=data.states.replace(force=ground_force))


def install_ground_effect(sim: Sim) -> None:
    """Install the force stage immediately before first-principles integration."""
    insert_fn_before(sim.step_pipeline, "integration", ground_effect_fn)
    sim.build_step_fn()


def main(plot: bool = True) -> None:
    from crazyflow.sim import Sim

    sim = Sim(n_drones=1, drone="cf21B_500", control="state")
    install_ground_effect(sim)

    upper_pos = np.array([0.0, 0.0, 0.5])

    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=jnp.array([[upper_pos]]))
    )
    sim.build_default_data()

    duration = 5.0
    speed = 1 / duration

    command = np.zeros((1, 1, 16))
    command[..., 9:13] = [0.0, 0.0, 0.0, 1.0]  
    command[0, 0, :3] = upper_pos
    heights, vertical_forces = [], []

    for step in range(int(duration * sim.control_freq)):
        t = step / sim.control_freq
        command[0, 0, :3] = [0.0, 0.0, 0.5 - speed * t/2]
        command[0, 0, 3:6] = [0, 0.0, -speed]
        sim.state_control(command)
        sim.step(sim.freq // sim.control_freq)
        heights.append(float(sim.data.states.pos[0, 0, 2]))
        vertical_forces.append(float(sim.data.states.force[0, 0, 2]))
        sim.render()

    sim.close()
    if plot:
        import matplotlib.pyplot as plt

        plt.plot(heights, vertical_forces)
        plt.xlabel("Drone height (m)")
        plt.ylabel("Ground-effect force (N)")
        plt.gca().invert_xaxis()
        plt.show()



if __name__ == "__main__":
    main()
