"""Query contacts between drones and the scene with sphere and box collision geometry.

The default collision geometry is a sphere around the drone frame. `use_box_collision` replaces it
with a tighter oriented box, useful for narrow-gap flight and accurate contact debugging. The drones
take off once with each geometry and touch the floor until they lift off. Press 3 in the viewer to
show the collision geometry.
"""

import numpy as np

from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.sim import use_box_collision


def main():
    """Spawn multiple drones in multiple worlds and check for contacts with both geometries."""
    n_worlds, n_drones = 2, 3
    fps = 60
    for box in (False, True):
        sim = Sim(n_worlds=n_worlds, n_drones=n_drones, dynamics=Dynamics.so_rpy)
        use_box_collision(sim, enable=box)

        cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))
        cmd[..., 3] = sim.data.params.mass[0] * 9.81 * 1.1
        for i in range(int(2 * sim.control_freq)):
            sim.attitude_control(cmd)
            sim.step(sim.freq // sim.control_freq)
            if ((i * fps) % sim.control_freq) < fps:
                sim.render()
                print(f"Contacts: {sim.contacts().any()}")
        sim.close()


if __name__ == "__main__":
    main()
