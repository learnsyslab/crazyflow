"""Hardware descriptions for the supported drone platforms.

This package bundles the physical assets that define each drone configuration: the MuJoCo MJCF scene
files, their referenced meshes (``assets/``), and the core physical parameters shared across all
dynamics (``params.toml`` with mass, inertia, thrust limits, and the gravity vector). These describe
the *hardware* and are independent of the dynamics formulation used to simulate it
(see [crazyflow.dynamics][]).

Use ``available_drones`` to enumerate the supported configurations and
[crazyflow.dynamics.load_params][] to read the parameters of a drone.
"""

# Currently supported platforms:
# * **cf2x_L250** — Crazyflie 2.x
# * **cf2x_P250** — Crazyflie 2.x with plus propellers
# * **cf2x_T350** — Crazyflie 2.x with thrust upgrade kit
# * **cf21B_500** — Crazyflie 2.1 Brushless with 500 mAh battery
available_drones: tuple[str, ...] = ("cf2x_L250", "cf2x_P250", "cf2x_T350", "cf21B_500")

__all__ = ["available_drones"]
