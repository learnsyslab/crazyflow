"""Hardware descriptions for the supported drone platforms.

This package bundles the MuJoCo MJCF scene files that define each drone configuration and their
referenced meshes (``assets/``). For the physical params, see [crazyflow.dynamics.load_params][].

Use ``available_drones`` to enumerate the supported configurations.
"""

# Currently supported platforms:
# * **cf2x_L250** — Crazyflie 2.x
# * **cf2x_P250** — Crazyflie 2.x with plus propellers
# * **cf2x_T350** — Crazyflie 2.x with thrust upgrade kit
# * **cf21B_500** — Crazyflie 2.1 Brushless with 500 mAh battery
available_drones: tuple[str, ...] = ("cf2x_L250", "cf2x_P250", "cf2x_T350", "cf21B_500")

__all__ = ["available_drones"]
