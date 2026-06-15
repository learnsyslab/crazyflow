"""crazyflow.drones: hardware descriptions for the supported drone platforms.

This package bundles the physical assets that define each drone configuration: the
MuJoCo MJCF scene files, their referenced meshes (``assets/``), and the physical
parameters shared across all dynamics models (``params.toml`` — mass, inertia, thrust
and torque curves, …). These describe the *hardware* and are independent of the
dynamics formulation used to simulate it (see [crazyflow.dynamics][]).

Use [available_drones][crazyflow.drones.available_drones] to enumerate the supported
configurations.
"""

# Currently supported platforms:
# * **cf2x_L250** — Crazyflie 2.x with L250 propellers (31.9 g)
# * **cf2x_P250** — Crazyflie 2.x with P250 propellers (31.8 g)
# * **cf2x_T350** — Crazyflie 2.x with T350 propellers (37.9 g)
# * **cf21B_500** — Crazyflie 2.1 Brushless with 500 propellers (43.4 g)
available_drones: tuple[str, ...] = ("cf2x_L250", "cf2x_P250", "cf2x_T350", "cf21B_500")

__all__ = ["available_drones"]
