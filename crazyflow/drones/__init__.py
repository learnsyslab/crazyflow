"""Hardware descriptions for the supported drone platforms.

This package bundles the physical assets that define each drone configuration: the MuJoCo MJCF scene
files, their referenced meshes (``assets/``), and the physical parameters shared across all dynamics
(``params.toml`` with mass, inertia, thrust and torque curves, …). These describe the *hardware* and
are independent of the dynamics formulation used to simulate it (see [crazyflow.dynamics][]).

Use ``available_drones`` to enumerate the supported configurations, and ``load_params`` to read all
physical parameters of a drone.
"""

import tomllib
from pathlib import Path

# Currently supported platforms:
# * **cf2x_L250** — Crazyflie 2.x
# * **cf2x_P250** — Crazyflie 2.x with plus propellers
# * **cf2x_T350** — Crazyflie 2.x with thrust upgrade kit
# * **cf21B_500** — Crazyflie 2.1 Brushless with 500 mAh battery
available_drones: tuple[str, ...] = ("cf2x_L250", "cf2x_P250", "cf2x_T350", "cf21B_500")

__all__ = ["available_drones", "load_params"]


def load_params(drone: str) -> dict:
    """Load all physical parameters of a drone from ``params.toml``.

    Returns the raw values (lists/scalars) for the whole drone.

    Args:
        drone: Name of the drone configuration, e.g. ``"cf2x_L250"``.
    """
    with open(Path(__file__).parent / "params.toml", "rb") as f:
        params = tomllib.load(f)
    if drone not in params or drone not in available_drones:
        raise KeyError(f"Drone `{drone}` not found in drones/params.toml")
    return params[drone]
