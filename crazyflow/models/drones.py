"""Available drone configurations with pre-fitted parameters.

Each string in [available_drones][crazyflow.models.drones.available_drones] is a valid
``drone_model`` argument for [parametrize][crazyflow.models.core.parametrize] and
[load_params][crazyflow.models.core.load_params]. The corresponding parameter files live in
``crazyflow/models/data/params.toml`` (physical parameters) and in each model sub-package's
``params.toml`` (fitted coefficients).

Currently supported platforms:

* **cf2x_L250** — Crazyflie 2.x with L250 propellers (31.9 g)
* **cf2x_P250** — Crazyflie 2.x with P250 propellers (31.8 g)
* **cf2x_T350** — Crazyflie 2.x with T350 propellers (37.9 g)
* **cf21B_500** — Crazyflie 2.1 Brushless with 500 propellers (43.4 g)
"""

available_drones: tuple[str] = ("cf2x_L250", "cf2x_P250", "cf2x_T350", "cf21B_500")
