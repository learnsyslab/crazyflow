# Adding a drone

A drone is defined by data files only. The `Drone` enum lists the MJCF files in `crazyflow/drones`, and a dynamics model or controller supports a drone when its own `params.toml` has a section for it. Adding a platform therefore means adding files and sections, not registering anything in Python.

```python
from crazyflow import Drone
from crazyflow.dynamics import supported_dynamics

Drone.cf2x_L250  # 'cf2x_L250'
supported_dynamics(Drone.cf2x_L250)  # (first_principles, so_rpy, so_rpy_rotor, so_rpy_rotor_drag)
```

Pick a short name such as `cf2x_L250` (platform, then variant) and use it everywhere below.

## 1. MuJoCo model

Add `crazyflow/drones/<name>.xml` with its meshes under `crazyflow/drones/assets/<name>/`. This file is what makes the drone appear in `Drone`. The simulator attaches the body named `drone` once per drone, so that body is required. If you also provide a `drone_fused` body whose visual geometry is a single mesh, users can select it with `Sim(fused_mjx_model=True)` for cheaper rendering. See [MuJoCo Integration](mujoco.md) for how the scene is assembled.

## 2. Dynamics parameters

Each dynamics model has its own `crazyflow/dynamics/<dynamics>/params.toml`. Add a `[<name>]` section to every model you want to offer for the drone. The commented example at the top of each file lists the keys the model needs, and all of them must be set:

- Mass, inertia and the per-motor thrust limits appear in every model. The fitted `so_rpy` models only use the inertia to apply external torques, so an estimate works there at the cost of wrong reactions to disturbance torques.
- `first_principles` additionally needs the hardware constants: arm length, thrust and torque curves and mixing matrix. The remaining keys have to be set but not identified: `rotor_dyn_coef = [1/tau, 0.0, 1/tau, 0.0]` is a symmetric first order rotor model with time constant `tau`, a zero `drag_matrix` disables drag, and a zero `prop_inertia` drops the gyroscopic torque of the propellers.
- The fitted `so_rpy`, `so_rpy_rotor` and `so_rpy_rotor_drag` models need identified coefficients. Use the [system identification pipeline](dynamics/system-identification.md) to obtain them from flight data.

Gravity is global and lives in `crazyflow/dynamics/params.toml`. A model without a section is simply not offered for that drone. [`supported_dynamics`][crazyflow.dynamics.supported_dynamics] and [`supported_drones`][crazyflow.dynamics.supported_drones] report the available pairs, and `Sim` raises `KeyError` for any other combination.

## 3. Controller parameters

Add `[<name>.core]`, `[<name>.state2attitude]`, `[<name>.attitude2force_torque]` and `[<name>.body_rate2force_torque]` sections to `crazyflow/control/mellinger/params.toml`. These reproduce the onboard firmware, so the values may deliberately differ from the physical constants in step 2. See [Mellinger controller](control/mellinger.md).

## 4. Documentation

Add the platform to the table in [Parametrization](dynamics/parametrize.md#available-drone-configurations).

## 5. Run the tests

The test suite parametrizes over `Drone` and over the supported drone-dynamics pairs, so the new drone is tested without any changes to the tests. In particular, `tests/integration/test_models.py` constructs a `Sim` for every supported pair, which loads the MJCF, the dynamics parameters and the controller parameters together.

```bash
pixi run -e tests tests
```
