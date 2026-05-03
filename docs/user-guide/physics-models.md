# Physics Models

Crazyflow supports four physics models, selectable via the `Physics` enum. All models share the same state representation and control interface, so you can swap them at construction time without changing any other code.

```python
from crazyflow.sim import Sim, Physics

sim = Sim(physics=Physics.first_principles)
```

## Available models

| Model | Enum value | Command input | Description |
|---|---|---|---|
| First principles | `Physics.first_principles` | Rotor RPM | Full analytical model with identified parameters |
| SO(3) + RPY | `Physics.so_rpy` | Roll/pitch/yaw + thrust | Simplified fitted model |
| SO(3) + RPY + rotor | `Physics.so_rpy_rotor` | Roll/pitch/yaw + thrust | Adds first-order rotor dynamics |
| SO(3) + RPY + rotor + drag | `Physics.so_rpy_rotor_drag` | Roll/pitch/yaw + thrust | Adds translational and rotational drag |

`Physics.default` resolves to `Physics.first_principles`.

## First-principles model

The first-principles model derives forces and torques analytically from motor speeds using identified physical parameters: mass, arm length, propeller constants, and the full inertia tensor. It operates at the rotor-velocity level and is the most accurate model for sim-to-real transfer.

```python
from crazyflow.sim import Sim, Physics
from crazyflow.control import Control

# Force-torque and rotor_vel control modes require first_principles
sim = Sim(
    physics=Physics.first_principles,
    control=Control.rotor_vel,
)
sim.reset()
```

Parameters accessible through `sim.data.params`:

| Parameter | Description |
|---|---|
| `mass` | Drone mass, kg |
| `J` | Inertia matrix, kg·m² |
| `L` | Motor arm length, m |
| `rpm2thrust` | Thrust coefficient, N/(RPM²) |
| `rpm2torque` | Torque coefficient, Nm/(RPM²) |
| `mixing_matrix` | Maps rotor RPMs² to [thrust, tx, ty, tz] |
| `rotor_dyn_coef` | First-order rotor time constant |

## Fitted models (so_rpy family)

The `so_rpy` models are identified from flight data using a small number of flight minutes. They take higher-level commands (roll/pitch/yaw setpoints + collective thrust in Newtons) and are faster to simulate because they skip the rotor-velocity level.

These models are a good choice when:

- You are training RL agents and want speed over fidelity
- Your controller outputs attitude targets (as most Crazyflie firmware does)
- You do not need rotor-level detail

```python
from crazyflow.sim import Sim, Physics
from crazyflow.control import Control

sim = Sim(
    physics=Physics.so_rpy_rotor_drag,  # most accurate of the fitted family
    control=Control.attitude,
)
sim.reset()
```

The `so_rpy_rotor_drag` variant includes translational drag, which captures the velocity-dependent deceleration effect visible in aggressive flights. It is the recommended fitted model for sim-to-real experiments.

## Control mode compatibility

| Physics model | `Control.state` | `Control.attitude` | `Control.force_torque` | `Control.rotor_vel` |
|---|---|---|---|---|
| `first_principles` | ✓ | ✓ | ✓ | ✓ |
| `so_rpy` | ✓ | ✓ | ✗ | ✗ |
| `so_rpy_rotor` | ✓ | ✓ | ✗ | ✗ |
| `so_rpy_rotor_drag` | ✓ | ✓ | ✗ | ✗ |

!!! warning
    Using `Control.force_torque` or `Control.rotor_vel` with a fitted model raises `ConfigError` at construction time.

## Next steps

- [Control Modes](control-modes.md) — command shapes and the control hierarchy
- [Object-Oriented API](oo-api.md) — full constructor arguments
