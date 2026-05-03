# Object-Oriented API

The `Sim` class is the main entry point. It provides a Python-level control loop that is easy to script and debug.

!!! note
    The OO API is not compatible with JAX transformations. If you need to run simulation inside `jax.jit`, `jax.grad`, or `jax.lax.scan`, use the [Functional API](functional-api.md) instead.

## Creating a Sim

All configuration is fixed at construction time.

```python
from crazyflow.sim import Sim, Physics
from crazyflow.sim.integration import Integrator
from crazyflow.control import Control

sim = Sim(
    n_worlds=1,
    n_drones=1,
    drone_model="cf2x_L250",       # Crazyflie 2.x with L250 props
    physics=Physics.first_principles,
    control=Control.state,
    integrator=Integrator.rk4,
    freq=500,                       # physics update rate, Hz
    state_freq=100,                 # state controller rate, Hz
    attitude_freq=500,              # attitude controller rate, Hz
    device="cpu",
)
sim.reset()
```

Key constructor arguments:

| Argument | Default | Description |
|---|---|---|
| `n_worlds` | 1 | Number of independent parallel environments |
| `n_drones` | 1 | Drones per world |
| `drone_model` | `"cf2x_L250"` | Drone configuration (see `drone_models.available_drones`) |
| `physics` | `Physics.default` | Physics model |
| `control` | `Control.default` | Control mode |
| `integrator` | `Integrator.default` | Numerical integrator |
| `freq` | 500 | Physics frequency, Hz |
| `device` | `"cpu"` | `"cpu"` or `"gpu"` |

## Control methods

All control methods take an array of shape `(n_worlds, n_drones, command_dim)` and stage it for the next `step` call.

### State control

The highest-level interface. A 13-element command sets desired position, velocity, acceleration, yaw, and angular rates. An internal Mellinger controller converts this to attitude commands.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.state)
sim.reset()

# [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]
cmd = np.zeros((1, 1, 13), dtype=np.float32)
cmd[0, 0, 2] = 0.5  # hover at 0.5 m

sim.state_control(cmd)
sim.step(sim.freq // sim.control_freq)
```

### Attitude control

Commands roll, pitch, yaw setpoints (rad) and a collective thrust (N). This level bypasses the position/velocity loop and is suitable for attitude tracking or RL agents that output attitude targets.

```python
import numpy as np
from crazyflow.sim import Sim, Physics
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.attitude, physics=Physics.so_rpy)
sim.reset()

# [roll, pitch, yaw, collective_thrust_N]
cmd = np.zeros((1, 1, 4), dtype=np.float32)
cmd[0, 0, 3] = float(sim.data.params.mass[0, 0, 0]) * 9.81  # hover thrust

sim.attitude_control(cmd)
sim.step(sim.freq // sim.control_freq)
```

### Force-torque control

Direct force and torque input, useful for testing dynamics or custom controllers. Requires `Physics.first_principles`.

```python
import numpy as np
from crazyflow.sim import Sim, Physics
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.force_torque, physics=Physics.first_principles)
sim.reset()

# [collective_force_N, torque_x_Nm, torque_y_Nm, torque_z_Nm]
cmd = np.zeros((1, 1, 4), dtype=np.float32)
cmd[0, 0, 0] = float(sim.data.params.mass[0, 0, 0]) * 9.81  # hover force

sim.force_torque_control(cmd)
sim.step(1)
```

### Rotor velocity control

The lowest level: directly command each motor's RPM. Requires `Physics.first_principles`.

```python
import numpy as np
from crazyflow.sim import Sim, Physics
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.rotor_vel, physics=Physics.first_principles)
sim.reset()

# [rpm_motor_0, rpm_motor_1, rpm_motor_2, rpm_motor_3]
cmd = np.full((1, 1, 4), 15_000.0, dtype=np.float32)

sim.rotor_vel_control(cmd)
sim.step(1)
```

## Stepping and resetting

`sim.step(n_steps)` advances the simulation by `n_steps` physics ticks. On each tick, the full step pipeline runs, including the control stack. Controllers fire at their configured rate (e.g. the state controller at `state_freq`, the attitude controller at `attitude_freq`), not on every physics tick. Between controller ticks, the previously staged command is held.

Passing more steps to a single `step(n_steps)` call is more efficient than multiple `step(1)` calls: XLA compiles the full loop into a single kernel. If you have staged a control command and do not need to set a new one, you can advance the simulation by any number of steps and the controllers will continue firing at the correct rate.

!!! note
    Changing `n_steps` between calls triggers recompilation. Keep it consistent inside a training or evaluation loop.

`sim.reset()` reinitialises all worlds to their default state. Pass a boolean mask of shape `(n_worlds,)` to reset only selected worlds: `True` resets that world, `False` leaves it unchanged. This is useful in RL training loops where episodes end at different times.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=4, n_drones=1, control=Control.state)
sim.reset()  # reset all worlds

# Stage a command and advance 50 physics steps (controllers fire at their rate)
cmd = np.zeros((4, 1, 13), dtype=np.float32)
cmd[..., 2] = 0.5
sim.state_control(cmd)
sim.step(50)

# Reset only worlds 0 and 2, leaving 1 and 3 running
import jax.numpy as jnp
mask = jnp.array([True, False, True, False])
sim.reset(mask=mask)
```

## Reading state

Access any state field through `sim.data.states`:

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=2, n_drones=3, control=Control.state)
sim.reset()

cmd = np.zeros((2, 3, 13), dtype=np.float32)
for _ in range(10):
    sim.state_control(cmd)
    sim.step(sim.freq // sim.control_freq)

# All drones in all worlds
pos = sim.data.states.pos        # (2, 3, 3)
quat = sim.data.states.quat      # (2, 3, 4)
vel = sim.data.states.vel        # (2, 3, 3)

# Drone 1 in world 0
pos_w0_d1 = sim.data.states.pos[0, 1]  # (3,)
```

## Rendering

`sim.render()` opens an interactive MuJoCo viewer or returns an image array for offscreen rendering.

```{ .python notest }
sim.render()                          # interactive window, world 0
sim.render(mode="rgb_array")          # returns (H, W, 3) uint8
sim.render(mode="depth_array")        # returns (H, W) float32
sim.render(world=1, camera="front")   # different world or named camera
sim.close()                           # close the viewer
```

## Domain randomization

Physical parameters can be randomized per-world using the `randomize` helpers:

```python
import jax
import numpy as np
from crazyflow.sim import Sim
from crazyflow.randomize import randomize_inertia, randomize_mass

sim = Sim(n_worlds=4, n_drones=1)
sim.reset()

nominal_mass = sim.data.params.mass
noise = jax.random.normal(jax.random.key(0), nominal_mass.shape) * 2e-3
randomize_mass(sim, nominal_mass + noise)

nominal_J = sim.data.params.J
J_noise = jax.random.normal(jax.random.key(1), nominal_J.shape) * 1e-6
randomize_inertia(sim, nominal_J + J_noise)
```

To randomize only a subset of worlds, pass a boolean mask:

```python
import jax
import numpy as np
from crazyflow.sim import Sim
from crazyflow.randomize import randomize_mass

sim = Sim(n_worlds=4, n_drones=1)
sim.reset()

import jax.numpy as jnp
mask = jnp.array([True, True, False, False])  # only worlds 0 and 1
nominal_mass = sim.data.params.mass
noise = jax.random.normal(jax.random.key(0), nominal_mass.shape) * 2e-3
randomize_mass(sim, nominal_mass + noise, mask=mask)
```

## Next steps

- [Functional API](functional-api.md) — run simulation inside `jax.jit` and `jax.grad`
- [Pipelines](pipelines.md) — insert custom stages for disturbances and logging
