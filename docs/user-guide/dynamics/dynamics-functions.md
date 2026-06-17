# Dynamics functions

A dynamics function takes the current state of the drone and a command, and returns the time derivatives of that state. Integrate those derivatives forward and you have a simulation step; evaluate them symbolically and you have an MPC model. The same function serves both purposes.

Every dynamics in `crazyflow.dynamics` shares the same state representation:

| Variable | Shape | Description |
|---|---|---|
| `pos` | `(3,)` | Position in world frame [m] |
| `quat` | `(4,)` | Attitude as unit quaternion, scalar-last `xyzw` |
| `vel` | `(3,)` | Linear velocity in world frame [m/s] |
| `ang_vel` | `(3,)` | Angular velocity in body frame [rad/s] |

What differs between dynamics is the command interface, which parameters are needed, and how much physical detail is captured. The table below gives a quick overview — the sections that follow explain each one and when to reach for it.

| Module | `cmd` input | Rotor dynamics | Key added params |
|---|---|---|---|
| `first_principles` | Motor RPMs `(4,)` | Yes | `rpm2thrust`, `rpm2torque`, `mixing_matrix`, `L`, `prop_inertia` |
| `so_rpy_rotor_drag` | rpyt `(4,)` | Yes | `thrust_time_coef`, `drag_matrix` |
| `so_rpy_rotor` | rpyt `(4,)` | Yes | `thrust_time_coef` |
| `so_rpy` | rpyt `(4,)` | No | — |

## first_principles

The full rigid-body dynamics, derived analytically from first principles. The command is four individual motor RPMs. It computes forces and torques from the RPMs using polynomial thrust and torque curves, applies the mixing matrix to find body-frame moments, and integrates using Newton–Euler equations. Propeller inertia and gyroscopic effects are included. No fitting to flight data is required — all parameters are physical constants you can measure or look up.

Working at the rotor-velocity level means you need a controller that converts higher-level commands, such as position setpoints or attitude plus collective thrust, down to individual motor RPMs. The [controllers](../control/index.md) in `crazyflow.control` provide a matching set designed for exactly this interface.

```python
import numpy as np
from crazyflow.dynamics import parametrize
from crazyflow.dynamics.first_principles import dynamics

dynamics = parametrize(dynamics, drone="cf2x_L250")

pos, vel, ang_vel = np.zeros((3,)), np.zeros((3,)), np.zeros((3,))
quat = np.array([0.0, 0.0, 0.0, 1.0])
cmd = np.full((4,), 15_000.0)
rotor_vel = np.full((4,), 12_000.0)

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = dynamics(
    pos, quat, vel, ang_vel,
    cmd,  # shape (4,) — motor RPMs
    rotor_vel,  # shape (4,) — current motor RPMs; pass None to skip rotor dynamics
)
```

See [`crazyflow.dynamics.first_principles`][crazyflow.dynamics.first_principles] in the API reference for the full parameter list.

## so_rpy_rotor_drag

A fitted second-order dynamics where the command is `[roll_rad, pitch_rad, yaw_rad, thrust_N]` — the same interface used by most flight controller firmware. First-order thrust dynamics model motor spin-up delay, and a linear body-frame drag term accounts for aerodynamic resistance. All coefficients are identified from flight data rather than derived from physics, which makes it easy to calibrate and well-suited to real-time control.

```{ .python continuation }
from crazyflow.dynamics.so_rpy_rotor_drag import dynamics

dynamics = parametrize(dynamics, drone="cf2x_L250")

# Reuses pos, quat, vel, ang_vel from above; the command interface is what differs
cmd = np.array([0.0, 0.0, 0.0, 0.31])  # [roll_rad, pitch_rad, yaw_rad, thrust_N]
rotor_vel = np.array([0.31])  # shape (1,) — current thrust state [N]; None to skip thrust dynamics

pos_dot, quat_dot, vel_dot, ang_vel_dot, rotor_vel_dot = dynamics(
    pos, quat, vel, ang_vel, cmd, rotor_vel
)
```

## so_rpy_rotor

The same as `so_rpy_rotor_drag` but without the drag term. Use this when aerodynamic drag is negligible — for example, in low-speed indoor flight — or when you want a slightly simpler dynamics to calibrate.

```python
from crazyflow.dynamics.so_rpy_rotor import dynamics
```

## so_rpy

The simplest dynamics: no rotor dynamics, no drag. The attitude dynamics are a fitted second-order system driven directly by the roll/pitch/yaw command. It does not accept a `rotor_vel` argument and returns four derivatives instead of five. This is the fastest to evaluate and the easiest to understand, making it a good baseline for control design and learning-based methods where simulation throughput matters most.

```python
from crazyflow.dynamics.so_rpy import dynamics
```

## External disturbances

All four dynamics accept optional `dist_f` (external force, world frame, N) and `dist_t` (external torque, body frame, N·m) arguments. These are useful for modelling wind, contact forces, or other perturbations without modifying the dynamics itself.

```python
import numpy as np
from crazyflow.dynamics import parametrize
from crazyflow.dynamics.so_rpy import dynamics

dynamics = parametrize(dynamics, drone="cf2x_L250")
pos, vel, ang_vel = np.zeros((3,)), np.zeros((3,)), np.zeros((3,))
quat = np.array([0.0, 0.0, 0.0, 1.0])
cmd = np.array([0.0, 0.0, 0.0, 0.31])
dist_f = np.array([0.05, 0.0, 0.0])  # 50 mN headwind [N]
dist_t = np.zeros(3)

# so_rpy has no rotor dynamics, so it returns four derivatives
pos_dot, quat_dot, vel_dot, ang_vel_dot = dynamics(
    pos, quat, vel, ang_vel, cmd, dist_f=dist_f, dist_t=dist_t
)
```

## Checking rotor dynamics support

If you are writing code that works with multiple dynamics, [`dynamics_features`][crazyflow.dynamics.dynamics_features] tells you programmatically whether a given dynamics function supports rotor dynamics.

```python
from crazyflow.dynamics import dynamics_features
from crazyflow.dynamics.first_principles import dynamics as fp
from crazyflow.dynamics.so_rpy import dynamics as srpy

dynamics_features(fp)  # {'rotor_dynamics': True}
dynamics_features(srpy)  # {'rotor_dynamics': False}
```

---

With an understanding of the available dynamics, the next step is binding one to a specific drone configuration. That's what [`parametrize`](parametrize.md) does.
