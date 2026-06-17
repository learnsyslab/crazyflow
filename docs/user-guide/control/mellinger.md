# Mellinger controller

The Mellinger controller converts a full-state setpoint into individual motor speeds through three chained pure functions. The implementation closely follows the Crazyflie firmware to minimise sim-to-real gap.

## State representation

All three stages share the same state convention:

| Variable | Shape | Units | Description |
|---|---|---|---|
| `pos` | `(..., 3)` | m | Position in world frame |
| `quat` | `(..., 4)` | | Attitude as unit quaternion, scalar-last `xyzw` |
| `vel` | `(..., 3)` | m/s | Linear velocity in world frame |
| `ang_vel` | `(..., 3)` | rad/s | Angular velocity in body frame |

## Stage 1: State to attitude {#state-to-attitude}

`state2attitude` is the position control loop. It converts a full-state setpoint into an attitude and collective thrust command (RPYT).

**Inputs:**

| Argument | Shape | Description |
|---|---|---|
| `pos` | `(..., 3)` | Current position [m] |
| `quat` | `(..., 4)` | Current attitude, xyzw |
| `vel` | `(..., 3)` | Current velocity [m/s] |
| `cmd` | `(..., 13)` | Setpoint: `[x, y, z, vx, vy, vz, ax, ay, az, yaw, rr, pr, yr]` |
| `pos_err_i` | `(..., 3)` or `None` | Position integral error from the previous call. `None` initialises to zero |
| `ctrl_freq` | `float` | Control frequency in Hz (default 100) |

**Outputs:**

| Return | Shape | Description |
|---|---|---|
| `rpyt` | `(..., 4)` | Attitude + thrust: `[roll_rad, pitch_rad, yaw_rad, thrust_N]` |
| `pos_err_i` | `(..., 3)` | Position integral error. Pass back as `pos_err_i` on the next call |

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")

pos = np.zeros(3)
quat = np.array([0.0, 0.0, 0.0, 1.0])
vel = np.zeros(3)
cmd = np.zeros(13)  # setpoint at origin, yaw = 0

rpyt, pos_err_i = ctrl(pos, quat, vel, cmd)
rpyt.shape  # (4,)
pos_err_i.shape  # (3,)
```

## Stage 2: Attitude to force/torque {#attitude-to-force-torque}

`attitude2force_torque` is the attitude control loop. It converts an RPYT command into collective thrust and body-frame torques.

**Inputs:**

| Argument | Shape | Description |
|---|---|---|
| `quat` | `(..., 4)` | Current attitude, xyzw |
| `ang_vel` | `(..., 3)` | Current angular velocity in body frame [rad/s] |
| `cmd` | `(..., 4)` | RPYT from stage 1: `[roll_rad, pitch_rad, yaw_rad, thrust_N]` |
| `prev_ang_vel` | `(..., 3)` or `None` | Angular velocity from the previous call. `None` initialises to zero |
| `r_int_error` | `(..., 3)` or `None` | Angular velocity integral error from the previous call. `None` initialises to zero |
| `ctrl_freq` | `int` | Control frequency in Hz (default 500) |

**Outputs:**

| Return | Shape | Description |
|---|---|---|
| `force` | `(..., 1)` | Collective thrust [N] |
| `torque` | `(..., 3)` | Body-frame torques [N·m] |
| `r_int_error` | `(..., 3)` | Angular velocity integral error. Pass back as `r_int_error` on the next call |

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import attitude2force_torque

ctrl = parametrize(attitude2force_torque, "cf2x_L250")

quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity, no rotation
ang_vel = np.zeros(3)
cmd = np.array([0.0, 0.0, 0.0, 0.3])  # level attitude, 0.3 N thrust

force, torque, r_int_err = ctrl(quat, ang_vel, cmd)
force.shape  # (1,)
torque.shape  # (3,)
```

## Stage 3: Force/torque to rotor velocities {#force-torque-to-rotor-velocities}

`force_torque2rotor_vel` converts collective thrust and body-frame torques into individual motor speeds, accounting for the motor mixing matrix.

**Inputs:**

| Argument | Shape | Description |
|---|---|---|
| `force` | `(..., 1)` | Desired collective thrust [N] |
| `torque` | `(..., 3)` | Desired body-frame torques [N·m] |

**Outputs:**

| Return | Shape | Description |
|---|---|---|
| `rotor_speeds` | `(..., 4)` | Individual motor speeds [RPM] |

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import force_torque2rotor_vel

ctrl = parametrize(force_torque2rotor_vel, "cf2x_L250")

force = np.array([0.2])  # total thrust [N]
torque = np.zeros(3)  # no corrective torque

rotor_speeds = ctrl(force, torque)
rotor_speeds.shape  # (4,)
```

## Chaining all three stages

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import (
    attitude2force_torque,
    force_torque2rotor_vel,
    state2attitude,
)

state_ctrl = parametrize(state2attitude, "cf2x_L250")
att_ctrl = parametrize(attitude2force_torque, "cf2x_L250")
rotor_ctrl = parametrize(force_torque2rotor_vel, "cf2x_L250")

pos = np.array([0.0, 0.0, 1.0])  # 1 m altitude
quat = np.array([0.0, 0.0, 0.0, 1.0])
vel = np.zeros(3)
ang_vel = np.zeros(3)
cmd = np.zeros(13)
cmd[:3] = np.array([0.0, 0.0, 1.0])  # hover at 1 m

rpyt, _ = state_ctrl(pos, quat, vel, cmd)
force, torque, _ = att_ctrl(quat, ang_vel, rpyt)
rotor_speeds = rotor_ctrl(force, torque)
rotor_speeds.shape  # (4,)
```

Integral errors from each stage should be passed back on the next call. See [Integral errors](integral-errors.md) for the full pattern.
