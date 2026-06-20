# Integral errors

The Mellinger controller uses integral terms to reject steady-state errors. Because every controller in `crazyflow.control` is a **pure function**, integral state cannot be stored inside the function. It is an explicit return value that you pass back on the next call.

## Initialisation

You have two ways to start the integral error at zero:

1. Pass `None` (or omit the argument). The controller then creates the zero array internally.
2. Create the zero array yourself and pass it explicitly. The integral error has the same shape as the position (for `pos_err_i`) or angular velocity (for `r_int_error`), so a zero array of that shape works.

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")
pos = np.zeros(3)
quat = np.array([0.0, 0.0, 0.0, 1.0])
vel = np.zeros(3)
cmd = np.zeros(13)

# Option 1: let the controller initialise the integral error.
rpyt, pos_err_i = ctrl(pos, quat, vel, cmd, pos_err_i=None)

# Option 2: initialise it yourself.
rpyt, pos_err_i = ctrl(pos, quat, vel, cmd, pos_err_i=np.zeros(3))
```

Under `jax.jit`, prefer option 2. Passing `None` on the first call and an array on the following calls changes the argument structure, so JAX traces and compiles the function twice. Passing a zero array from the start keeps the input structure constant and compiles only once.

## Carrying errors across timesteps

Pass the returned error straight back as `pos_err_i` on the next call:

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")
pos = np.zeros(3)
quat = np.array([0.0, 0.0, 0.0, 1.0])
vel = np.zeros(3)
cmd = np.zeros(13)
cmd[0] = 1.0  # 1 m setpoint error in x

pos_err_i = None
for _ in range(10):
    rpyt, pos_err_i = ctrl(pos, quat, vel, cmd, pos_err_i=pos_err_i)  # carry forward

# After 10 steps at 100 Hz, integral error is approximately 10 * 0.01 = 0.1 m.
```

## Both stages have integral errors

`state2attitude` tracks position error via `pos_err_i`. `attitude2force_torque` tracks angular velocity error via `r_int_error`. Manage them independently:

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import attitude2force_torque, state2attitude

state_ctrl = parametrize(state2attitude, "cf2x_L250")
att_ctrl = parametrize(attitude2force_torque, "cf2x_L250")

pos = np.zeros(3)
quat = np.array([0.0, 0.0, 0.0, 1.0])
vel = np.zeros(3)
ang_vel = np.zeros(3)
cmd = np.zeros(13)

pos_err_i = None
r_int_error = None

for _ in range(5):
    rpyt, pos_err_i = state_ctrl(pos, quat, vel, cmd, pos_err_i=pos_err_i)
    force, torque, r_int_error = att_ctrl(quat, ang_vel, rpyt, r_int_error=r_int_error)
```
