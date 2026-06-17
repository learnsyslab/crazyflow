# Transform utilities

These utilities sit below the dynamics functions and deal with the physical conversions that arise when interfacing with hardware or building custom dynamics components. You won't need them for most use cases, but they are useful when working directly with motor forces, PWM commands, or angular velocity representations.

## Motor and rotor conversions

These functions live in [`crazyflow.dynamics.transform`][crazyflow.dynamics.transform] and convert between motor RPM, body forces, body torques, and PWM.

### `motor_force2rotor_vel`

Inverts the polynomial thrust model (`rpm2thrust`) to find the RPM that produces a given force. Useful when you have a desired per-motor thrust and need to convert it to a motor command.

```python
import numpy as np
from crazyflow.dynamics.core import load_params
from crazyflow.dynamics.transform import motor_force2rotor_vel

params = load_params("first_principles", "cf2x_L250")
forces = np.array([0.08, 0.08, 0.08, 0.08])  # [N] per motor
rotor_vel = motor_force2rotor_vel(forces, params["rpm2thrust"])  # [RPM]
```

### `rotor_vel2body_force`

Computes the total thrust force vector in the body frame from motor RPMs. For a level drone this is purely along the z-axis.

```python
import numpy as np
from crazyflow.dynamics.core import load_params
from crazyflow.dynamics.transform import rotor_vel2body_force

params = load_params("first_principles", "cf2x_L250")
rotor_vel = np.array([14_000.0, 14_000.0, 14_000.0, 14_000.0])  # [RPM]
body_force = rotor_vel2body_force(rotor_vel, params["rpm2thrust"])
# shape (..., 3)
```

### `force2pwm` and `pwm2force`

Convert between per-motor thrust in Newtons and the PWM integer sent to the motor controller. Useful when interfacing with Crazyflie firmware, which communicates in PWM.

```python
from crazyflow.dynamics.core import load_params
from crazyflow.dynamics.transform import force2pwm, pwm2force

params = load_params("first_principles", "cf2x_L250")
pwm = force2pwm(0.08, params["thrust_max"], params["pwm_max"])
force_back = pwm2force(pwm, params["thrust_max"], params["pwm_max"])
```

All four functions support batched inputs: leading batch dimensions are broadcast through.

## Rotation utilities

These functions live in [`crazyflow.dynamics.utils.rotation`][crazyflow.dynamics.utils.rotation] and convert between angular velocity representations. They are useful when mixing dynamics that use different rotational state variables, or when implementing a state estimator that works in Euler angles while the dynamics use quaternions.

```python
import numpy as np
from crazyflow.dynamics.utils.rotation import (
    ang_vel2quat_dot,
    ang_vel2rpy_rates,
    rpy_rates2ang_vel,
)

quat = np.array([0.0, 0.0, 0.0, 1.0])
ang_vel = np.zeros(3)

rpy_rates = ang_vel2rpy_rates(quat, ang_vel)  # (..., 3)
ang_vel = rpy_rates2ang_vel(quat, rpy_rates)  # (..., 3)
quat_dot = ang_vel2quat_dot(quat, ang_vel)  # (..., 4)
```

All three support arbitrary leading batch dimensions on `quat` and `ang_vel` / `rpy_rates`.

---

For the complete function signatures and docstrings, see the [API reference](../../api/index.md).
