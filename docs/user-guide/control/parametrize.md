# Parametrize

Every controller function takes physical parameters (gains, mass, mixing matrix, PWM bounds) as keyword-only arguments, and the exact values differ per drone. Rather than passing them at every call site, [`parametrize`][crazyflow.control.parametrize] loads them for a named drone and binds them upfront, so call sites only need to provide state and command.

The parameters stay individually accessible after binding. Because they are plain keyword-argument defaults on a `functools.partial`, any of them can be overridden at call time, or batched across a set of environments, without re-parametrizing the function. This makes it straightforward to randomize physical properties across a simulated batch.

```python
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")

# Inspect what was bound
list(ctrl.keywords.keys())
# ['mass', 'kp', 'kd', 'ki', 'gravity_vec', 'mass_thrust',
#  'int_err_max', 'thrust_max', 'pwm_max']
```

## Overriding parameters at call time

Because `parametrize` returns a `functools.partial`, the bound parameters are just keyword-argument defaults. Pass a different value at call time to override for that call only; `ctrl.keywords` is not modified:

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")
pos = np.zeros(3)
quat = np.array([0.0, 0.0, 0.0, 1.0])
vel = np.zeros(3)
cmd = np.zeros(13)

# Simulate with a heavier drone for this call only.
rpyt, _ = ctrl(pos, quat, vel, cmd, mass=0.035)
```

To make a change persist across all future calls, mutate `ctrl.keywords` directly:

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")
ctrl.keywords["mass"] = np.float64(0.035)
```

!!! warning
    `ctrl.keywords` is a mutable dict shared across all references to the same partial. Call `parametrize` again for an independent copy.

## Available drone configurations

The following configurations ship with pre-fitted parameters:

| `drone` | Platform |
|---|---|
| `"cf2x_L250"` | Crazyflie 2.x |
| `"cf2x_P250"` | Crazyflie 2.x, plus propellers |
| `"cf2x_T350"` | Crazyflie 2.x, thrust upgrade kit |
| `"cf21B_500"` | Crazyflie 2.1 Brushless |

Pass the drone name as a plain string:

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")
pos = np.zeros(3)
quat = np.array([0.0, 0.0, 0.0, 1.0])
vel = np.zeros(3)
cmd = np.zeros(13)
rpyt, _ = ctrl(pos, quat, vel, cmd)
```

## Loading raw parameters

Use [`load_params`][crazyflow.control.core.load_params] to inspect or override the values that `parametrize` would bind for a specific controller function:

```python
from crazyflow.control.core import load_params
from crazyflow.control.mellinger import state2attitude

params = load_params(state2attitude, "cf2x_L250")
float(params["mass"])  # 0.029
```

## Switching array backends

By default parameters are stored as NumPy arrays. Pass `xp` to convert them upfront, which avoids per-call conversion overhead in frameworks like PyTorch or JAX:

```{ .python notest }
import torch
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250", xp=torch)
```

You can also specify a compute device:

```python
import jax
import jax.numpy as jnp
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250", xp=jnp, device=jax.devices("cpu")[0])
```

The output backend is always inferred from the arrays you pass at call time, regardless of where the parameters live.
