# Batching

All controllers are built on Array API operations that broadcast over leading dimensions. Add a leading batch dimension to your state and command arrays and the controller evaluates all instances in a single call, with no loops and no special API.

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")

N = 100
pos = np.zeros((N, 3))
quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (N, 1))
vel = np.zeros((N, 3))
cmd = np.zeros((N, 13))

rpyt, int_pos_err = ctrl(pos, quat, vel, cmd)
rpyt.shape  # (100, 4)
```

## Higher-dimensional batches

Any number of leading dimensions works. A common pattern is a grid of environments, each containing multiple drones:

```python
import numpy as np
from crazyflow.control import parametrize
from crazyflow.control.mellinger import state2attitude

ctrl = parametrize(state2attitude, "cf2x_L250")

# 10 environments, 5 drones each
pos = np.zeros((10, 5, 3))
quat = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (10, 5, 4)).copy()
vel = np.zeros((10, 5, 3))
cmd = np.zeros((10, 5, 13))

rpyt, _ = ctrl(pos, quat, vel, cmd)
rpyt.shape  # (10, 5, 4)
```
