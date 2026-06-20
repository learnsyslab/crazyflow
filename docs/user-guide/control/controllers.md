# Controllers

A controller is a function that maps the current drone state and a command to actuator outputs. Every controller in `crazyflow.control` is:

- **A pure function**: no hidden state; integral errors are explicit return values you pass back on the next call
- **Array-API compatible**: works identically with NumPy, JAX, PyTorch, or any compliant library
- **Batchable**: add leading dimensions to any input array and the function evaluates all instances at once

## The Mellinger pipeline

The Mellinger controller [[1]](#references) is split into three stages that form a pipeline. Each stage can be used on its own, or all three can be chained to convert a full-state setpoint into individual motor speeds.

| Stage | Function | Takes | Produces |
|---|---|---|---|
| 1 | [`state2attitude`](mellinger.md#state-to-attitude) | State + 13-element setpoint | RPYT command + position integral error |
| 2 | [`attitude2force_torque`](mellinger.md#attitude-to-force-torque) | Attitude + RPYT command | Collective force, body torques + angular velocity integral error |
| 3 | [`force_torque2rotor_vel`](mellinger.md#force-torque-to-rotor-velocities) | Force + torques | 4 motor speeds [RPM] |

## Available controllers

| Module | Controller | Stages |
|---|---|---|
| `crazyflow.control.mellinger` | Mellinger | `state2attitude`, `attitude2force_torque`, `force_torque2rotor_vel` |

## References

[1] D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors," ICRA 2011, doi: 10.1109/ICRA.2011.5980409.
