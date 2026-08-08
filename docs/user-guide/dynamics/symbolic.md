# Symbolic dynamics (CasADi)

Optimization-based control methods like nonlinear MPC, trajectory optimization and moving-horizon estimation need symbolic dynamics: an expression graph the solver can differentiate through and evaluate analytically. Every dynamics in `crazyflow.dynamics` has a `symbolic_dynamics` function that returns [CasADi](https://web.casadi.org/) `MX` expressions, validated to be numerically equivalent to the numeric `dynamics` implementation.

The return signature is always `(X_dot, X, U, Y)`:

- `X` — state vector (CasADi `MX` column vector)
- `U` — input vector
- `X_dot` — state derivative, as a symbolic expression in `X` and `U`
- `Y` — output vector (position + attitude)

Physical parameters are bound with [`parametrize`][crazyflow.dynamics.parametrize], exactly as for the numeric dynamics.

## first_principles

```python
import casadi as cs
from crazyflow.dynamics.first_principles import symbolic_dynamics
from crazyflow.dynamics.core import parametrize

symbolic_dynamics = parametrize(symbolic_dynamics, "cf2x_L250")

# include rotor velocity in the state vector
X_dot, X, U, Y = symbolic_dynamics(model_rotor_vel=True, model_dist_f=False, model_dist_t=False)

f = cs.Function("f", [X, U], [X_dot])
```

State vector layout with `model_rotor_vel=True`:

| Indices | Variable | Units |
|---|---|---|
| 0–2 | `pos` | m |
| 3–6 | `quat` (xyzw) | — |
| 7–9 | `vel` | m/s |
| 10–12 | `ang_vel` | rad/s |
| 13–16 | `rotor_vel` | RPM |

See [`crazyflow.dynamics.first_principles`][crazyflow.dynamics.first_principles] in the API reference for the full list of accepted parameters.

## Fitted dynamics — quaternion form

`symbolic_dynamics` on the fitted dynamics converts them to quaternion + angular velocity state, matching the `dynamics` function signature. This makes it straightforward to swap between `first_principles` and a fitted dynamics in a solver without changing the state layout.

```python
from crazyflow.dynamics.so_rpy_rotor_drag import symbolic_dynamics
from crazyflow.dynamics.core import parametrize

X_dot, X, U, Y = parametrize(symbolic_dynamics, "cf2x_L250")(model_rotor_vel=True)
```

## Fitted dynamics — Euler form

The fitted dynamics also expose `symbolic_dynamics_euler`, which works directly in roll/pitch/yaw + RPY-rate state. This is the natural representation of these dynamics — they are fitted in Euler angles — and it avoids the trigonometric overhead of converting to and from quaternions inside the solver. For most NMPC applications on the fitted dynamics, this is the variant to use.

```python
from crazyflow.dynamics.so_rpy_rotor_drag import symbolic_dynamics_euler
from crazyflow.dynamics.core import parametrize

symbolic_dynamics_euler = parametrize(symbolic_dynamics_euler, "cf2x_L250")
X_dot, X, U, Y = symbolic_dynamics_euler(model_rotor_vel=True)
```

State vector layout with `model_rotor_vel=True`:

| Indices | Variable | Units |
|---|---|---|
| 0–2 | `pos` | m |
| 3–5 | `rpy` (roll/pitch/yaw) | rad |
| 6–8 | `vel` | m/s |
| 9–11 | `drpy` (RPY rates) | rad/s |
| 12–15 | thrust state (only index 12 enters the dynamics) | N |

## Wrapping for Acados / IPOPT

Both functions return raw CasADi expressions. Wrap them in a `cs.Function` to pass to any CasADi-based solver:

```python
import casadi as cs
from crazyflow.dynamics.so_rpy_rotor_drag import symbolic_dynamics_euler
from crazyflow.dynamics.core import parametrize

symbolic_dynamics_euler = parametrize(symbolic_dynamics_euler, "cf2x_L250")
X_dot, X, U, Y = symbolic_dynamics_euler(model_rotor_vel=True)

f = cs.Function("f", [X, U], [X_dot])
# Pass f directly to Acados, IPOPT, or any CasADi-based solver
```

## With disturbance states

Setting `model_dist_f=True` or `model_dist_t=True` appends the disturbance vectors to the state, which is useful for augmented-state estimators:

```python
from crazyflow.dynamics.first_principles import symbolic_dynamics
from crazyflow.dynamics.core import parametrize

symbolic_dynamics = parametrize(symbolic_dynamics, "cf2x_L250")

# dist_f (3,) and dist_t (3,) appended to state
X_dot, X, U, Y = symbolic_dynamics(model_rotor_vel=True, model_dist_f=True, model_dist_t=True)
# X is now 17 + 3 + 3 = 23 elements long
```

---

The dynamics covered so far all come with pre-fitted parameters for the supported Crazyflie platforms. For other drones, the next page explains how to extract parameters from your own flight data.
