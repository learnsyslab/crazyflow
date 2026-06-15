# API Reference

This section is auto-generated from the Crazyflow source code using [mkdocstrings](https://mkdocstrings.github.io/).

## Module overview

| Module | Description |
|---|---|
| `crazyflow.sim` | Core `Sim` class and simulation pipeline |
| `crazyflow.sim.pipeline` | `OrderedDict`-based pipeline helpers (`append_fn`, `insert_fn_before`, `replace_fn`, etc.) |
| `crazyflow.sim.data` | `SimData`, `SimState`, `SimControls`, `SimParams`, `SimCore` pytrees |
| `crazyflow.sim.functional` | Pure functional control API for use inside `jax.jit` |
| `crazyflow.dynamics` | `Dynamics` enum and dynamics model implementations |
| `crazyflow.sim.integration` | `Integrator` enum, Euler, RK4, and symplectic Euler |
| `crazyflow.sim.sensors` | Raycasting and sensor extraction utilities |
| `crazyflow.sim.symbolic` | CasADi symbolic model API |
| `crazyflow.control` | `Control` enum |
| `crazyflow.control.mellinger` | Mellinger controller data and parameters |
| `crazyflow.envs` | Gymnasium vectorized environments |
| `crazyflow.utils` | Grid utilities and pytree helpers |

Navigate the full generated reference using the sidebar.
