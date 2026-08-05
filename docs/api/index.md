# API Reference

This section is auto-generated from the Crazyflow source code using [mkdocstrings](https://mkdocstrings.github.io/).

## Module overview

| Module | Description |
|---|---|
| `crazyflow.sim` | Core `Sim` class and simulation pipeline |
| `crazyflow.sim.pipeline` | `OrderedDict`-based pipeline helpers (`append_fn`, `insert_fn_before`, `replace_fn`, etc.) |
| `crazyflow.sim.data` | `SimData`, `SimState`, `SimControls`, `SimParams`, `SimCore` pytrees |
| `crazyflow.sim.functional` | Pure functional control API for use inside `jax.jit` |
| `crazyflow.dynamics` | `Dynamics` enum and dynamics implementations |
| `crazyflow.sim.integration` | `Integrator` enum, Euler, RK4, and symplectic Euler |
| `crazyflow.sim.sensors` | Raycast depth (`sensors.depth`) and gaussian splat RGB (`sensors.splat`) sensors |
| `crazyflow.sim.splat` | Gaussian splat plugin: `attach_splats` and the web-based `SplatViewer` |
| `crazyflow.control` | `Control` enum |
| `crazyflow.control.mellinger` | Mellinger controller data and parameters |
| `crazyflow.envs` | Gymnasium vectorized environments |
| `crazyflow.utils` | Grid utilities and pytree helpers |

Navigate the full generated reference using the sidebar.
