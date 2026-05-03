# User Guide

In-depth documentation for every part of the simulator.

- [Simulator Overview](sim-overview.md) — `SimData` layout, worlds, drones, and the data convention
- [Object-Oriented API](oo-api.md) — `Sim` class, control methods, rendering, and reset
- [Functional API](functional-api.md) — purely functional interface for JAX transformations
- [Physics Models](physics-models.md) — first-principles vs. fitted models, when to use each
- [Control Modes](control-modes.md) — state, attitude, force/torque, and rotor velocity control
- [Pipelines](pipelines.md) — composable step and reset pipelines, randomization, and disturbances
- [Visualization](visualization.md) — rendering modes, cameras, raycasting, and materials
- [MuJoCo Integration](mujoco.md) — MJCF scene construction, adding objects, and sync internals
- [Gymnasium Environments](gymnasium-envs.md) — vectorized environments for RL training
