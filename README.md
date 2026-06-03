![Crazyflow Logo](https://github.com/learnsyslab/crazyflow/raw/main/docs/img/logo.png)
--------------------------------------------------------------------------------
<div align="center">

  **Fast, parallelizable simulations of Quadrotor drones with JAX.**

  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
  [![arXiv](https://img.shields.io/badge/arXiv-2606.01478-b31b1b.svg)](https://arxiv.org/abs/2606.01478)
  [![Tests](https://github.com/learnsyslab/crazyflow/actions/workflows/testing.yml/badge.svg)](https://github.com/learnsyslab/crazyflow/actions/workflows/testing.yml)
  [![Ruff](https://github.com/learnsyslab/crazyflow/actions/workflows/ruff.yml/badge.svg)](https://github.com/learnsyslab/crazyflow/actions/workflows/ruff.yml)
  [![Docs](https://github.com/learnsyslab/crazyflow/actions/workflows/docs.yml/badge.svg)](https://learnsyslab.github.io/crazyflow)

</div>

Crazyflow is a research simulator for quadrotors. It runs batched, differentiable simulations on CPU and GPU via JAX, with analytical and abstracted models for the Crazyflie 2.x family.

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=4096, n_drones=1, control=Control.state)
cmd = np.zeros((4096, 1, 13))
cmd[..., 2] = 0.5  # hover at 0.5 m across all worlds

for _ in range(100):
    sim.state_control(cmd)
    sim.step(sim.freq // sim.control_freq)
    sim.render()
```

## Documentation

[learnsyslab.github.io/crazyflow](https://learnsyslab.github.io/crazyflow) — installation, user guide, examples, and API reference.

## Features

- **n\_worlds x n\_drones** — batched over independent environments and multi-drone swarms simultaneously
- **GPU-accelerated** — up to 914 M steps/s on an RTX 4090 (first-principles physics, 262 K worlds)
- **Differentiable** — `jax.grad` works through the full dynamics and control pipeline
- **First-principles models** — physics model using first-principles equations and parameters identified from real-world measurements
- **Abstracted models** — three physics models fitted from real Crazyflie flight data
- **Modular pipelines** — step and reset are tuples of plain JAX functions; insert anything, anywhere
- **MuJoCo integration** — onscreen and offscreen rendering, raycasting, and contact detection via MJX

## Installation

```bash
pip install crazyflow           # CPU
pip install "crazyflow[gpu]"    # GPU (Linux x86-64, CUDA 12)
```

Developer install with editable submodules ([pixi](https://pixi.sh/) required):

```bash
git clone --recurse-submodules git@github.com:learnsyslab/crazyflow.git
cd crazyflow
pixi shell
```

## Performance

First-principles physics, one drone. CPU: AMD Ryzen 9 7950X. GPU: NVIDIA RTX 4090.

| n\_worlds | CPU steps/s | GPU steps/s |
|---|---|---|
| 64 | 3.3 M | 1.2 M |
| 1 024 | 9.2 M | 18.7 M |
| 16 384 | 11.9 M | 257 M |
| 65 536 | 15.6 M | 678 M |
| 262 144 | 12.6 M | 914 M |

Full benchmarks including multi-drone scaling are in the [documentation](https://learnsyslab.github.io/crazyflow).

## Related packages

Crazyflow is built on two companion packages that can also be used independently:

| Package | Description |
|---|---|
| [drone-models](https://github.com/learnsyslab/drone-models) | Drone dynamics models (first-principles and fitted) compatible with NumPy, JAX, and PyTorch. Used by Crazyflow as the physics backend. |
| [drone-controllers](https://github.com/learnsyslab/drone-controllers) | Reference controller implementations including the Mellinger geometric controller. Used by Crazyflow to provide the state and attitude control modes. |

Both are installed automatically as dependencies. For development, they are included as submodules in `submodules/` and installed in editable mode by the pixi environment.

## Citation

```bibtex
@misc{schuck2026crazyflow,
      title={Crazyflow: An Accurate, GPU-Accelerated, Differentiable Drone Simulator in JAX}, 
      author={Martin Schuck and Marcel P. Rath and Yufei Hua and AbhisheK Goudar and SiQi Zhou and Angela P. Schoellig},
      year={2026},
      eprint={2606.01478},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.01478}, 
}
```
