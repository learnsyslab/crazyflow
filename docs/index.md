# Crazyflow

<div align="center">
  <img src="img/logo.svg" alt="Crazyflow Logo" width="300"/>
</div>

**Fast, parallelizable simulations of Crazyflie drones with JAX.**

Crazyflow is a research simulator for Crazyflie-style quadrotors that runs millions of independent environments in parallel on CPU or GPU. It is built on JAX, exposes a differentiable dynamics pipeline, and ships identified models for the Crazyflie 2.x family.

---

## Showcase

<!-- VIDEO SLIDER: uncomment and replace src paths once videos are ready.
     Suggested layout mirrors crisp_controllers: a row of 3–4 thumbnail cards
     with autoplay-on-hover, each linking to a full-size video.

<div class="video-grid">

  <div class="video-card">
    <video autoplay loop muted playsinline>
      <source src="videos/hover_demo.mp4" type="video/mp4">
    </video>
    <p>Hover — state control with Mellinger</p>
  </div>

  <div class="video-card">
    <video autoplay loop muted playsinline>
      <source src="videos/swarm.mp4" type="video/mp4">
    </video>
    <p>Swarm — 64 drones, 16 parallel worlds</p>
  </div>

  <div class="video-card">
    <video autoplay loop muted playsinline>
      <source src="videos/racing_blender.mp4" type="video/mp4">
    </video>
    <p>Drone racing — Blender render</p>
  </div>

  <div class="video-card">
    <video autoplay loop muted playsinline>
      <source src="videos/mppi_blender.mp4" type="video/mp4">
    </video>
    <p>MPPI forest navigation — 500K rollout worlds, Blender render</p>
  </div>

</div>
-->

<!-- VIDEO: BPTT training — loss curves + rendered trajectory from policy trained end-to-end -->
<!-- Suggested: embed as a plain <video> tag below once the clip is ready:

<video width="100%" autoplay loop muted playsinline>
  <source src="videos/bptt_training.mp4" type="video/mp4">
</video>
-->

---

## Supported drones

<!-- DRONE GRID: replace the placeholder image paths once renders are available.
     The list of available models comes from drone_models.available_drones.

<div class="drone-grid" markdown>

| Model | Description |
|-------|-------------|
| ![cf2x_L250](img/drones/cf2x_L250.png){ width=120 } | **cf2x_L250** — Crazyflie 2.x, L250 propellers |
| ![cf2x_T350](img/drones/cf2x_T350.png){ width=120 } | **cf2x_T350** — Crazyflie 2.x, T350 propellers |

</div>
-->

<!-- Placeholder until drone renders are available: -->
All models come from the [drone-models](https://learnsyslab.github.io/drone-models/) library. Available configurations: `cf2x_L250`, `cf2x_T350`, `cf21B_500`, and any model returned by `drone_models.available_drones()`.

---

## Performance

<!-- Benchmark data sources:
     crazyflow          commit 29a321149a04b4580bc1010c04f25e7f48d0ac40
     crazyflow_experiments commit 6b65eeedefe32690f1e5ca7818d62439314f0de5
-->

Throughput for one drone across parallel worlds, first-principles physics. CPU: AMD Ryzen 9 7950X. GPU: NVIDIA RTX 4090.

```vegalite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "width": "container",
  "height": 300,
  "config": {"view": {"stroke": "transparent"}},
  "data": {
    "values": [
      {"nw":1,"dev":"CPU","sps":403294},{"nw":4,"dev":"CPU","sps":884432},
      {"nw":16,"dev":"CPU","sps":1202200},{"nw":64,"dev":"CPU","sps":3309800},
      {"nw":256,"dev":"CPU","sps":6656300},{"nw":1024,"dev":"CPU","sps":9214400},
      {"nw":4096,"dev":"CPU","sps":8865400},{"nw":16384,"dev":"CPU","sps":11898000},
      {"nw":65536,"dev":"CPU","sps":15609000},{"nw":262144,"dev":"CPU","sps":12569000},
      {"nw":1048576,"dev":"CPU","sps":8554100},
      {"nw":1,"dev":"GPU","sps":21494},{"nw":4,"dev":"GPU","sps":70557},
      {"nw":16,"dev":"GPU","sps":253727},{"nw":64,"dev":"GPU","sps":1168000},
      {"nw":256,"dev":"GPU","sps":4095700},{"nw":1024,"dev":"GPU","sps":18697000},
      {"nw":4096,"dev":"GPU","sps":65107000},{"nw":16384,"dev":"GPU","sps":257190000},
      {"nw":65536,"dev":"GPU","sps":678220000},{"nw":262144,"dev":"GPU","sps":913980000},
      {"nw":1048576,"dev":"GPU","sps":699520000}
    ]
  },
  "mark": {"type": "line", "point": {"filled": true, "size": 40}},
  "encoding": {
    "x": {
      "field": "nw", "type": "quantitative",
      "scale": {"type": "log", "base": 2},
      "axis": {
        "title": "n_worlds",
        "tickCount": 6,
        "gridOpacity": 0.3,
        "labelExpr": "'2^' + round(log(datum.value)/log(2))"
      }
    },
    "y": {
      "field": "sps", "type": "quantitative",
      "scale": {"type": "log"},
      "axis": {"title": "Steps / second", "tickCount": 5, "gridOpacity": 0.3, "format": ".2s"}
    },
    "color": {
      "field": "dev", "type": "nominal",
      "scale": {"domain": ["CPU","GPU"], "range": ["#2196F3","#4CAF50"]},
      "legend": {"title": null}
    }
  }
}
```

GPU throughput across `n_worlds` and `n_drones` (RTX 4090). Empty cells exceed available GPU memory. Color encodes steps per second on a log scale.

```vegalite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "width": "container",
  "height": 300,
  "config": {"view": {"stroke": "transparent"}},
  "data": {
    "values": [
      {"nw":"2^0","nd":"2^0","sps":21494},{"nw":"2^2","nd":"2^0","sps":70557},{"nw":"2^4","nd":"2^0","sps":253727},{"nw":"2^6","nd":"2^0","sps":1168000},{"nw":"2^8","nd":"2^0","sps":4095700},{"nw":"2^10","nd":"2^0","sps":18697000},{"nw":"2^12","nd":"2^0","sps":65107000},{"nw":"2^14","nd":"2^0","sps":257190000},{"nw":"2^16","nd":"2^0","sps":678220000},{"nw":"2^18","nd":"2^0","sps":913980000},{"nw":"2^20","nd":"2^0","sps":699520000},
      {"nw":"2^0","nd":"2^2","sps":18279},{"nw":"2^2","nd":"2^2","sps":72522},{"nw":"2^4","nd":"2^2","sps":290679},{"nw":"2^6","nd":"2^2","sps":1161400},{"nw":"2^8","nd":"2^2","sps":4609300},{"nw":"2^10","nd":"2^2","sps":16042000},{"nw":"2^12","nd":"2^2","sps":51070000},{"nw":"2^14","nd":"2^2","sps":155390000},{"nw":"2^16","nd":"2^2","sps":184350000},{"nw":"2^18","nd":"2^2","sps":140930000},{"nw":"2^20","nd":"2^2","sps":97411000},
      {"nw":"2^0","nd":"2^4","sps":16060},{"nw":"2^2","nd":"2^4","sps":72711},{"nw":"2^4","nd":"2^4","sps":290997},{"nw":"2^6","nd":"2^4","sps":1161900},{"nw":"2^8","nd":"2^4","sps":4016500},{"nw":"2^10","nd":"2^4","sps":12750000},{"nw":"2^12","nd":"2^4","sps":39799000},{"nw":"2^14","nd":"2^4","sps":46391000},{"nw":"2^16","nd":"2^4","sps":35329000},{"nw":"2^18","nd":"2^4","sps":24479000},
      {"nw":"2^0","nd":"2^6","sps":18249},{"nw":"2^2","nd":"2^6","sps":72484},{"nw":"2^4","nd":"2^6","sps":290029},{"nw":"2^6","nd":"2^6","sps":1009200},{"nw":"2^8","nd":"2^6","sps":3205100},{"nw":"2^10","nd":"2^6","sps":9851600},{"nw":"2^12","nd":"2^6","sps":11530000},{"nw":"2^14","nd":"2^6","sps":8824300},{"nw":"2^16","nd":"2^6","sps":6112800},
      {"nw":"2^0","nd":"2^8","sps":18193},{"nw":"2^2","nd":"2^8","sps":72629},{"nw":"2^4","nd":"2^8","sps":253004},{"nw":"2^6","nd":"2^8","sps":798453},{"nw":"2^8","nd":"2^8","sps":2472200},{"nw":"2^10","nd":"2^8","sps":2874800},{"nw":"2^12","nd":"2^8","sps":2204800},{"nw":"2^14","nd":"2^8","sps":1536800},
      {"nw":"2^0","nd":"2^10","sps":18268},{"nw":"2^2","nd":"2^10","sps":63107},{"nw":"2^4","nd":"2^10","sps":200804},{"nw":"2^6","nd":"2^10","sps":609339},{"nw":"2^8","nd":"2^10","sps":718658},{"nw":"2^10","nd":"2^10","sps":553150},{"nw":"2^12","nd":"2^10","sps":381016},
      {"nw":"2^0","nd":"2^12","sps":15873},{"nw":"2^2","nd":"2^12","sps":49898},{"nw":"2^4","nd":"2^12","sps":154616},{"nw":"2^6","nd":"2^12","sps":180297},{"nw":"2^8","nd":"2^12","sps":137266},{"nw":"2^10","nd":"2^12","sps":95417}
    ]
  },
  "mark": "rect",
  "encoding": {
    "x": {
      "field": "nw", "type": "ordinal",
      "sort": ["2^0","2^2","2^4","2^6","2^8","2^10","2^12","2^14","2^16","2^18","2^20"],
      "axis": {"title": "n_worlds", "labelAngle": 0, "grid": false}
    },
    "y": {
      "field": "nd", "type": "ordinal",
      "sort": ["2^12","2^10","2^8","2^6","2^4","2^2","2^0"],
      "axis": {"title": "n_drones", "grid": false}
    },
    "color": {
      "field": "sps", "type": "quantitative",
      "scale": {"type": "log", "range": ["#00ffff", "#7f00ff", "#ff00ff"]},
      "legend": {"title": "Steps / s", "format": ".2s", "gradientLength": 280}
    }
  }
}
```

*Numbers are illustrative placeholders and will be replaced with measured benchmarks before release.*

---

## Why Crazyflow

Most simulators offer either vectorized environments for RL training or multi-drone swarm simulation — rarely both, and rarely with accurate onboard flight dynamics for every agent. Crazyflow is built around both simultaneously. The entire simulator is structured around an `n_worlds × n_drones` batch dimension: `n_worlds` gives you massively parallel independent environments, and `n_drones` gives you full swarm simulation inside each one, each drone running its own accurate, identified flight model and control stack. Scaling to millions of parallel instances requires no code changes.

Simulating the full Crazyflie firmware stack with GPU acceleration and differentiability is not possible with existing tools, so Crazyflow reimplements the entire dynamics and control stack in JAX. This gives accelerated, fully batchable simulation that runs on CPU and GPU without modification. Differentiability comes as a direct consequence: `jax.grad` works through physics, control, and integration without any manual gradient derivations, enabling gradient-based policy optimization, system identification, and sensitivity analysis out of the box.

To make research possible rather than just evaluation, the simulator is designed to be fully open to modification. The step and reset pipelines are plain tuples of JAX functions. There are no fixed hooks or plugin interfaces — you splice in your own dynamics, disturbances, randomization, or reward shaping at any point, and the JIT compiler fuses everything into a single kernel.

For perception and collision, Crazyflow integrates MuJoCo and MJX. GUI rendering uses the MuJoCo viewer directly. Depth sensing, raycasting, and contact detection run through MJX, which keeps them batchable over worlds and compatible with JAX transformations.

## Quick install

```bash
pip install crazyflow
```

See [Installation](get-started/installation.md) for GPU, developer, and from-source options.

## Minimal example

```python
import numpy as np
from crazyflow.sim import Sim
from crazyflow.control import Control

sim = Sim(n_worlds=1, n_drones=1, control=Control.state)
sim.reset()

# State command: [x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate]
cmd = np.zeros((1, 1, 13), dtype=np.float32)
cmd[0, 0, 2] = 0.5  # hover at 0.5 m

sim.state_control(cmd)
sim.step(sim.freq // sim.control_freq)

pos = sim.data.states.pos[0, 0]  # shape (3,) — position of world 0, drone 0
```

## Where to go next

- [Quick Start](get-started/quick-start.md) — step-by-step walkthrough of the object-oriented API
- [Functional API](user-guide/functional-api.md) — JIT compilation, autodiff, and `jax.lax.scan` rollouts
- [Examples](examples/index.md) — runnable scripts covering hover, gradients, batched simulation, and more
- [API Reference](api/index.md) — full Python API
