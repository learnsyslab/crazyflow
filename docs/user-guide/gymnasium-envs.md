# Gymnasium Environments

Crazyflow ships a set of [Gymnasium](https://gymnasium.farama.org/) vectorized environments built on top of `Sim`. They follow the standard `VectorEnv` interface and are suitable for training RL agents with frameworks such as Stable Baselines3, CleanRL, or custom JAX trainers.

## Available environments

| Class | Task | Observation | Action |
|---|---|---|---|
| `DroneEnv` | Base class (no reward) | pos, quat, vel, ang_vel | attitude or force/torque |
| `ReachPosEnv` | Reach a target position | pos, quat, vel, ang_vel, target | attitude |
| `ReachVelEnv` | Match a target velocity | vel, ang_vel, target_vel | attitude |
| `LandingEnv` | Land safely | pos, quat, vel, ang_vel | attitude |
| `Figure8Env` | Follow a figure-8 trajectory | pos, quat, vel, ang_vel, phase | attitude |

All environments run `num_envs` parallel instances backed by a single `Sim` with `n_worlds=num_envs`.

## Basic usage

```{ .python notest }
from crazyflow.envs import Figure8Env

env = Figure8Env(num_envs=16, device="cpu")
obs, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()  # random policy for illustration
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Constructor arguments

All environments accept these common arguments:

| Argument | Default | Description |
|---|---|---|
| `num_envs` | 1 | Number of parallel environments |
| `max_episode_time` | 10.0 | Episode length before truncation, seconds |
| `physics` | `Physics.so_rpy` | Physics model |
| `drone_model` | `"cf2x_L250"` | Drone configuration |
| `freq` | 500 | Physics frequency, Hz |
| `device` | `"cpu"` | `"cpu"` or `"gpu"` |
| `reset_randomization` | `None` | Optional `SimData → SimData` function applied at reset |

## Action normalization

`NormalizeActionsWrapper` rescales the action space to `[-1, 1]`, which simplifies policy learning:

```{ .python notest }
from crazyflow.envs import Figure8Env
from crazyflow.envs.norm_actions_wrapper import NormalizeActionsWrapper

env = NormalizeActionsWrapper(Figure8Env(num_envs=32))
obs, info = env.reset()
action = env.action_space.sample()  # in [-1, 1]^4
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

## Reset randomization

Pass a `reset_randomization` callable to vary initial conditions between episodes. The function receives `SimData` and a JAX random key and must return updated `SimData`:

```{ .python notest }
import jax
import jax.numpy as jnp
from crazyflow.envs import ReachPosEnv
from crazyflow.sim.data import SimData

def randomize(data: SimData, key: jax.Array) -> SimData:
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, data.states.pos.shape) * 0.05
    return data.replace(states=data.states.replace(pos=data.states.pos + noise))

env = ReachPosEnv(num_envs=64, reset_randomization=randomize)
```

## Next steps

- [Examples](../examples/index.md) — figure-8 and RL training examples
- [Functional API](functional-api.md) — building fully jittable training loops with `jax.lax.scan`
