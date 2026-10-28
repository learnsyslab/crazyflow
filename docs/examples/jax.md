# JAX examples

Applying JAX transformations to the simulator: differentiating through the dynamics with `jax.grad`, distributing the simulation across devices, and caching compiled functions. See [Sharding](../user-guide/sharding.md) for the details on multiple devices.

---

## Gradient descent through dynamics

--8<-- example "examples/jax/gradient.py" media="output"

---

## Gradients and command clipping

--8<-- example "examples/jax/gradient_clipping.py" media="plot"

---

## Sharding across devices

--8<-- example "examples/jax/sharding.py" media="output"

---

## Persistent compilation cache

--8<-- example "examples/jax/cache.py" media="output" runs="2"
