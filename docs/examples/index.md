# Examples

These examples build on each other — each one introduces one new concept on top of the previous. Start from the top if you're new, or jump to whichever section covers what you need.

---

## Hover

A single drone commanded to hold a fixed height using state control. This is the minimal end-to-end loop: create a `Sim`, reset it, apply a state command, and step forward.

```{ .python notest }
--8<-- "examples/hover.py"
```

```bash
python examples/hover.py
```

---

## Attitude control

Commanding roll, pitch, yaw, and collective thrust directly. This level bypasses the Mellinger position loop and is typical for RL agents that output attitude targets.

```{ .python notest }
--8<-- "examples/attitude.py"
```

---

## Gradient descent through dynamics

Because the simulator is built entirely from JAX operations, `jax.grad` can differentiate through it. Starting the drone above the target height keeps it away from the floor, so the floor-clipping stage never fires and gradients flow freely through the entire trajectory.

```{ .python notest }
--8<-- "examples/gradient.py"
```

---

## Domain randomization

Varying physical parameters per world at reset. Each world gets a slightly different mass, so identical commands produce diverging trajectories.

```{ .python notest }
--8<-- "examples/randomize.py"
```

---

## Disturbance injection

Inserting a random external force and torque into the step pipeline. The disturbance fires on every physics tick, so the drone fights wind-like perturbations.

```{ .python notest }
--8<-- "examples/disturbance.py"
```

---

## Cameras and RGBD

Offscreen rendering returns RGB and depth images on every frame. The FPV camera (`fpv_cam`) is attached to the drone and moves with it.

```{ .python notest }
--8<-- "examples/cameras.py"
```

```bash
python examples/cameras.py
```

---

## LED deck and materials

`change_material` updates the RGBA colour and emission of any named material on any subset of drones at runtime.

```{ .python notest }
--8<-- "examples/led_deck.py"
```

```bash
python examples/led_deck.py
```

---

## Contact queries

The default collision geometry is a sphere around the drone frame. `use_box_collision` replaces it with a tighter oriented box, useful for narrow-gap flight and accurate contact debugging.

```{ .python notest }
--8<-- "examples/contacts.py"
```

---

## Raycasting and depth sensing

`render_depth` fires rays from a camera and returns per-pixel distances. This is faster than full RGB rendering and useful for obstacle sensing or depth-based controllers.

```{ .python notest }
--8<-- "examples/raycasting.py"
```

```bash
python examples/raycasting.py
```

---

## Gymnasium environment

Evaluating a random policy in the figure-8 environment. The env wraps `Sim` behind the standard Gymnasium `VectorEnv` interface.

```{ .python notest }
--8<-- "examples/figure8.py"
```
