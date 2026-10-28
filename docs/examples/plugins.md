# Plugin examples

Extending the step and reset pipelines with custom stages for domain randomization, disturbances, action delays, aerodynamic effects, and additional outputs. See [Pipelines](../user-guide/pipelines.md) for how stages are composed.

---

## Domain randomization

--8<-- example "examples/plugins/randomize.py" media="video" cam_config="distance=1.7 elevation=-25 lookat=0,0,0.2"

---

## Disturbance injection

--8<-- example "examples/plugins/disturbance.py" media="video,plot" cam_config="distance=1.0 azimuth=135 elevation=-15 lookat=0.1,0.1,0.15"

---

## Action delay

--8<-- example "examples/plugins/action_delay.py" media="video,output"

---

## Downwash

--8<-- example "examples/plugins/downwash.py" media="video,plot" cam_config="distance=1.5 elevation=-5 lookat=0,0,0.85"

---

## Ground effect

--8<-- example "examples/plugins/ground_effect.py" media="plot"

---

## State derivatives

--8<-- example "examples/plugins/derivatives.py" media="plot,output"

---

## IMU

--8<-- example "examples/plugins/imu.py" media="plot,output"
