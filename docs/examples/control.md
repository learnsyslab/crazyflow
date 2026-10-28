# Control examples

Flying drones with the different control modes, from state commands down to force and torque, comparing the dynamics models, and model-based control. See [Control Modes](../user-guide/control/index.md) for the controller behind each mode.

---

## Hover

--8<-- example "examples/control/hover.py" media="video" cam_config="distance=0.7 elevation=-10 lookat=0.07,0.07,0.17"

---

## Changing the state

--8<-- example "examples/control/change_pos.py" media="video" cam_config="distance=0.9 azimuth=135 elevation=-15 lookat=0.25,0.25,0.3"

---

## Spirals

--8<-- example "examples/control/spiral.py" media="video" camera="track_cam:0"

---

## Attitude control

--8<-- example "examples/control/attitude.py" media="video" camera="track_cam:0"

---

## Body rate control

--8<-- example "examples/control/body_rate.py" media="video" camera="track_cam:0"

---

## Force and torque control

--8<-- example "examples/control/force_torque.py" media="video" cam_config="distance=0.7 elevation=-15 lookat=0,0,0.06"

---

## Comparing integrators

--8<-- example "examples/control/integrators.py" media="plot,output"

---

## Comparing dynamics models

--8<-- example "examples/control/dynamics.py" media="plot,output"

---

## Symbolic dynamics

--8<-- example "examples/control/symbolic.py"

---

## Sampling-based MPC

--8<-- example "examples/control/sampling.py" media="video"
