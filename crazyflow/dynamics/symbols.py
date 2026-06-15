"""Symbols used in the symbolic drone models."""

import casadi as cs

from crazyflow.dynamics.utils import rotation

# States
px, py, pz = cs.MX.sym("px"), cs.MX.sym("py"), cs.MX.sym("pz")
pos = cs.vertcat(px, py, pz)  # Position
"""Symbolic drone position.

Can be used to define symbolic CasADi expressions that are passed to model-based optimizers such as
Acados.

:meta hide-value:
"""
# States
qw, qx, qy, qz = cs.MX.sym("qw"), cs.MX.sym("qx"), cs.MX.sym("qy"), cs.MX.sym("qz")
quat = cs.vertcat(qx, qy, qz, qw)  # Quaternions
rot = rotation.cs_quat2matrix(quat)  # Rotation matrix from body to world frame
vx, vy, vz = cs.MX.sym("vx"), cs.MX.sym("vy"), cs.MX.sym("vz")
vel = cs.vertcat(vx, vy, vz)  # Velocity
wx, wy, wz = cs.MX.sym("wx"), cs.MX.sym("wy"), cs.MX.sym("wz")
ang_vel = cs.vertcat(wx, wy, wz)  # Angular velocity
w1, w2, w3, w4 = cs.MX.sym("w1"), cs.MX.sym("w2"), cs.MX.sym("w3"), cs.MX.sym("w4")
rotor_vel = cs.vertcat(w1, w2, w3, w4)  # Motor thrust
dfx, dfy, dfz = cs.MX.sym("dfx"), cs.MX.sym("dfy"), cs.MX.sym("dfz")
dist_f = cs.vertcat(dfx, dfy, dfz)  # Disturbance forces
dtx, dty, dtz = cs.MX.sym("dtx"), cs.MX.sym("dty"), cs.MX.sym("dtz")
dist_t = cs.vertcat(dtx, dty, dtz)  # Disturbance torques

# Inputs
cmd_w1, cmd_w2, cmd_w3, cmd_w4 = (
    cs.MX.sym("cmd_w1"),
    cs.MX.sym("cmd_w2"),
    cs.MX.sym("cmd_w3"),
    cs.MX.sym("cmd_w4"),
)
cmd_rotor_vel = cs.vertcat(cmd_w1, cmd_w2, cmd_w3, cmd_w4)
cmd_force = cs.vertcat(cmd_w1, cmd_w2, cmd_w3, cmd_w4)
cmd_roll, cmd_pitch, cmd_yaw = (cs.MX.sym("cmd_roll"), cs.MX.sym("cmd_pitch"), cs.MX.sym("cmd_yaw"))
cmd_thrust = cs.MX.sym("cmd_thrust")
cmd_rpyt = cs.vertcat(cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust)

# Special states for the so_rpy models
roll, pitch, yaw = cs.MX.sym("roll"), cs.MX.sym("pitch"), cs.MX.sym("yaw")
rpy = cs.vertcat(roll, pitch, yaw)
droll, dpitch, dyaw = cs.MX.sym("droll"), cs.MX.sym("dpitch"), cs.MX.sym("dyaw")
drpy = cs.vertcat(droll, dpitch, dyaw)
