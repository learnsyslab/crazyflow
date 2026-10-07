r"""Second-order fitted RPY dynamics (no rotor dynamics).

Rotational dynamics are modelled as a fitted second-order linear system driven by roll, pitch, and
yaw commands. Translational dynamics are driven by the collective thrust command directly, with no
motor spin-up lag. The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``.

\[
\begin{aligned}
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    m\dot{\mathbf{v}} &= \mathbf{f}_\mathrm{g}
        + \mathbf{R}\,\mathbf{e}_\mathrm{z}
          (c_\mathrm{acc} + c_\mathrm{f} f_{\Sigma,\mathrm{cmd}}), \\
    \ddot{\boldsymbol{\Psi}} &=
        \boldsymbol{c}_{\boldsymbol{\Psi},1}\,\boldsymbol{\Psi}
        + \boldsymbol{c}_{\boldsymbol{\Psi},2}\,\dot{\boldsymbol{\Psi}}
        + \boldsymbol{c}_{\boldsymbol{\Psi},3}\,\boldsymbol{\Psi}_\mathrm{cmd},
\end{aligned}
\]

where \(\mathbf{f}_\mathrm{g} = m\mathbf{g}\).

!!! note
    This is the native Euler-angle form, matching
    [symbolic_dynamics_euler][crazyflow.dynamics.so_rpy.symbolic_dynamics_euler]. The simulation
    does not integrate this state directly. It shares the common ``[pos, quat, vel, ang_vel]`` state
    with the other models and advances the orientation from the body angular velocity
    \({}^{\mathcal{B}}\boldsymbol{\omega}\), converting \(\ddot{\boldsymbol{\Psi}} \leftrightarrow
    {}^{\mathcal{B}}\dot{\boldsymbol{\omega}}\) through the kinematic Jacobian at every step.
    Integrating from \({}^{\mathcal{B}}\boldsymbol{\omega}\) rather than \(\dot{\boldsymbol{\Psi}}\)
    makes the discrete trajectory differ slightly from integrating the Euler state directly. The
    difference, however, is negligible at our default frequency of 500 Hz.

| Variable | Name | Description |
| --- | --- | --- |
| \(\mathbf{p}\) | `pos` | Position in m |
| \(\mathbf{v}\) | `vel` | Velocity in m/s |
| \(\boldsymbol{\Psi} = [\phi,\theta,\psi]^{\top}\) | | Roll, pitch, and yaw in rad, from `quat` |
| \(\dot{\boldsymbol{\Psi}}\) | | Roll, pitch, and yaw rates in rad/s, from `ang_vel` |
| \(\boldsymbol{\Psi}_\mathrm{cmd}\) | `cmd[:3]` | Commanded roll, pitch, and yaw in rad |
| \(f_{\Sigma,\mathrm{cmd}}\) | `cmd[3]` | Commanded collective thrust in N |
| \(\mathbf{f}_\mathrm{g}\) | | Gravitational force |
| \(\mathbf{R}\) | | Rotation from body to world frame |
| \(\mathbf{e}_\mathrm{z}\) | | Unit vector in z direction |
| \({}^{\mathcal{B}}(\cdot)\) | | Quantity in the body frame, world frame if unmarked |

| Parameter | Name | Description |
| --- | --- | --- |
| \(m\) | `mass` | Mass in kg |
| \(\mathbf{g}\) | `gravity_vec` | Gravity vector in m/s² |
| \(c_\mathrm{acc}\) | `acc_coef` | Thrust offset |
| \(c_\mathrm{f}\) | `cmd_f_coef` | Thrust scaling coefficient |
| \(\boldsymbol{c}_{\boldsymbol{\Psi},1}\) | `rpy_coef` | Rotational dynamics coefficients |
| \(\boldsymbol{c}_{\boldsymbol{\Psi},2}\) | `rpy_rates_coef` | Rotational dynamics coefficients |
| \(\boldsymbol{c}_{\boldsymbol{\Psi},3}\) | `cmd_rpy_coef` | Rotational dynamics coefficients |
"""

from crazyflow.dynamics.so_rpy.dynamics import (
    Params,
    dynamics,
    sim_dynamics,
    symbolic_dynamics,
    symbolic_dynamics_euler,
)

__all__ = ["Params", "dynamics", "sim_dynamics", "symbolic_dynamics", "symbolic_dynamics_euler"]
