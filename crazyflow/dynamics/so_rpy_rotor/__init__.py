r"""Second-order fitted RPY dynamics with first-order thrust dynamics.

Extends ``so_rpy`` by adding a scalar thrust state \(f_\Sigma\) that captures motor spin-up and
spin-down with a first-order lag. Rotational dynamics remain a fitted second-order linear system
driven by RPY commands. The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``. The
``rotor_vel`` state is the current thrust in Newtons (not motor RPMs), carried as four entries of
which only the first enters the dynamics.

\[
\begin{aligned}
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    m\dot{\mathbf{v}} &= \mathbf{f}_\mathrm{g}
        + \mathbf{R}\,\mathbf{e}_\mathrm{z} (c_\mathrm{acc} + c_\mathrm{f} f_\Sigma), \\
    \dot{f}_\Sigma &= c_\tau (f_{\Sigma,\mathrm{cmd}} - f_\Sigma), \\
    \ddot{\boldsymbol{\Psi}} &=
        \boldsymbol{c}_{\boldsymbol{\Psi},1}\,\boldsymbol{\Psi}
        + \boldsymbol{c}_{\boldsymbol{\Psi},2}\,\dot{\boldsymbol{\Psi}}
        + \boldsymbol{c}_{\boldsymbol{\Psi},3}\,\boldsymbol{\Psi}_\mathrm{cmd},
\end{aligned}
\]

where \(\mathbf{f}_\mathrm{g} = m\mathbf{g}\).

This is the native Euler-angle form. For how the simulation integrates this state in quaternion +
angular velocity coordinates, see [so_rpy][crazyflow.dynamics.so_rpy].

| Variable | Name | Description |
| --- | --- | --- |
| \(\mathbf{p}\) | `pos` | Position in m |
| \(\mathbf{v}\) | `vel` | Velocity in m/s |
| \(\boldsymbol{\Psi} = [\phi,\theta,\psi]^{\top}\) | | Roll, pitch, and yaw in rad, from `quat` |
| \(\dot{\boldsymbol{\Psi}}\) | | Roll, pitch, and yaw rates in rad/s, from `ang_vel` |
| \(f_\Sigma\) | `rotor_vel[0]` | Collective thrust in N |
| \(\boldsymbol{\Psi}_\mathrm{cmd}\) | `cmd[:3]` | Commanded roll, pitch, and yaw in rad |
| \(f_{\Sigma,\mathrm{cmd}}\) | `cmd[3]` | Commanded collective thrust in N |
| \(\mathbf{f}_\mathrm{g}\) | | Gravitational force |
| \(\mathbf{R}\) | | Rotation from body to world frame |
| \(\mathbf{e}_\mathrm{z}\) | | Unit vector in z direction |

| Parameter | Name | Description |
| --- | --- | --- |
| \(m\) | `mass` | Mass in kg |
| \(\mathbf{g}\) | `gravity_vec` | Gravity vector in m/s² |
| \(c_\tau\) | `thrust_dyn_coef` | Thrust dynamics coefficient in 1/s |
| \(c_\mathrm{acc}\) | `acc_coef` | Thrust offset |
| \(c_\mathrm{f}\) | `cmd_f_coef` | Thrust scaling coefficient |
| \(\boldsymbol{c}_{\boldsymbol{\Psi},1}\) | `rpy_coef` | Rotational dynamics coefficients |
| \(\boldsymbol{c}_{\boldsymbol{\Psi},2}\) | `rpy_rates_coef` | Rotational dynamics coefficients |
| \(\boldsymbol{c}_{\boldsymbol{\Psi},3}\) | `cmd_rpy_coef` | Rotational dynamics coefficients |
"""

from crazyflow.dynamics.so_rpy_rotor.dynamics import (
    Params,
    dynamics,
    sim_dynamics,
    symbolic_dynamics,
    symbolic_dynamics_euler,
)

__all__ = ["Params", "dynamics", "sim_dynamics", "symbolic_dynamics", "symbolic_dynamics_euler"]
