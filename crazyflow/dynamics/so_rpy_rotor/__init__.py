r"""Second-order fitted RPY dynamics with first-order thrust dynamics.

Extends ``so_rpy`` by adding a scalar thrust state \(f_\Sigma\) that captures motor spin-up and
spin-down with a first-order lag. Rotational dynamics remain a fitted second-order linear system
driven by RPY commands. The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``. The
``rotor_vel`` state is the current thrust in Newtons (not motor RPMs), carried as four entries of
which only the first enters the dynamics.

\[
\begin{aligned}
    \dot{f}_\Sigma &= c_\tau (f_{\Sigma,\mathrm{cmd}} - f_\Sigma), \\
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    m\dot{\mathbf{v}} &= m\mathbf{g}
        + (c_\mathrm{acc} + c_\mathrm{f} f_\Sigma)\,\mathbf{R}\,\mathbf{e}_\mathrm{z}, \\
    \ddot{\boldsymbol{\Psi}} &=
        \boldsymbol{c}_{\boldsymbol{\Psi},1}\,\boldsymbol{\Psi}
        + \boldsymbol{c}_{\boldsymbol{\Psi},2}\,\dot{\boldsymbol{\Psi}}
        + \boldsymbol{c}_{\boldsymbol{\Psi},3}\,\boldsymbol{\Psi}_\mathrm{cmd},
\end{aligned}
\]

where \(f_\Sigma\) is the collective thrust, \(\mathbf{p}\) and \(\mathbf{v}\) are the position and
velocity, \(m\) is the mass, \(\mathbf{g}\) is the gravity vector, \(\mathbf{e}_\mathrm{z}\) is the
unit vector in z direction, \(\mathbf{R} =
{}^{\mathcal{I}}\mathbf{R}_{\mathcal{B}}(\boldsymbol{\Psi})\) is the rotation from body to world
frame, \(\boldsymbol{\Psi} = [\phi,\theta,\psi]^{\top}\) holds the roll, pitch, and yaw angles with
rates \(\dot{\boldsymbol{\Psi}}\), and \(f_{\Sigma,\mathrm{cmd}}\) and
\(\boldsymbol{\Psi}_\mathrm{cmd}\) are the commanded collective thrust and attitude. The thrust
dynamics coefficient \(c_\tau\), the thrust offset \(c_\mathrm{acc}\), the thrust scaling
coefficient \(c_\mathrm{f}\), and the rotational coefficients
\(\boldsymbol{c}_{\boldsymbol{\Psi},1}\), \(\boldsymbol{c}_{\boldsymbol{\Psi},2}\), and
\(\boldsymbol{c}_{\boldsymbol{\Psi},3}\) are identified from flight data.

This is the native Euler-angle form. For how the simulation integrates this state in quaternion +
angular velocity coordinates, see [so_rpy][crazyflow.dynamics.so_rpy].
"""

from crazyflow.dynamics.so_rpy_rotor.dynamics import (
    Params,
    dynamics,
    sim_dynamics,
    symbolic_dynamics,
    symbolic_dynamics_euler,
)

__all__ = ["Params", "dynamics", "sim_dynamics", "symbolic_dynamics", "symbolic_dynamics_euler"]
