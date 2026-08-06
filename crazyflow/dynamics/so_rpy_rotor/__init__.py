r"""Second-order fitted RPY dynamics with first-order thrust dynamics.

Extends ``so_rpy`` by adding a scalar thrust state \(F\) that captures motor spin-up and spin-down
with a first-order lag. Rotational dynamics remain a fitted second-order linear system driven by RPY
commands. The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``. The ``rotor_vel``
state is the current thrust in Newtons (not motor RPMs), carried as four entries of which only
the first enters the dynamics.

\[
\begin{aligned}
    \dot{F} &= \frac{1}{\tau}(F_{\mathrm{cmd}} - F), \\
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    m\dot{\mathbf{v}} &= m\mathbf{g}
        + (c_{\mathrm{acc}} + c_f F)\,R\,\mathbf{e}_z, \\
    \ddot{\boldsymbol{\psi}} &=
        c_{\psi}\,\boldsymbol{\psi}
        + c_{\dot{\psi}}\,\dot{\boldsymbol{\psi}}
        + c_u\,\mathbf{u}_{\mathrm{rpy}},
\end{aligned}
\]

where \(\tau\) is the thrust time constant, \(\boldsymbol{\psi} = [\phi,\theta,\psi]^{\top}\) are
the roll/pitch/yaw angles with rates \(\dot{\boldsymbol{\psi}}\), and
\(R = {}^{\mathcal{I}}R_{\mathcal{B}}(\boldsymbol{\psi})\) is the rotation from body to world frame.

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
