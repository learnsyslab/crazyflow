r"""Second-order fitted RPY dynamics model with first-order thrust dynamics.

Extends ``so_rpy`` by adding a scalar thrust state \(F\) that captures motor
spin-up and spin-down with a first-order lag. Rotational dynamics remain a
fitted second-order linear system driven by RPY commands.
The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``.
The ``rotor_vel`` state is the current thrust in Newtons (not motor RPMs).

\[
\begin{aligned}
    \dot{F} &= \frac{1}{\tau}(F_{\mathrm{cmd}} - F), \\
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    \dot{\mathbf{q}} &= \tfrac{1}{2}
        \begin{bmatrix}0 \\ {}^{\mathcal{B}}\boldsymbol{\omega}\end{bmatrix}
        \otimes \mathbf{q}, \\
    m\dot{\mathbf{v}} &= m\mathbf{g}
        + (c_{\mathrm{acc}} + c_f F)\,R\,\mathbf{e}_z, \\
    \ddot{\boldsymbol{\psi}} &=
        c_{\psi}\,\boldsymbol{\psi}
        + c_{\dot{\psi}}\,\dot{\boldsymbol{\psi}}
        + c_u\,\mathbf{u}_{\mathrm{rpy}},
\end{aligned}
\]

where \(\tau\) is the thrust time constant, \(\boldsymbol{\psi} = [\phi,\theta,\psi]^{\top}\)
are roll/pitch/yaw angles extracted from \(\mathbf{q}\), and
\(R = {}^{\mathcal{I}}R_{\mathcal{B}}(\mathbf{q})\) is the rotation from body to world frame.
"""

from crazyflow.dynamics.so_rpy_rotor.model import (
    Params,
    dynamics,
    sim_dynamics,
    symbolic_dynamics,
    symbolic_dynamics_euler,
)

__all__ = ["Params", "dynamics", "sim_dynamics", "symbolic_dynamics", "symbolic_dynamics_euler"]
