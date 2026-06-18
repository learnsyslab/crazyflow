r"""Second-order fitted RPY dynamics with thrust dynamics and linear drag.

Extends ``so_rpy_rotor`` by adding a body-frame linear drag term to the translational dynamics.
Rotational dynamics remain a fitted second-order linear system, and thrust spin-up uses a
first-order lag. The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``. The 
``rotor_vel`` state is the current thrust in Newtons (not motor RPMs).

\[
\begin{aligned}
    \dot{F} &= \frac{1}{\tau}(F_{\mathrm{cmd}} - F), \\
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    \dot{\mathbf{q}} &= \tfrac{1}{2}
        \mathbf{q} \otimes \begin{bmatrix} {}^{\mathcal{B}}\boldsymbol{\omega}\\0 \end{bmatrix}, \\
    m\dot{\mathbf{v}} &= m\mathbf{g}
        + (c_{\mathrm{acc}} + c_f F)\,R\,\mathbf{e}_z
        + R\,D_b\,R^{\top}\mathbf{v}, \\
    \ddot{\boldsymbol{\psi}} &=
        c_{\psi}\,\boldsymbol{\psi}
        + c_{\dot{\psi}}\,\dot{\boldsymbol{\psi}}
        + c_u\,\mathbf{u}_{\mathrm{rpy}},
\end{aligned}
\]

where \(\tau\) is the thrust time constant, \(\boldsymbol{\psi} = [\phi,\theta,\psi]^{\top}\) are
roll/pitch/yaw angles extracted from \(\mathbf{q}\),
\(R = {}^{\mathcal{I}}R_{\mathcal{B}}(\mathbf{q})\) is the rotation from body to world frame, and
\(D_b\) is the diagonal body-frame aerodynamic drag matrix.
"""

from crazyflow.dynamics.so_rpy_rotor_drag.dynamics import (
    Params,
    dynamics,
    sim_dynamics,
    symbolic_dynamics,
    symbolic_dynamics_euler,
)

__all__ = ["Params", "dynamics", "sim_dynamics", "symbolic_dynamics", "symbolic_dynamics_euler"]
