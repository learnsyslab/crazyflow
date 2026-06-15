r"""Second-order fitted RPY dynamics model (no rotor dynamics).

Rotational dynamics are modelled as a fitted second-order linear system driven
by roll, pitch, and yaw commands. Translational dynamics are driven by the
collective thrust command directly, with no motor spin-up lag.
The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``.

\[
\begin{aligned}
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    \dot{\mathbf{q}} &= \tfrac{1}{2}
        \begin{bmatrix}0 \\ {}^{\mathcal{B}}\boldsymbol{\omega}\end{bmatrix}
        \otimes \mathbf{q}, \\
    m\dot{\mathbf{v}} &= m\mathbf{g}
        + (c_{\mathrm{acc}} + c_f F_{\mathrm{cmd}})\,R\,\mathbf{e}_z, \\
    \ddot{\boldsymbol{\psi}} &=
        c_{\psi}\,\boldsymbol{\psi}
        + c_{\dot{\psi}}\,\dot{\boldsymbol{\psi}}
        + c_u\,\mathbf{u}_{\mathrm{rpy}},
\end{aligned}
\]

where \(\boldsymbol{\psi} = [\phi,\theta,\psi]^{\top}\) are roll/pitch/yaw
angles extracted from \(\mathbf{q}\), \(R = {}^{\mathcal{I}}R_{\mathcal{B}}(\mathbf{q})\)
is the rotation from body to world frame, and \({}^{\mathcal{B}}\boldsymbol{\omega}\)
is recovered from \(\ddot{\boldsymbol{\psi}}\) via the kinematic Jacobian.
The coefficients \(c_{\psi}\), \(c_{\dot{\psi}}\), \(c_u\) are identified from
flight data.
"""

from crazyflow.dynamics.so_rpy.model import (
    Params,
    dynamics,
    sim_dynamics,
    symbolic_dynamics,
    symbolic_dynamics_euler,
)

__all__ = ["Params", "dynamics", "sim_dynamics", "symbolic_dynamics", "symbolic_dynamics_euler"]
