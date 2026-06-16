r"""Full rigid-body dynamics for a quadrotor.

This package implements Newton-Euler dynamics based on physical constants —
mass, inertia, motor thrust and torque curves, arm length, and drag
coefficients. The command interface is four motor angular velocities in RPM.
No data fitting is required; all parameters are measurable physical quantities.

Motor forces and torques are quadratic polynomials in RPM:

\[
    f_{p,i} = k_0 + k_1 \Omega_i + k_2 \Omega_i^2, \qquad
    \tau_{p,i} = m_0 + m_1 \Omega_i + m_2 \Omega_i^2.
\]

When rotor dynamics are modelled, each motor RPM evolves as:

\[
    \dot{\Omega}_i = \begin{cases}
        c_1 (\Omega_{\mathrm{cmd},i} - \Omega_i)
        + c_2 (\Omega_{\mathrm{cmd},i}^2 - \Omega_i^2)
            & \Omega_{\mathrm{cmd},i} \geq \Omega_i \\[4pt]
        c_3 (\Omega_{\mathrm{cmd},i} - \Omega_i)
        + c_4 (\Omega_{\mathrm{cmd},i}^2 - \Omega_i^2)
            & \Omega_{\mathrm{cmd},i} < \Omega_i
    \end{cases}
\]

The rigid-body equations of motion are:

\[
\begin{aligned}
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    \dot{\mathbf{q}} &= \tfrac{1}{2}
        \begin{bmatrix}0 \\ {}^{\mathcal{B}}\boldsymbol{\omega}\end{bmatrix}
        \otimes \mathbf{q}, \\
    m\dot{\mathbf{v}} &= m\mathbf{g}
        + R\,{}^{\mathcal{B}}\mathbf{f}_t
        + R\,{}^{\mathcal{B}}\mathbf{f}_a, \\
    \mathbf{J}\,{}^{\mathcal{B}}\dot{\boldsymbol{\omega}} &=
        {}^{\mathcal{B}}\mathbf{t}_\Sigma
        - {}^{\mathcal{B}}\boldsymbol{\omega}
          \times \mathbf{J}\,{}^{\mathcal{B}}\boldsymbol{\omega},
\end{aligned}
\]

where \(R = {}^{\mathcal{I}}R_{\mathcal{B}}(\mathbf{q})\) is the rotation from body to world
frame, and the forces and torques are:

\[
\begin{aligned}
    {}^{\mathcal{B}}\mathbf{f}_t &=
        \mathbf{e}_z \textstyle\sum_{i=1}^{4} f_{p,i}, \\
    {}^{\mathcal{B}}\mathbf{f}_a &= D_b\,R^{\top}\mathbf{v}, \\
    {}^{\mathcal{B}}\mathbf{t}_\Sigma &=
        {}^{\mathcal{B}}\mathbf{t}_t
        + {}^{\mathcal{B}}\mathbf{t}_d
        + {}^{\mathcal{B}}\mathbf{t}_i,
\end{aligned}
\]

with:

\[
\begin{aligned}
    {}^{\mathcal{B}}\mathbf{t}_t &=
        \frac{l}{\sqrt{2}}
        \begin{bmatrix}1&0&0\\0&1&0\\0&0&0\end{bmatrix}
        M\,\mathbf{f}_p, \\
    {}^{\mathcal{B}}\mathbf{t}_d &=
        \begin{bmatrix}0&0&0\\0&0&0\\0&0&1\end{bmatrix}
        M\,\boldsymbol{\tau}_p, \\
    {}^{\mathcal{B}}\mathbf{t}_i &= J_p
        \begin{bmatrix}
            -{}^{\mathcal{B}}\omega_y\;\mathbf{m}_z^{\top}\boldsymbol{\Omega} \\
            -{}^{\mathcal{B}}\omega_x\;\mathbf{m}_z^{\top}\boldsymbol{\Omega} \\
            \mathbf{m}_z^{\top}\dot{\boldsymbol{\Omega}}
        \end{bmatrix},
\end{aligned}
\]

where \(D_b\) is the body-frame drag matrix, \(l\) is the motor arm length,
\(J_p\) is the propeller moment of inertia, \(M\) is the \(3\times 4\) mixing
matrix, and \(\mathbf{m}_z\) is its last row.
"""

from crazyflow.dynamics.first_principles.model import (
    Params,
    dynamics,
    sim_dynamics,
    symbolic_dynamics,
)

__all__ = ["dynamics", "symbolic_dynamics", "sim_dynamics", "Params"]
