r"""Full rigid-body dynamics for a quadrotor.

This package implements Newton-Euler dynamics based on physical parameters: mass, inertia, motor
thrust and torque curves, arm length, and drag coefficients. The command interface is four motor
angular velocities in RPM. Mass and arm length are measured directly and the propeller inertia is
taken from CAD data. The thrust and torque curves are fitted to load cell data, and the inertia and
drag coefficients are identified from flight data.

Motor forces and torques are quadratic polynomials in RPM:

\[
    f_{\mathrm{m},i} = k_0 + k_1 \Omega_i + k_2 \Omega_i^2, \qquad
    t_{\mathrm{m},i} = m_0 + m_1 \Omega_i + m_2 \Omega_i^2,
\]

where \(\Omega_i\) is the rotor speed of motor \(i\) in RPM, \(f_{\mathrm{m},i}\) and
\(t_{\mathrm{m},i}\) are its thrust and drag torque, and \(k_0, k_1, k_2\) and \(m_0, m_1, m_2\) are
the thrust and torque curve coefficients.

When rotor dynamics are modelled, each motor RPM evolves as:

\[
    \dot{\Omega}_i = \begin{cases}
        \hat{c}_\mathrm{v} (\Omega_{\mathrm{cmd},i} - \Omega_i)
        + \hat{c}_\mathrm{d} (\Omega_{\mathrm{cmd},i}^2 - \Omega_i^2)
            & \Omega_{\mathrm{cmd},i} \geq \Omega_i \\[4pt]
        \check{c}_\mathrm{v} (\Omega_{\mathrm{cmd},i} - \Omega_i)
        + \check{c}_\mathrm{d} (\Omega_{\mathrm{cmd},i}^2 - \Omega_i^2)
            & \Omega_{\mathrm{cmd},i} < \Omega_i
    \end{cases}
\]

where \(\Omega_{\mathrm{cmd},i}\) is the commanded rotor speed, \(\hat{c}_\mathrm{v}\) and
\(\hat{c}_\mathrm{d}\) are the viscous damping and drag parameters during acceleration, and
\(\check{c}_\mathrm{v}\) and \(\check{c}_\mathrm{d}\) those during deceleration.

The rigid-body equations of motion are:

\[
\begin{aligned}
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    \dot{\mathbf{q}} &= \tfrac{1}{2}
        \mathbf{q} \otimes \begin{bmatrix} {}^{\mathcal{B}}\boldsymbol{\omega}\\0 \end{bmatrix}, \\
    m\dot{\mathbf{v}} &= m\mathbf{g}
        + \mathbf{R}\,{}^{\mathcal{B}}\mathbf{f}_\mathrm{t}
        + \mathbf{R}\,{}^{\mathcal{B}}\mathbf{f}_\mathrm{a}, \\
    \mathbf{J}\,{}^{\mathcal{B}}\dot{\boldsymbol{\omega}} &=
        {}^{\mathcal{B}}\mathbf{t}_\Sigma
        - {}^{\mathcal{B}}\boldsymbol{\omega}
          \times \mathbf{J}\,{}^{\mathcal{B}}\boldsymbol{\omega},
\end{aligned}
\]

where \(\mathbf{p}\) and \(\mathbf{v}\) are the position and velocity, \(\mathbf{q}\) is the
orientation as a scalar-last quaternion, \({}^{\mathcal{B}}\boldsymbol{\omega}\) is the angular
velocity, \({}^{\mathcal{B}}(\cdot)\) denotes the body frame, \(\otimes\) is the quaternion product,
\(m\) is the mass, \(\mathbf{J}\) is the inertia matrix, \(\mathbf{g}\) is the gravity vector,
\(\mathbf{R} = {}^{\mathcal{I}}\mathbf{R}_{\mathcal{B}}(\mathbf{q})\) is the rotation from body to
world frame, and the forces and torques are:

\[
\begin{aligned}
    {}^{\mathcal{B}}\mathbf{f}_\mathrm{t} &=
        \mathbf{e}_\mathrm{z} \textstyle\sum_{i=1}^{4} f_{\mathrm{m},i}, \\
    {}^{\mathcal{B}}\mathbf{f}_\mathrm{a} &= \mathbf{C}_\mathrm{a}\,\mathbf{R}^{\top}\mathbf{v}, \\
    {}^{\mathcal{B}}\mathbf{t}_\Sigma &=
        {}^{\mathcal{B}}\mathbf{t}_\mathrm{t}
        + {}^{\mathcal{B}}\mathbf{t}_\mathrm{d}
        + {}^{\mathcal{B}}\mathbf{t}_\mathrm{g}
        + {}^{\mathcal{B}}\mathbf{t}_\mathrm{r},
\end{aligned}
\]

with:

\[
\begin{aligned}
    {}^{\mathcal{B}}\mathbf{t}_\mathrm{t} &=
        l
        \begin{bmatrix}1&0&0\\0&1&0\\0&0&0\end{bmatrix}
        \mathbf{M}\,\mathbf{f}_\mathrm{m}, \\
    {}^{\mathcal{B}}\mathbf{t}_\mathrm{d} &=
        \begin{bmatrix}0&0&0\\0&0&0\\0&0&1\end{bmatrix}
        \mathbf{M}\,\mathbf{t}_\mathrm{m}, \\
    {}^{\mathcal{B}}\mathbf{t}_\mathrm{g} &= J_\mathrm{p}
        \left({}^{\mathcal{B}}\boldsymbol{\omega} \times \mathbf{e}_\mathrm{z}\right)
        \mathbf{e}_\mathrm{z}^{\top} \mathbf{M}\,\boldsymbol{\Omega}, \\
    {}^{\mathcal{B}}\mathbf{t}_\mathrm{r} &= J_\mathrm{p}\,\mathbf{e}_\mathrm{z}\,
        \mathbf{e}_\mathrm{z}^{\top} \mathbf{M}\,\dot{\boldsymbol{\Omega}},
\end{aligned}
\]

where \({}^{\mathcal{B}}\mathbf{f}_\mathrm{t}\) and \({}^{\mathcal{B}}\mathbf{f}_\mathrm{a}\) are
the thrust and aerodynamic drag force, \({}^{\mathcal{B}}\mathbf{t}_\mathrm{t}\),
\({}^{\mathcal{B}}\mathbf{t}_\mathrm{d}\), \({}^{\mathcal{B}}\mathbf{t}_\mathrm{g}\), and
\({}^{\mathcal{B}}\mathbf{t}_\mathrm{r}\) are the thrust, drag, gyroscopic, and reaction torque,
\(\mathbf{e}_\mathrm{z}\) is the unit vector in z direction, \(\mathbf{C}_\mathrm{a}\) is the
body-frame drag matrix, \(l\) is the distance of the motors to the body axes, \(\mathbf{M}\) is
the \(3\times 4\) mixing matrix, \(J_\mathrm{p}\) is the combined inertia of one propeller and its
motor, and \(\mathbf{f}_\mathrm{m}\), \(\mathbf{t}_\mathrm{m}\), and \(\boldsymbol{\Omega}\) stack
the four motor thrusts, torques, and rotor speeds. The gyroscopic and reaction torques convert
\(\boldsymbol{\Omega}\) and \(\dot{\boldsymbol{\Omega}}\) to rad/s internally, so \(J_\mathrm{p}\)
is in SI units.
"""

from crazyflow.dynamics.first_principles.dynamics import (
    Params,
    dynamics,
    sim_dynamics,
    symbolic_dynamics,
)

__all__ = ["dynamics", "symbolic_dynamics", "sim_dynamics", "Params"]
