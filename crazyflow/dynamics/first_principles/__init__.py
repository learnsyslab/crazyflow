r"""Full rigid-body dynamics for a quadrotor.

This package implements Newton-Euler dynamics based on physical parameters: mass, inertia, motor
thrust and torque curves, arm length, and drag coefficients. The command interface is four motor
angular velocities in RPM. Mass and arm length are measured directly and the propeller inertia is
taken from CAD data. The thrust and torque curves are fitted to load cell data, and the inertia and
drag coefficients are identified from flight data.

When rotor dynamics are modelled, the rotor speeds evolve as:

\[
    \dot{\boldsymbol{\Omega}} = \begin{cases}
        \hat{c}_\mathrm{v} (\boldsymbol{\Omega}_\mathrm{cmd} - \boldsymbol{\Omega})
        + \hat{c}_\mathrm{d} (\boldsymbol{\Omega}_\mathrm{cmd}^2 - \boldsymbol{\Omega}^2)
            & \forall\, \boldsymbol{\Omega}_\mathrm{cmd} \geq \boldsymbol{\Omega}, \\
        \check{c}_\mathrm{v} (\boldsymbol{\Omega}_\mathrm{cmd} - \boldsymbol{\Omega})
        + \check{c}_\mathrm{d} (\boldsymbol{\Omega}_\mathrm{cmd}^2 - \boldsymbol{\Omega}^2)
            & \forall\, \boldsymbol{\Omega}_\mathrm{cmd} < \boldsymbol{\Omega}.
    \end{cases}
\]

Each motor produces a thrust and a drag torque, both quadratic polynomials of its rotor speed in
RPM:

\[
    f_{\mathrm{m},i} = k_{\mathrm{f},0} + k_{\mathrm{f},1} \Omega_i + k_{\mathrm{f},2} \Omega_i^2,
    \qquad
    t_{\mathrm{m},i} = k_{\mathrm{t},0} + k_{\mathrm{t},1} \Omega_i + k_{\mathrm{t},2} \Omega_i^2.
\]

The rigid-body equations of motion are:

\[
\begin{aligned}
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    \dot{\mathbf{q}} &= \tfrac{1}{2}
        \mathbf{q} \otimes \begin{bmatrix} {}^{\mathcal{B}}\boldsymbol{\omega}\\0 \end{bmatrix}, \\
    m\dot{\mathbf{v}} &= \mathbf{f}_\Sigma, \\
    \mathbf{J}\,{}^{\mathcal{B}}\dot{\boldsymbol{\omega}} &=
        {}^{\mathcal{B}}\mathbf{t}_\Sigma
        - {}^{\mathcal{B}}\boldsymbol{\omega}
          \times \mathbf{J}\,{}^{\mathcal{B}}\boldsymbol{\omega},
\end{aligned}
\]

where the total force and torque are:

\[
\begin{aligned}
    \mathbf{f}_\Sigma &= \mathbf{f}_\mathrm{g}
        + \mathbf{R}\,{}^{\mathcal{B}}\mathbf{f}_\mathrm{t}
        + \mathbf{R}\,{}^{\mathcal{B}}\mathbf{f}_\mathrm{a}, \\
    {}^{\mathcal{B}}\mathbf{t}_\Sigma &= {}^{\mathcal{B}}\mathbf{t}_\mathrm{t}
        + {}^{\mathcal{B}}\mathbf{t}_\mathrm{d}
        + {}^{\mathcal{B}}\mathbf{t}_\mathrm{g}
        + {}^{\mathcal{B}}\mathbf{t}_\mathrm{r}.
\end{aligned}
\]

The individual terms are defined as:

\[
\begin{aligned}
    \mathbf{f}_\mathrm{g} &= m\mathbf{g}, \\
    {}^{\mathcal{B}}\mathbf{f}_\mathrm{t} &=
        \mathbf{e}_\mathrm{z} \textstyle\sum_{i=1}^{4} f_{\mathrm{m},i}, \\
    {}^{\mathcal{B}}\mathbf{f}_\mathrm{a} &= \mathbf{C}_\mathrm{a}\,{}^{\mathcal{B}}\mathbf{v}, \\
    {}^{\mathcal{B}}\mathbf{t}_\mathrm{t} &=
        l\,\mathrm{diag}(1, 1, 0)\,\mathbf{M}\,\mathbf{f}_\mathrm{m}, \\
    {}^{\mathcal{B}}\mathbf{t}_\mathrm{d} &=
        \mathrm{diag}(0, 0, 1)\,\mathbf{M}\,\mathbf{t}_\mathrm{m}, \\
    {}^{\mathcal{B}}\mathbf{t}_\mathrm{g} &= J_\mathrm{p}
        \left({}^{\mathcal{B}}\boldsymbol{\omega} \times \mathbf{e}_\mathrm{z}\right)
        \mathbf{e}_\mathrm{z}^{\top} \mathbf{M}\,\boldsymbol{\Omega}, \\
    {}^{\mathcal{B}}\mathbf{t}_\mathrm{r} &= J_\mathrm{p}\,\mathbf{e}_\mathrm{z}\,
        \mathbf{e}_\mathrm{z}^{\top} \mathbf{M}\,\dot{\boldsymbol{\Omega}}.
\end{aligned}
\]

The gyroscopic and reaction torques convert \(\boldsymbol{\Omega}\) and
\(\dot{\boldsymbol{\Omega}}\) to rad/s.

| Variable | Name | Description |
| --- | --- | --- |
| \(\boldsymbol{\Omega}\), \(\Omega_i\) | `rotor_vel` | Rotor speeds in RPM |
| \(\boldsymbol{\Omega}_\mathrm{cmd}\) | `cmd` | Commanded rotor speeds in RPM |
| \(\mathbf{f}_\mathrm{m}\), \(f_{\mathrm{m},i}\) | | Motor thrusts |
| \(\mathbf{t}_\mathrm{m}\), \(t_{\mathrm{m},i}\) | | Motor drag torques |
| \(\mathbf{p}\) | `pos` | Position in m |
| \(\mathbf{q}\) | `quat` | Orientation as a scalar-last quaternion |
| \(\mathbf{v}\) | `vel` | Velocity in m/s |
| \({}^{\mathcal{B}}\boldsymbol{\omega}\) | `ang_vel` | Angular velocity in rad/s |
| \(\mathbf{f}_\Sigma\) | | Total force |
| \({}^{\mathcal{B}}\mathbf{t}_\Sigma\) | | Total torque |
| \(\mathbf{f}_\mathrm{g}\) | | Gravitational force |
| \({}^{\mathcal{B}}\mathbf{f}_\mathrm{t}\) | | Thrust force |
| \({}^{\mathcal{B}}\mathbf{f}_\mathrm{a}\) | | Aerodynamic drag force |
| \({}^{\mathcal{B}}\mathbf{t}_\mathrm{t}\) | | Thrust torque |
| \({}^{\mathcal{B}}\mathbf{t}_\mathrm{d}\) | | Aerodynamic counter torque of the propellers |
| \({}^{\mathcal{B}}\mathbf{t}_\mathrm{g}\) | | Gyroscopic torque |
| \({}^{\mathcal{B}}\mathbf{t}_\mathrm{r}\) | | Reaction torque |
| \(\mathbf{R}\) | | Rotation from body to world frame |
| \(\mathbf{e}_\mathrm{z}\) | | Unit vector in z direction |
| \({}^{\mathcal{B}}(\cdot)\) | | Quantity in the body frame, world frame if unmarked |
| \(\otimes\) | | Quaternion product |
| \(\mathrm{diag}(\cdot)\) | | Diagonal matrix |

| Parameter | Name | Description |
| --- | --- | --- |
| \(\hat{c}_\mathrm{v}\) | `rotor_dyn_coef` | Viscous damping on spin-up, entry 0 |
| \(\hat{c}_\mathrm{d}\) | `rotor_dyn_coef` | Drag on spin-up, entry 1 |
| \(\check{c}_\mathrm{v}\) | `rotor_dyn_coef` | Viscous damping on spin-down, entry 2 |
| \(\check{c}_\mathrm{d}\) | `rotor_dyn_coef` | Drag on spin-down, entry 3 |
| \(k_{\mathrm{f},0}, k_{\mathrm{f},1}, k_{\mathrm{f},2}\) | `rpm2thrust` | Thrust curve |
| \(k_{\mathrm{t},0}, k_{\mathrm{t},1}, k_{\mathrm{t},2}\) | `rpm2torque` | Torque curve |
| \(m\) | `mass` | Mass in kg |
| \(\mathbf{J}\) | `J` | Inertia matrix in kg m² |
| \(\mathbf{g}\) | `gravity_vec` | Gravity vector in m/s² |
| \(l\) | `L` | Distance of the motors to the body axes in m |
| \(\mathbf{M}\) | `mixing_matrix` | Mixing matrix of motor placement and spin direction |
| \(J_\mathrm{p}\) | `prop_inertia` | Combined inertia of one propeller and its motor in kg m² |
| \(\mathbf{C}_\mathrm{a}\) | `drag_matrix` | Drag coefficients in matrix form in N/(m/s) |
"""

from crazyflow.dynamics.first_principles.dynamics import (
    Params,
    dynamics,
    sim_dynamics,
    symbolic_dynamics,
)

__all__ = ["dynamics", "symbolic_dynamics", "sim_dynamics", "Params"]
