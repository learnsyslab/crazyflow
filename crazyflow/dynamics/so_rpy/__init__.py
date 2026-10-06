r"""Second-order fitted RPY dynamics (no rotor dynamics).

Rotational dynamics are modelled as a fitted second-order linear system driven by roll, pitch, and
yaw commands. Translational dynamics are driven by the collective thrust command directly, with no
motor spin-up lag. The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``.

\[
\begin{aligned}
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    m\dot{\mathbf{v}} &= m\mathbf{g}
        + (c_\mathrm{acc} + c_\mathrm{f} f_{\Sigma,\mathrm{cmd}})
          \,\mathbf{R}\,\mathbf{e}_\mathrm{z}, \\
    \ddot{\boldsymbol{\Psi}} &=
        \boldsymbol{c}_{\boldsymbol{\Psi},1}\,\boldsymbol{\Psi}
        + \boldsymbol{c}_{\boldsymbol{\Psi},2}\,\dot{\boldsymbol{\Psi}}
        + \boldsymbol{c}_{\boldsymbol{\Psi},3}\,\boldsymbol{\Psi}_\mathrm{cmd},
\end{aligned}
\]

where \(\mathbf{p}\) and \(\mathbf{v}\) are the position and velocity, \(m\) is the mass,
\(\mathbf{g}\) is the gravity vector, \(\mathbf{e}_\mathrm{z}\) is the unit vector in z direction,
\(\mathbf{R} = {}^{\mathcal{I}}\mathbf{R}_{\mathcal{B}}(\boldsymbol{\Psi})\) is the rotation from
body to world frame, \(\boldsymbol{\Psi} = [\phi,\theta,\psi]^{\top}\) holds the roll, pitch, and
yaw angles with rates \(\dot{\boldsymbol{\Psi}}\), and \(f_{\Sigma,\mathrm{cmd}}\) and
\(\boldsymbol{\Psi}_\mathrm{cmd}\) are the commanded collective thrust and attitude. The thrust
offset \(c_\mathrm{acc}\), the thrust scaling coefficient \(c_\mathrm{f}\), and the rotational
coefficients \(\boldsymbol{c}_{\boldsymbol{\Psi},1}\), \(\boldsymbol{c}_{\boldsymbol{\Psi},2}\), and
\(\boldsymbol{c}_{\boldsymbol{\Psi},3}\) are identified from flight data.

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
"""

from crazyflow.dynamics.so_rpy.dynamics import (
    Params,
    dynamics,
    sim_dynamics,
    symbolic_dynamics,
    symbolic_dynamics_euler,
)

__all__ = ["Params", "dynamics", "sim_dynamics", "symbolic_dynamics", "symbolic_dynamics_euler"]
