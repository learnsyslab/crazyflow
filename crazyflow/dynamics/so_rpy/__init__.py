r"""Second-order fitted RPY dynamics (no rotor dynamics).

Rotational dynamics are modelled as a fitted second-order linear system driven by roll, pitch, and
yaw commands. Translational dynamics are driven by the collective thrust command directly, with no
motor spin-up lag. The command interface is ``[roll_rad, pitch_rad, yaw_rad, thrust_N]``.

\[
\begin{aligned}
    \dot{\mathbf{p}} &= \mathbf{v}, \\
    m\dot{\mathbf{v}} &= m\mathbf{g}
        + (c_{\mathrm{acc}} + c_f F_{\mathrm{cmd}})\,R\,\mathbf{e}_z, \\
    \ddot{\boldsymbol{\psi}} &=
        c_{\psi}\,\boldsymbol{\psi}
        + c_{\dot{\psi}}\,\dot{\boldsymbol{\psi}}
        + c_u\,\mathbf{u}_{\mathrm{rpy}},
\end{aligned}
\]

The vector \(\boldsymbol{\psi} = [\phi,\theta,\psi]^{\top}\) holds the roll, pitch, and yaw angles
with rates \(\dot{\boldsymbol{\psi}}\). The coefficients \(c_{\psi}\), \(c_{\dot{\psi}}\), and
\(c_u\) are identified from flight data.

!!! note
    This is the native Euler-angle form, matching
    [symbolic_dynamics_euler][crazyflow.dynamics.so_rpy.symbolic_dynamics_euler]. The simulation
    does not integrate this state directly. It shares the common ``[pos, quat, vel, ang_vel]`` state
    with the other models and advances the orientation from the body angular velocity
    \({}^{\mathcal{B}}\boldsymbol{\omega}\), converting \(\ddot{\boldsymbol{\psi}} \leftrightarrow
    {}^{\mathcal{B}}\dot{\boldsymbol{\omega}}\) through the kinematic Jacobian at every step.
    Integrating from \({}^{\mathcal{B}}\boldsymbol{\omega}\) rather than \(\dot{\boldsymbol{\psi}}\)
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
