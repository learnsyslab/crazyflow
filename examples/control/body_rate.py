import os
from functools import partial
from typing import Callable

import jax.numpy as jnp
import numpy as np

os.environ["SCIPY_ARRAY_API"] = "1"

from scipy.spatial.transform import Rotation as R

from crazyflow.control import Control, parametrize
from crazyflow.control.mellinger import state2attitude
from crazyflow.sim import Sim

kp_att = 8.0  # Proportional gain from attitude error to body rates


def control(
    t: float,
    obs: dict[str, np.ndarray],
    pos_start: np.ndarray,
    position_ctrl: Callable,
    pos_err_i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the body rate command to track a circle with a slow climb.

    The attitude command of the position controller is converted into body rates with a
    proportional attitude loop.

    Args:
        t: Time since the start of the trajectory in s.
        obs: Drone position, orientation, and velocity.
        pos_start: Start position of the drone.
        position_ctrl: Controller that maps the state and a full state command to an attitude
            command. Any controller that outputs [roll, pitch, yaw, thrust] can be used here.
        pos_err_i: Integral error of the position controller from the previous call.

    Returns:
        The body rate command [roll_rate, pitch_rate, yaw_rate, thrust] in rad/s and N, and the
        updated integral error.
    """
    # Full state command with velocity and acceleration feedforward
    cmd = np.zeros(13)
    cmd[:3] = pos_start + np.array([np.cos(t) - 1, np.sin(t), 0.2 * t])
    cmd[3:6] = np.array([-np.sin(t), np.cos(t), 0.2])
    cmd[6:9] = np.array([-np.cos(t), -np.sin(t), 0.0])
    cmd[9] = t  # Yaw
    rpyt, pos_err_i = position_ctrl(obs["pos"], obs["quat"], obs["vel"], cmd, pos_err_i)
    rot_err = (R.from_quat(obs["quat"]).inv() * R.from_euler("xyz", rpyt[:3])).as_rotvec()
    return np.concatenate([kp_att * rot_err, rpyt[3:]]), pos_err_i


def main():
    sim = Sim(control=Control.body_rate, body_rate_freq=250)
    # The firmware has no dedicated body rate mode. Its attitude terms level the drone at the
    # current yaw and would counteract the commanded rates. Disable them to track body rates.
    body_rate = sim.data.controls.body_rate
    params = body_rate.params | {"kR": jnp.zeros(3), "ki_m": jnp.zeros(3)}
    sim.data = sim.data.replace(
        controls=sim.data.controls.replace(body_rate=body_rate.replace(params=params))
    )
    sim.build_default_data()
    sim.reset()
    duration = 6.5
    fps = 60

    # We use the Mellinger position controller to generate attitude commands. This could be any
    # controller that outputs [roll, pitch, yaw, thrust], e.g. a learned policy.
    position_ctrl = partial(parametrize(state2attitude, sim.drone), ctrl_freq=sim.control_freq)
    pos_err_i = np.zeros(3)
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 4))  # [roll_rate, pitch_rate, yaw_rate, thrust]
    pos_start = np.asarray(sim.data.states.pos[0, 0])
    for i in range(int(duration * sim.control_freq)):
        # Convert the states to numpy so that the controller runs in numpy instead of eager JAX
        obs = {
            "pos": np.asarray(sim.data.states.pos[0, 0]),
            "quat": np.asarray(sim.data.states.quat[0, 0]),
            "vel": np.asarray(sim.data.states.vel[0, 0]),
        }
        cmd[0, 0, :], pos_err_i = control(
            i / sim.control_freq, obs, pos_start, position_ctrl, pos_err_i
        )
        sim.body_rate_control(cmd)
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


if __name__ == "__main__":
    main()
