"""Example showing how to change the used camera and how to extract the pixel information."""

import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib import animation

from crazyflow.control import Control
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator
from crazyflow.sim.physics import Physics


def control(t: float, t_tot: float) -> np.ndarray:
    phi = 2 * np.pi * t / t_tot + np.pi
    circle = np.array([np.cos(phi), np.sin(phi)])
    cmd = np.zeros((1, 1, 13))
    cmd[..., :2] = circle  # xy
    cmd[..., 2] = 0.1 + 0.5 * t / t_tot  # z
    cmd[..., -4] = 1.9 * np.pi * t / t_tot  # yaw

    return cmd


def add_smiley(sim: Sim):
    # Add 3d object to sim
    # create box spec from an XML string
    box_xml = """
    <mujoco model="box_model">
      <worldbody>
        <body name="cube" pos="0 0 0">
          <geom type="box" size="0.05 0.05 0.05" rgba="0.8 0.4 0.2 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    box_spec = mujoco.MjSpec.from_string(box_xml)
    frame = sim.spec.worldbody.add_frame()
    boxes = [
        # eyes
        ((0.0, -0.15, 0.6), (1, 0, 0, 0)),
        ((0.0, 0.15, 0.6), (1, 0, 0, 0)),
        # mouth
        ((0.0, -0.2, 0.4), (1, 0, 0, 0)),
        ((0.0, 0.2, 0.4), (1, 0, 0, 0)),
        ((0.0, -0.1, 0.3), (1, 0, 0, 0)),
        ((0.0, 0.0, 0.3), (1, 0, 0, 0)),
        ((0.0, 0.1, 0.3), (1, 0, 0, 0)),
    ]
    for i, x in enumerate(boxes):
        box_body = box_spec.body("cube")
        box = frame.attach_body(box_body, "", f":{i}")
        box.pos = x[0]
        box.quat = x[1]
    sim.build_mjx()
    sim.build_reset_fn()


def main(show_plot: bool = False, save_plot: bool = False):
    """Example showing the rendering feature and saving a gif via FuncAnimation."""
    # Setup sim
    sim = Sim(
        n_drones=1,
        control=Control.state,
        integrator=Integrator.rk4,
        physics=Physics.first_principles,
        drone_model="cf2x_T350",
    )
    add_smiley(sim)
    sim.reset()
    pos = sim.data.states.pos.at[...].set([-1, 0, 0])
    states = sim.data.states.replace(pos=pos)
    sim.data = sim.data.replace(states=states)
    duration = 5
    fps = 50
    timings = []

    # Set up matplotlib rendering
    resolution = (160, 120)
    rgb = np.zeros((resolution[1], resolution[0], 3))
    d = np.zeros((resolution[1], resolution[0]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    im1 = ax1.imshow(rgb)
    ax1.set_title("RGB")
    ax1.axis("off")
    im2 = ax2.imshow(d, cmap="viridis")
    ax2.set_title("Depth")
    ax2.axis("off")
    fig.tight_layout()

    # Animation setup
    def update_frame(_):  # noqa: ANN202
        t = sim.data.core.steps[0, 0] / sim.freq
        sim.state_control(control(t, duration))
        sim.step(sim.freq // fps)

        t1 = time.perf_counter()
        # mode: Either "human" for the regular window, "rgb_array" for an RGB array,
        #       "depth_array" for a depth array, or "rgbd_tuple" for both at the same time.
        # camera: The name or id of the camera. The names are specified in the corresponding
        #         xml file in drone_models. For example, "fpv_cam:0" is the first-person view camera
        #         of the first drone, "track_cam:0" is the tracking camera of the first drone.
        #         Id -1 is the global camera.
        rgbd = sim.render(
            width=resolution[0], height=resolution[1], mode="rgbd_tuple", camera="fpv_cam:0"
        )
        t2 = time.perf_counter()
        timings.append(t2 - t1)
        if rgbd is None:
            return im1, im2
        rgb, depth = rgbd
        im1.set_data(rgb)
        im2.set_data(depth)
        im2.set_clim(np.nanmin(depth), np.nanmax(depth))
        return im1, im2

    anim = animation.FuncAnimation(
        fig, update_frame, frames=int(duration * fps), interval=1000 / fps, blit=True, repeat=False
    )
    if show_plot:
        plt.show()
    if save_plot:
        anim.save("cameras.gif", writer="pillow", fps=fps)

    sim.close()

    t_mean = np.mean(timings)
    print(f"Average render time {t_mean * 1000:.2f}ms, eqivalent to {1 / t_mean:.2f}fps")
    print("For more optimized depth rendering, check out the raycasting.py example.")


if __name__ == "__main__":
    main(show_plot=True, save_plot=False)
