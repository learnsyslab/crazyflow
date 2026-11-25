"""Web viewer for Crazyflow built on mjviser."""

from __future__ import annotations

import atexit
import copy
import time
import webbrowser
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING

import jax
import mujoco
import numpy as np
import trimesh
import viser
from mjviser import ViserMujocoScene
from scipy.spatial.transform import RigidTransform
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from mujoco.mjx import Data
    from numpy.typing import NDArray

    from crazyflow.sim.data import SimData
    from crazyflow.sim.splat import SplatData

try:
    import splax
    from splax.io import apply_activations, sh_to_rgb
except ImportError:
    splax = None  # splax is optional and only required for splats


class ViserViewer:
    """Visualize crazyflow in a viser web viewer."""

    def __init__(self, mj_model: mujoco.MjModel, data: SimData, port: int = 8080):
        self.server = viser.ViserServer(port=port)
        self.paused, self._advance = False, False
        self._view = (None, None)  # Last world and camera
        self._clock = (0.0, 0.0, np.nan)  # Wall time, sim time, speed
        self._markers = {"lines": defaultdict(list), "spheres": [], "cylinders": []}
        self._marker_handles = []
        self._scene_loaded = False
        self._mj_model = mj_model
        self._init_scene(data.states.pos.shape[0])
        self._init_body_offsets()
        self._init_splats(data)
        distance = self._frame_drones(data.states.pos[0])
        self._init_menu(distance, splats=bool(self._splats))
        self._show_meshes(not self._splats)
        self._show_splats(bool(self._splats))
        # Block like a window opening so the sim only starts once everything is ready
        if webbrowser.open(f"http://localhost:{self.server.get_port()}"):
            while not self.server.get_clients():
                time.sleep(0.1)
        atexit.register(self.close)  # Also mark the page as disconnected on crashes and Ctrl+C

    def add_lines(self, points: NDArray, rgba: NDArray, thickness: float):
        """Draw line segments in the next frame.

        Args:
            points: Start and end point of each segment, shape (N, 2, 3).
            rgba: Color, shape (4,). Lines ignore the opacity.
            thickness: Line thickness in meters.
        """
        colors = np.broadcast_to(rgba[:3], points.shape)
        self._markers["lines"][thickness].append((points, colors))

    def add_spheres(self, pos: NDArray, radius: float, rgba: NDArray):
        """Draw spheres in the next frame.

        Args:
            pos: Sphere centers, shape (N, 3).
            radius: Sphere radius in meters.
            rgba: Color, shape (4,).
        """
        n = len(pos)
        wxyz = R.identity(n).as_quat(scalar_first=True)
        self._markers["spheres"].append((pos, wxyz, np.full(n, radius), np.tile(rgba, (n, 1))))

    def add_cylinder(self, p1: NDArray, p2: NDArray, radius: float, rgba: NDArray):
        """Draw a cylinder between two points in the next frame.

        Args:
            p1: Start point, shape (3,).
            p2: End point, shape (3,).
            radius: Cylinder radius in meters.
            rgba: Color, shape (4,).
        """
        rot, _ = R.align_vectors(p2 - p1, [0.0, 0.0, 1.0])
        scale = [radius, radius, np.linalg.norm(p2 - p1)]
        wxyz = rot.as_quat(scalar_first=True)
        self._markers["cylinders"].append(([(p1 + p2) / 2], [wxyz], [scale], [rgba]))

    def render(
        self,
        mjx_data: Data,
        sim_time: NDArray,
        world: int | None,
        camera: int | str | None,
        cam_config: dict | None,
        capture: bool,
        width: int,
        height: int,
    ) -> NDArray | None:
        """Show the current state and wait until the shown world runs in real time.

        Args:
            mjx_data: MJX data of all worlds.
            sim_time: Simulation time of all worlds, shape (n_worlds, 1).
            world: Index of the world to show. None keeps the world selected in the menu.
            camera: Camera name or id. -1 is the free camera. None keeps the selected camera.
            cam_config: Free camera configuration.
            capture: Whether to return the rendered image.
            width: Image width of the capture.
            height: Image height of the capture.

        Returns:
            The captured RGB image with shape (height, width, 3) if capture is set, else None.
        """
        # World and camera only apply when they change to avoid overriding the menu
        last_world, last_camera = self._view
        self._view = (world, camera)
        # mjviser has no world slider for a single world
        if world not in (None, last_world) and self.scene.num_envs > 1:
            inputs = self.server.gui._gui_input_handle_from_uuid.values()
            next(handle for handle in inputs if handle.label == "Select").value = world
        if camera not in (None, last_camera):
            self._cameras.value = self._mj_model.cam(camera).name if camera != -1 else "Free"
        if cam_config is not None:
            self._set_free_camera(**cam_config)
        xpos, xmat, mocap_pos, mocap_quat, qpos, qvel = jax.device_get(
            (
                mjx_data.xpos,
                mjx_data.xmat,
                mjx_data.mocap_pos,
                mjx_data.mocap_quat,
                mjx_data.qpos,
                mjx_data.qvel,
            )
        )
        root = RigidTransform.from_components(
            xpos[:, self._root], R.from_matrix(xmat[:, self._root])
        )
        body = root * self._offset
        with self.server.atomic():
            if self._meshes_visible:  # Hidden meshes need no updates
                self.scene.update_from_arrays(
                    body.translation,
                    body.rotation.as_matrix(),
                    mocap_pos,
                    mocap_quat,
                    qpos=qpos,
                    qvel=qvel,
                )
            if self._splats_visible:  # Hidden splats need no updates
                self._draw_splats(mocap_pos, mocap_quat)
            self._draw_markers()
            if self._cameras.value != "Free":
                self._follow_camera(body)
        image = None
        if capture:
            clients = self.server.get_clients()
            assert clients, "Capturing needs a browser connected to the viewer"
            client = max(clients.items())[1]  # The newest client
            image, previous = client.get_render(height, width), None
            # The browser loads meshes asynchronously after connecting, so wait for a stable image
            while not self._scene_loaded and not np.array_equal(image, previous):
                image, previous = client.get_render(height, width), image
            self._scene_loaded = True
        self._wait(float(sim_time[self.scene.env_idx, 0]))
        return image

    def close(self):
        atexit.unregister(self.close)
        # Browsers only let scripts close tabs with a single history entry. If it does, it shows the
        # disconnect modal.
        with self.server.gui.add_modal("Disconnected"):
            self.server.gui.add_markdown("## The simulation has ended.")
        close_tab = viser._messages.RunJavascriptMessage(source="window.close()")
        self.server.gui._websock_interface.queue_message(close_tab)  # No public API for this
        self.server.flush()
        time.sleep(0.2)  # Give the websocket time to deliver both messages before it closes
        self.server.stop()

    def _init_scene(self, n_worlds: int):
        """Create the mjviser scene of all worlds."""
        model = copy.deepcopy(self._mj_model)  # mjviser rescales the model in place, so we copy it
        # mjviser ignores geoms with transparent geom colors, but not geoms with transparent
        # materials, so we copy over the transparency
        model.geom_rgba[(model.geom_matid >= 0) & (model.mat_rgba[model.geom_matid, 3] == 0), 3] = 0
        self.scene = ViserMujocoScene(self.server, model, num_envs=n_worlds)
        self.scene.show_only_selected = True  # Only show a single world unless hiding is disabled
        # we need the grids when switching to dark mode, so we store their handles and properties
        self._grids = [
            (grid, (grid.plane_color, grid.plane_opacity, grid.cell_color, grid.section_color))
            for grid in self.server.scene._handle_from_node_name.values()
            if isinstance(grid, viser.GridHandle)
        ]

    def _init_body_offsets(self):
        """Compute the constant poses of all bodies and cameras relative to their parents."""
        # MJX does not move the children of mocap bodies, so we pose every body relative to its
        # root ourselves. Crazyflow bodies have no joints, so these offsets are constant.
        mj_model, mj_data = self._mj_model, mujoco.MjData(self._mj_model)
        mujoco.mj_kinematics(mj_model, mj_data)
        self._root = mj_model.body_rootid
        world_pose = RigidTransform.from_components(
            mj_data.xpos, R.from_quat(mj_data.xquat, scalar_first=True)
        )
        self._offset = world_pose[self._root].inv() * world_pose
        self._body_to_cam = RigidTransform.from_components(
            mj_model.cam_pos, R.from_quat(mj_model.cam_quat, scalar_first=True)
        )

    def _init_splats(self, data: SimData):
        """Upload the static scene splat and one splat per drone, if splats are attached."""
        self._splats, self._drone_splats = [], []
        if "splats" not in data.plugins:  # The plugin key of crazyflow.sim.splat
            return
        if splax is None:
            raise ImportError("Splax is required for splats, but could not be imported")
        self._drone_mocap_ids = jax.device_get(data.core.drone_mocap_ids)

        splats: SplatData = jax.device_get(data.plugins["splats"])
        scales, opacities = jax.device_get(
            apply_activations(splats.log_scales, splats.logit_opacities)
        )
        rot = R.from_quat(splats.quats, scalar_first=True).as_matrix()
        params = {
            "centers": splats.means,
            "covariances": np.einsum("nij,nj,nkj->nik", rot, scales**2, rot),
            # viser has no spherical harmonics and needs displayable colors
            "rgbs": np.clip(jax.device_get(sh_to_rgb(splats.sh_colors[:, 0])), 0.0, 1.0),
            "opacities": opacities[:, None],
        }
        self._drone_splats = [
            self.server.scene.add_gaussian_splats(
                f"/splats/{i}:drone", **{key: value[start:stop] for key, value in params.items()}
            )
            for i, (start, stop) in enumerate(splats.slices)
        ]
        self._splats = self._drone_splats.copy()
        if n_scene := splats.slices[0][0] if splats.slices else len(splats.means):
            self._splats.append(
                self.server.scene.add_gaussian_splats(
                    "/splats/scene", **{key: value[:n_scene] for key, value in params.items()}
                )
            )

    def _frame_drones(self, drone_pos: NDArray) -> float:
        """Center the free camera on the drones and return a distance that shows most of them."""
        center = np.mean(drone_pos, axis=0)
        # 90th percentile so a few outliers do not push the camera far away
        radius = np.percentile(np.linalg.norm(drone_pos - center, axis=-1), 90) + 0.25
        fov = np.deg2rad(60)  # mjviser's default FOV
        distance = max(radius / np.sin(fov / 2), 1.0)
        # Camera offset for mjviser's default azimuth and elevation
        direction = -R.from_euler("ZY", [120, 20], degrees=True).as_matrix()[:, 0]
        self.scene.mj_model.stat.center = center  # Center of new cameras
        # "Reset View" in the browser returns to the initial camera
        self.server.initial_camera.fov = fov
        self.server.initial_camera.look_at = center
        self.server.initial_camera.position = center + distance * direction
        return distance

    def _init_menu(self, distance: float, splats: bool):
        """Create the mjviser menu with a Controls tab and hotkeys."""
        self._menu = self.scene.create_visualization_gui(camera_distance=distance)
        with self._menu.add_tab("Controls", icon=viser.Icon.KEYBOARD):
            self._pause_button = self.server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
            step_button = self.server.gui.add_button("Step", icon=viser.Icon.PLAYER_TRACK_NEXT)
            self._speed = self.server.gui.add_number("Speed", initial_value=1.0, min=0.01)
            dark_mode = self.server.gui.add_checkbox("Dark mode", initial_value=False)
            if splats:
                show_splats = self.server.gui.add_checkbox("Gaussian splats", initial_value=True)
                show_meshes = self.server.gui.add_checkbox("Meshes", initial_value=False)
                show_splats.on_update(lambda _: self._show_splats(show_splats.value))
                show_meshes.on_update(lambda _: self._show_meshes(show_meshes.value))
            self.server.gui.add_markdown(
                "| Key | Action |\n|---|---|\n| Space | Pause / play |\n| Tab | Next camera |\n"
                "| C | Contact forces |\n| H | Hide menu |\n"
                "| 0-4 | Geom groups |"
            )
        self._pause_button.on_click(lambda _: self._toggle_pause())
        step_button.on_click(lambda _: self._advance_frame())
        dark_mode.on_update(lambda _: self._set_dark_mode(dark_mode.value))
        # W, A, S, D, Q, E and the arrow keys are reserved for viser's camera movement
        commands = [
            ("space", "Pause", self._toggle_pause),
            ("tab", "Next camera", self._next_camera),
            ("C", "Contact forces", self._toggle_contacts),
            ("H", "Hide menu", self._toggle_menu),
        ]
        commands += [(str(i), f"Geom group {i}", partial(self._toggle_group, i)) for i in range(5)]
        for key, label, action in commands:
            command = self.server.gui.add_command(label, hotkey=key)
            command.on_trigger(lambda _, action=action: action())
        self._cameras = self.scene._camera_dropdown

    def _draw_splats(self, mocap_pos: NDArray, mocap_quat: NDArray):
        """Move each drone splat to its drone in the shown world."""
        # Drone splats are in the drone body frame, which the drone mocap bodies carry
        ids, world = self._drone_mocap_ids, self.scene.env_idx
        poses = zip(self._drone_splats, mocap_pos[world, ids], mocap_quat[world, ids])
        for splat, pos, quat in poses:
            splat.position, splat.wxyz = pos, quat

    def _draw_markers(self):
        """Replace last frame's markers with one batch per marker shape."""
        for handle in self._marker_handles:
            handle.remove()
        self._marker_handles = []
        # viser sets the thickness per handle, so lines get one batch per thickness
        for i, (thickness, lines) in enumerate(self._markers["lines"].items()):
            points, colors = map(np.concatenate, zip(*lines))
            handle = self.server.scene.add_line_segments(
                f"/markers/lines/{i}", points, colors, thickness=thickness
            )
            self._marker_handles.append(handle)
        # Unit meshes that the marker scales stretch to size
        meshes = {
            "spheres": trimesh.creation.icosphere(2),
            "cylinders": trimesh.creation.cylinder(1.0, 1.0),
        }
        for name, mesh in meshes.items():
            if self._markers[name]:
                pos, wxyz, scale, rgba = map(np.concatenate, zip(*self._markers[name]))
                handle = self.server.scene.add_batched_meshes_simple(
                    f"/markers/{name}",
                    mesh.vertices,
                    mesh.faces,
                    batched_wxyzs=wxyz,
                    batched_positions=pos,
                    batched_scales=scale,
                    batched_colors=rgba[:, :3],
                    batched_opacities=rgba[:, 3],
                )
                self._marker_handles.append(handle)
        for markers in self._markers.values():
            markers.clear()

    def _follow_camera(self, body: RigidTransform):
        """Move every client camera to the selected model camera."""
        cam = self._cameras.options.index(self._cameras.value) - 1
        cam_body = self._mj_model.cam_bodyid[cam]
        pose = body[self.scene.env_idx][cam_body] * self._body_to_cam[cam]
        # MuJoCo cameras look along their -z axis with +y up
        look, up = pose.rotation.apply([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = pose.translation
                client.camera.look_at = pose.translation + look
                client.camera.up_direction = up

    def _set_free_camera(
        self, distance: float, azimuth: float, elevation: float, lookat: list[float]
    ):
        """Move every client camera to a view of MuJoCo's free camera."""
        assert self._cameras.value == "Free", "cam_config needs the free camera"
        # MuJoCo's free camera looks along +x, turned up by the elevation and left by the azimuth
        forward, up = R.from_euler("ZY", [azimuth, -elevation], degrees=True).apply(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = lookat - distance * forward
                client.camera.look_at = lookat
                client.camera.up_direction = up

    def _wait(self, sim_time: float):
        """Hold while paused, then wait until the wall clock catches up with the sim time."""
        while self.paused and not self._advance:
            time.sleep(1 / 60)  # Pause refresh rate
        self._advance = False
        now, speed = time.perf_counter(), self._speed.value
        start_wall, start_sim, start_speed = self._clock
        target = start_wall + (sim_time - start_sim) / speed
        # The fixed start keeps sleep overshoots from adding up. Speed changes, resets, pauses and
        # slow frames restart the clock so the viewer does not race to catch up.
        if speed != start_speed or target < now - 0.2:
            self._clock = (now, sim_time, speed)
            return
        time.sleep(max(0.0, target - now))

    def _toggle_pause(self):
        self.paused = not self.paused
        self._pause_button.label = "Play" if self.paused else "Pause"
        self._pause_button.icon = viser.Icon.PLAYER_PLAY if self.paused else viser.Icon.PLAYER_PAUSE

    def _advance_frame(self):
        self._advance = True
        if not self.paused:
            self._toggle_pause()

    def _next_camera(self):
        options = list(self._cameras.options)
        self._cameras.value = options[(options.index(self._cameras.value) + 1) % len(options)]

    def _toggle_contacts(self):
        show = not self.scene.show_contact_points
        self.scene.show_contact_points = self.scene.show_contact_forces = show
        self.scene.request_update()

    def _toggle_menu(self):
        self._menu.visible = not self._menu.visible

    def _toggle_group(self, group: int):
        checkbox = self.scene._geom_group_checkboxes[group]
        checkbox.value = not checkbox.value

    def _set_dark_mode(self, dark: bool):
        self.server.gui.configure_theme(dark_mode=dark)
        # A dim scene matches the dark theme
        self.server.scene.configure_default_lights(enabled=not dark)
        self.server.scene.configure_environment_map(environment_intensity=0.15 if dark else 1.0)
        for grid, light in self._grids:
            colors = ((15, 15, 18), 1.0, (45, 45, 50), (70, 70, 78)) if dark else light
            grid.plane_color, grid.plane_opacity, grid.cell_color, grid.section_color = colors

    def _show_splats(self, visible: bool):
        self._splats_visible = visible
        for splat in self._splats:
            splat.visible = visible

    def _show_meshes(self, visible: bool):
        """Show or hide the meshes and floor."""
        self._meshes_visible = visible
        nodes = self.server.scene._handle_from_node_name
        for name in ("/bodies", "/fixed_bodies"):
            nodes[name].visible = visible
