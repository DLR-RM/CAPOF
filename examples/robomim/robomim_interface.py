"""Outer control loops for the robomimic Lift use-cases.

This module provides a shared base loop (:class:`RobomimBaseLoop`) with all
machinery common to both use-cases (environment handling, live viewer,
key-based live debugging, live plot, action (de)normalization), plus two thin
subclasses:

- :class:`ForceConditionedLoop`: task parameters are the gripper-cube contact
  forces (single skill).
- :class:`ObjectCentricLoop`: task parameter is the grasp status; the policy
  switches between a grasp skill (object-centric frame) and a lift skill
  (world frame).
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from capof.outer_control_loop import OuterControlLoop
from examples.robomim.live_viewer_utils import (
    KeyListener,
    LiveForcePlot,
    StuckMonitor,
    capture_cube_init_pose,
    get_gripper_cube_normal_forces,
    reset_cube,
)
from examples.robomim.robomim_env import RobomimEnv
from examples.robomim.utils import controller_output_range


class RobomimBaseLoop(OuterControlLoop):
    """Shared outer control loop for robomimic environments.

    Provides:
    - 6D velocity command output (3D linear + 3D angular)
    - Gripper command handling via ``mu_add``
    - Environment interaction (reset, step, render)
    - Optional live MuJoCo viewer with key-based debugging:
      * ``space``: re-place the cube (the sim clock keeps running, so the
        live plot trace continues)
      * ``r``: full env reset with a fresh random seed (the sim clock restarts,
        so the live plot trace is re-based)

    Args:
        X: (n_samples, n_features) demonstration data used to normalize the
            success threshold.
        capof: ContextAdaptivePolicyFramework instance.
        offset: Offset added to the success threshold per dimension.
        seed: Random seed for the environment.
        dataset_path: Path to the robomimic HDF5 dataset.
        R0: (3, 3) EEF home rotation (representation frame).
        control_rate: Control loop frequency in Hz.
        loop_limit: Maximum number of steps per episode (ignored in
            ``live_render`` mode, where the loop runs until the viewer is
            closed).
        threshold: Relative distance threshold for goal-reach detection.
        policy_combination: Policy combination method (e.g. ``"MoE"``).
        record_images: If True, record rendered frames for video export
            (typically disabled in ``live_render`` mode).
        live_render: If True, open a live MuJoCo viewer window.
    """

    #: Live-plot configuration passed to :class:`LiveForcePlot` (live_render
    #: mode only). ``None`` disables the plot.
    PLOT_CONFIG: ClassVar[Optional[dict[str, Any]]] = None
    #: Cube placement ranges forwarded to :class:`RobomimEnv` (see its docs).
    CUBE_XY_RANGE: ClassVar[tuple[float, float]] = (0.0, 0.0)
    CUBE_ROT_RANGE: ClassVar[tuple[float, float]] = (0.0, 0.0)

    def __init__(
        self,
        X: np.ndarray,
        capof: Any,
        offset: np.ndarray,
        seed: int,
        dataset_path: str,
        R0: np.ndarray,
        control_rate: float = 20,
        loop_limit: int = 500,
        threshold: float = 0.01,
        policy_combination: str = "MoE",
        record_images: bool = True,
        live_render: bool = False,
    ) -> None:
        super().__init__(X, capof, offset, loop_limit=loop_limit, threshold=threshold, policy_combination=policy_combination)

        self.loop_time = 1.0 / control_rate
        self.R0 = R0
        self.record_images = record_images
        self.live_render = live_render

        # Create environment
        self.env = RobomimEnv(
            dataset_path=dataset_path,
            seed=seed,
            live_render=live_render,
            cube_xy_range=self.CUBE_XY_RANGE,
            cube_rot_range=self.CUBE_ROT_RANGE,
        )

        output_min, output_max = controller_output_range(self.env.env_meta)
        self.Y_scale = (output_max - output_min) / 2.0
        self.Y_offset = (output_max + output_min) / 2.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _init_episode_state(self) -> None:
        """Initialize per-episode state before the loop (subclasses override)."""

    def run_loop(self) -> None:
        """Run the control loop (main entry point for executing an episode)."""
        # Reset tracking variables
        self.imgs: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.visited_states: list[np.ndarray] = []

        self._init_episode_state()

        # Reset environment and get initial observation
        self.obs, _ = self.env.reset()
        capture_cube_init_pose(self.env)  # remember cube pose for live-debugging resets

        # Initialize tcp_frame
        self.tcp_frame = np.eye(4)

        # Initialize process parameters
        self._parameter_initialization()

        # Start live-debugging aids (live_render only)
        self._key_listener: Optional[KeyListener] = None
        self._stuck_monitor: Optional[StuckMonitor] = None
        self._force_plot: Optional[LiveForcePlot] = None
        if self.live_render:
            self._key_listener = KeyListener(keys=("space", "r"))
            self._key_listener.start()
            self._stuck_monitor = StuckMonitor()
            if self.PLOT_CONFIG is not None:
                self._force_plot = LiveForcePlot(**self.PLOT_CONFIG)

        # Run the state-based execution loop
        self._state_based_execution_loop()

    # ------------------------------------------------------------------
    # Sensor / state readers
    # ------------------------------------------------------------------

    def _read_current_tcp_frame(self) -> None:
        """Read the current TCP frame from the environment observation."""
        self.tcp_frame[:3, 3] = self.obs["robot0_eef_pos"].copy()
        self.tcp_frame[:3, :3] = R.from_quat(self.obs["robot0_eef_quat"]).as_matrix()

        if self.record_images:
            self.imgs.append(self.env.render(mode="rgb_array"))

    def _read_current_hand_pose(self, blocking: bool) -> None:
        """Read the current gripper width from the environment observation."""
        self.hand_pose = self.obs["robot0_gripper_qpos"][0]

    def _read_current_obj_frame(self) -> None:
        """Read the current object frame (4x4) from the environment observation."""
        if self.env.task_name == "Lift":
            pos = self.obs["object"][:3]
            quat = self.obs["object"][3:7]
        else:
            pos = self.obs["object"][7:10]
            quat = self.obs["object"][10:]

        self.obj_frame = np.eye(4)
        self.obj_frame[:3, 3] = pos.copy()
        self.obj_frame[:3, :3] = R.from_quat(quat).as_matrix()

    # ------------------------------------------------------------------
    # Live viewer helpers
    # ------------------------------------------------------------------

    def _viewer_handle(self) -> Optional[Any]:
        """Return the passive viewer Handle, or None if it is not launched yet."""
        if not self.live_render:
            return None
        try:
            mjviewer = self.env._robomim_env.env.viewer  # MjviewerRenderer
            return mjviewer.viewer if mjviewer is not None else None
        except Exception:
            return None

    def _viewer_is_open(self) -> bool:
        """Check whether the live MuJoCo viewer window is still open.

        The passive viewer (mjviewer) is synced automatically in env.step();
        once the user closes the window, PassiveViewer.is_running() returns False.
        """
        handle = self._viewer_handle()
        return True if handle is None else handle.is_running()

    # ------------------------------------------------------------------
    # Loop generation
    # ------------------------------------------------------------------

    def _gen(self):
        """Generate frame indices until a stop condition is reached.

        In ``live_render`` mode the loop limit is ignored (run until the
        viewer window is closed) and key presses are handled here.
        """
        i = 0
        limit = None if self.live_render else self.loop_limit
        while not self.success and not self.stop_activated and (limit is None or i < limit):
            if self.live_render:
                if not self._viewer_is_open():
                    self.stop_activated = True
                    break
                if self._key_listener is not None and self._key_listener.was_pressed("space"):
                    # Re-place the cube; the sim clock keeps running, so the
                    # live plot trace continues
                    self._key_listener.clear("space")
                    reset_cube(self.env)
                if self._key_listener is not None and self._key_listener.was_pressed("r"):
                    # Full env reset with a fresh seed so the reset samples a
                    # new initial state instead of restoring the cached one;
                    # the sim clock restarts, so re-base the live plot
                    self._key_listener.clear("r")
                    self.obs, _ = self.env.reset(np.random.randint(0, 100))
                    if self._force_plot is not None:
                        self._force_plot.reset()
                if self._stuck_monitor is not None:
                    self._stuck_monitor.check(self.env)
            i += 1
            yield i

    # ------------------------------------------------------------------
    # Data (pre/post)processing
    # ------------------------------------------------------------------

    def _preprocess_pose(self) -> None:
        """Hook: transform the TCP frame into the representation frame.

        Default: no transform. Subclasses override (e.g. object-centric frame).
        """

    def _postprocess_pose(self) -> None:
        """Hook: transform the velocity twist / TCP frame back to the world frame.

        Default: no transform. Subclasses override (e.g. object-centric frame).
        """

    def _preprocess_data(self) -> None:
        """Preprocess the observation for the policy input.

        Applies the subclass pose transform, then brings the orientation into
        the representation frame (relative to ``R0``).
        """
        self._preprocess_pose()

        if self.capof.phi_dot_dim != 0:
            # Bring to representation frame
            self.tcp_frame[:3, :3] = np.linalg.inv(self.R0) @ self.tcp_frame[:3, :3]

    def _postprocess_data(self) -> None:
        """Postprocess the policy output.

        Maps the orientation velocity and TCP frame back to the world frame,
        applies the subclass inverse pose transform, updates the loop timing,
        and (if the policy provides an additional output) adopts the gripper
        command from ``mu_add``.
        """
        if self.capof.phi_dot_dim != 0:
            # Map orientation velocity to data coordinate frame
            self.vel_twist[3:6] = self.R0 @ self.vel_twist[3:6]

            # Transform TCP frame back to world frame
            self.tcp_frame[:3, :3] = self.R0 @ self.tcp_frame[:3, :3]

        self._postprocess_pose()

        self.dt = time.time() - self.loop_time
        self.loop_time = time.time()

        if len(self.mu_add) != 0:
            self.hand_pose = self.mu_add

    # ------------------------------------------------------------------
    # Safety / publishing
    # ------------------------------------------------------------------

    def _check_for_limits(self) -> None:
        """Clip the gripper command to the valid range."""
        self.hand_pose = np.clip(self.hand_pose, -1.0, 1.0)

    def _publish_vel_info(self) -> None:
        """Log the LfD input and feed the live plot (live_render only)."""
        self.visited_states.append(self.lfd_input.copy())

        if self._force_plot is not None:
            sim_time = self.env._robomim_env.env.sim.data._data.time
            self._force_plot.push(sim_time, self.task_parameters)
            self._force_plot.update(self._viewer_handle())

    # ------------------------------------------------------------------
    # Actuation / shutdown
    # ------------------------------------------------------------------

    def _send_command_to_robot(self) -> None:
        """Send the velocity command to the environment.

        Combines the 6D velocity twist with the gripper command, normalizes
        the 6D part into the environment action space, and steps the env.
        """
        action = np.concatenate([self.vel_twist, self.hand_pose])

        # Normalize actions (they were denormalized for the policy combination)
        action[:6] = (action[:6] - self.Y_offset) / self.Y_scale

        self.obs, reward, done, info = self.env.step(action)
        self.rewards.append(reward)

    def _safe_shutdown(self) -> None:
        """Stop live-debugging aids (safe to call multiple times)."""
        key_listener = getattr(self, "_key_listener", None)
        if key_listener is not None:
            key_listener.stop()
            self._key_listener = None
        self._stuck_monitor = None

    def _check_for_external_stop(self) -> None:
        """No external stop mechanism in simulation."""


class ForceConditionedLoop(RobomimBaseLoop):
    """Control loop for the force-conditioned use-case (single skill).

    The task parameters are the two gripper-cube contact forces (normal
    forces on the finger pads), which condition the policy on the grasp
    force. The cube starts at a fixed pose.
    """

    PLOT_CONFIG = {}  # LiveForcePlot defaults: two force lines, y in [0, 30] N
    CUBE_XY_RANGE = (0.0, 0.0)
    CUBE_ROT_RANGE = (0.0, 0.0)

    def _init_episode_state(self) -> None:
        """Initialize the (zero) contact-force task parameters for a new episode."""
        self.task_parameters = np.array([0.0, 0.0])

    def _read_current_task_parameters(self, blocking: bool) -> None:
        """Read the current gripper-cube contact forces (N) from the sim."""
        forces = get_gripper_cube_normal_forces(self.env._robomim_env)
        self.task_parameters = np.array([forces["finger1"], forces["finger2"]])


class ObjectCentricLoop(RobomimBaseLoop):
    """Control loop for the object-centric use-case (two skills).

    The task parameter is the binary grasp status. While the object is not
    grasped, the grasp skill runs in a symmetry-canonicalized object-centric
    frame (anchored at the cube, tracking it as it moves); after the grasp,
    the lift skill runs in the world frame. The cube starts at a random pose.
    """

    PLOT_CONFIG = {"line_names": ("grasp",), "title": "grasp", "y_range": (-0.1, 1.1)}
    CUBE_XY_RANGE = (-0.15, 0.15)
    CUBE_ROT_RANGE = (-np.pi, np.pi)

    def __init__(
        self,
        Xs: list[np.ndarray],
        capofs: list[Any],
        offset: np.ndarray,
        seed: int,
        dataset_path: str,
        R0s: list[np.ndarray],
        control_rate: float = 20,
        loop_limit: int = 500,
        threshold: float = 0.01,
        policy_combination: str = "MoE",
        record_images: bool = True,
        live_render: bool = False,
    ) -> None:
        """Create the two-skill loop.

        Like :class:`RobomimBaseLoop`, but ``Xs`` / ``capofs`` / ``R0s`` are
        lists of two entries (grasp skill, lift skill); the base loop is
        initialized with the grasp-skill entries and switches online by grasp
        status (see :meth:`_preprocess_pose`).
        """
        super().__init__(
            Xs[0],
            capofs[0],
            offset,
            seed,
            dataset_path,
            R0s[0],
            control_rate=control_rate,
            loop_limit=loop_limit,
            threshold=threshold,
            policy_combination=policy_combination,
            record_images=record_images,
            live_render=live_render,
        )

        self.object_centric = True
        self.capofs = capofs
        self.Xs = Xs
        self.R0s = R0s

    def _init_episode_state(self) -> None:
        """Start each episode in the (not-yet) grasped state."""
        self.grasp_status = False

    def _read_current_task_parameters(self, blocking: bool) -> None:
        """Read the current grasp status (0 = not grasped, 1 = grasped)."""
        self.task_parameters = np.array([self.grasp_status], dtype=int)

    def _compute_alignment(self) -> None:
        """Compute the symmetry-canonicalized object-centric frame.

        The cube has 4-fold z-symmetry (90 deg). We rotate the world by the
        residual (<= 45 deg) so that the cube's orientation matches the
        nearest 90-deg multiple, and anchor the frame at the cube's current
        position. This mirrors the training-time transform in
        :mod:`examples.robomim.data_processing`, keeps the EEF orientation within
        +/-45 deg of the home pose (avoiding the rotvec singularity), and
        makes the policy invariant to the cube's pose at inference.
        Recomputed every iteration so the frame tracks a moving object.
        """
        self._read_current_obj_frame()
        Q0 = self.obj_frame[:3, :3]
        self.p_cube0 = self.obj_frame[:3, 3].copy()

        S_targets = [R.from_euler("z", j * 90, degrees=True).as_matrix() for j in range(4)]
        angles = [R.from_matrix(S @ Q0.T).magnitude() for S in S_targets]
        self.R_align = S_targets[int(np.argmin(angles))] @ Q0.T

        # World -> object-centric frame as a 4x4 homogeneous transform
        self.T_rep = np.eye(4)
        self.T_rep[:3, :3] = self.R_align
        self.T_rep[:3, 3] = -self.R_align @ self.p_cube0
        self.T_rep_inv = np.linalg.inv(self.T_rep)

    def _preprocess_pose(self) -> None:
        """Switch skill by grasp status and transform into the object frame."""
        if self.grasp_status:
            self.capof = self.capofs[1]
            self.X = self.Xs[1]
            self.object_centric = False
            self.R0 = self.R0s[1]
        else:
            self.capof = self.capofs[0]
            self.X = self.Xs[0]
            self.object_centric = True
            self.R0 = self.R0s[0]

        if self.object_centric:
            # Re-read the current object frame and recompute the alignment
            self._compute_alignment()
            # Transform EEF into the object-centric frame (4x4 product)
            self.tcp_frame = self.T_rep @ self.tcp_frame

    def _postprocess_pose(self) -> None:
        """Transform the twist / TCP frame from the object frame back to world."""
        if self.object_centric:
            # Transform twist from object-centric frame back to world frame
            self.vel_twist[0:3] = self.R_align.T @ self.vel_twist[0:3]
            self.vel_twist[3:6] = self.R_align.T @ self.vel_twist[3:6]

            # Transform TCP frame from object-centric frame back to world frame
            self.tcp_frame = self.T_rep_inv @ self.tcp_frame

    def _send_command_to_robot(self) -> None:
        """Step the environment and update the grasp status (skill-switch input)."""
        super()._send_command_to_robot()

        # Update the grasp status from the environment's contact check
        self.grasp_status = self.env._robomim_env.env._check_grasp(
            gripper=self.env._robomim_env.env.robots[0].gripper["right"],
            object_geoms=self.env._robomim_env.env.cube.contact_geoms,
        )

    def _check_for_success(self) -> None:
        """No goal-reach success check: success is defined by the env reward.

        The base class would stop the loop after 10 frames near a goal pose,
        which is not meaningful for the two-skill lift task; the loop instead
        runs until ``loop_limit`` (or the viewer is closed in live mode).
        """
