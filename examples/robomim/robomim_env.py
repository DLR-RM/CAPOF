"""Gymnasium environment wrapper for robomimic tasks.

This module wraps a robomimic environment (e.g. the robosuite *Lift* task) in a
Gymnasium-style interface (``reset`` / ``step`` / ``render``) that the CAPOF
outer control loop can drive without knowing robomimic internals.

The wrapper provides:
- Raw robomimic observations (a concatenated low-dim vector, see ``obs_keys``).
- A normalized action space ``[-1, 1]``; the interface layer converts the CAPOF
  velocity twist into this space using the controller output bounds.
- Optional randomization of the initial EEF pose (deterministic per seed).
- Optional live (on-screen) rendering via the MuJoCo passive viewer.

The cube placement range is a constructor parameter so that the two use-cases
can share this class:

- force-conditioned: the cube always starts at one fixed pose
  (``cube_xy_range=(0.0, 0.0)``, ``cube_rot_range=(0.0, 0.0)``).
- object-centric: the cube starts at a random pose
  (``cube_xy_range=(-0.15, 0.15)``, ``cube_rot_range=(-pi, pi)``).
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import mujoco
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R


class RobomimEnv(gym.Env):
    """Gymnasium environment wrapper for robomimic tasks.

    Args:
        dataset_path: Path to the robomimic HDF5 dataset the environment is
            derived from (its metadata defines the task, controller, etc.).
        seed: Random seed for environment initialization.
        render_size: Edge length (px) of rendered images in ``rgb_array`` mode.
        obs_keys: Observation keys to concatenate into the low-dim observation.
        render_camera_name: Camera used for ``rgb_array`` rendering.
        camera_distance: How far to pull the render camera back along its line
            of sight (m). A wider view with less perspective distortion than
            raising ``cam_fovy``.
        live_render: If True, open an on-screen MuJoCo viewer window; otherwise
            render offscreen only.
        cube_xy_range: ``(min, max)`` range for randomizing the cube's initial
            x/y position. Use ``(0.0, 0.0)`` for a fixed position.
        cube_rot_range: ``(min, max)`` range (rad) for randomizing the cube's
            initial z-rotation. Use ``(0.0, 0.0)`` for a fixed orientation.

    Observation space:
        Concatenation of ``obs_keys`` (default: object pose + EEF pose +
        gripper width, 19D total). Values are raw (not normalized).

    Action space:
        Box in ``[-1, 1]`` with the environment's action dimension (7D for
        OSC_POSE: 6D Cartesian delta + 1D gripper).
    """

    def __init__(
        self,
        dataset_path: str = "",
        seed: int = 42,
        render_size: int = 256,
        obs_keys: Optional[list[str]] = None,
        render_camera_name: str = "agentview",
        camera_distance: float = 1.0,
        live_render: bool = False,
        cube_xy_range: tuple[float, float] = (0.0, 0.0),
        cube_rot_range: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.dataset_path = dataset_path
        self.render_size = render_size
        self.obs_keys = (
            obs_keys if obs_keys is not None else ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        )
        self.render_camera_name = render_camera_name
        self.camera_distance = camera_distance
        self.live_render = live_render

        # Track episode state
        self._episode_ended = False
        self._current_step = 0
        self._seed: Optional[int] = None
        # Cache of reset states per seed so re-resetting with the same seed is
        # cheap (robosuite resets are expensive).
        self.seed_state_map: dict[int, Any] = {}

        # Initialize obs modality mapping before creating the env
        ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": self.obs_keys, "rgb": ["agentview_image"]})

        # Create the robomimic environment
        self._robomim_env = self._create_environment(dataset_path)
        self.seed(seed)

        # Define observation and action spaces
        self._setup_spaces()

        # Base render-camera position (from the model XML; constant across resets).
        # Used to apply camera_distance idempotently in _configure_camera().
        model = self._robomim_env.env.sim.model._model
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.render_camera_name)
        self._cam_base_pos = model.cam_pos[cam_id].copy()

        # Configure cube placement (fixed or randomized, per use-case)
        placement = self._robomim_env.env.placement_initializer
        placement.x_range = [float(cube_xy_range[0]), float(cube_xy_range[1])]
        placement.y_range = [float(cube_xy_range[0]), float(cube_xy_range[1])]
        placement.rotation = [float(cube_rot_range[0]), float(cube_rot_range[1])]

    def _create_environment(self, dataset_path: str):
        """Create a robomimic environment from the dataset's environment metadata.

        Args:
            dataset_path: Path to the robomimic HDF5 dataset.

        Returns:
            The created robomimic environment.
        """
        self.env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        # Disable joint-space initialization noise; the initial EEF pose is
        # randomized in cartesian space instead (see _randomize_eef_pose).
        self.env_meta["env_kwargs"]["initialization_noise"] = None

        env = EnvUtils.create_env_from_metadata(
            env_meta=self.env_meta,
            render=self.live_render,
            render_offscreen=not self.live_render,
        )

        self.task_name: str = self.env_meta["env_name"]

        return env

    def _setup_spaces(self) -> None:
        """Set up observation and action spaces based on environment capabilities."""
        # Action space: normalized [-1, 1] with environment-specific dimension.
        # The interface layer converts 7D velocity commands into this space.
        action_dim = self._robomim_env.action_dimension
        low = np.full(action_dim, fill_value=-1.0, dtype=np.float32)
        high = np.full(action_dim, fill_value=1.0, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Get an example observation to determine the observation space
        # (requires one reset to produce a valid observation).
        self._robomim_env.reset()
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1.0, dtype=np.float32)
        high = np.full_like(obs_example, fill_value=1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def seed(self, seed: Optional[int] = None) -> None:
        """Set the random seed (numpy global state, as used by robosuite)."""
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to its initial state.

        Args:
            seed: Optional random seed. If given, the environment is reset to
                the (cached) state of that seed, so the same seed always yields
                the same initial state.
            options: Unused (Gymnasium API compatibility).

        Returns:
            Tuple of (initial observation, info dictionary).
        """
        if seed is not None:
            self.seed(seed)

        self._episode_ended = False
        self._current_step = 0

        if self._seed is not None:
            # Reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # Use cached state to avoid an expensive reset
                self._robomim_env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite uses the numpy global random state
                self.seed(seed)
                self._robomim_env.reset()
                self._randomize_eef_pose()

                state = self._robomim_env.get_state()["states"]
                self.seed_state_map[seed] = state
        else:
            # Random reset
            self._robomim_env.reset()
            self._randomize_eef_pose()

        return self._robomim_env.get_observation(), self._get_info()

    def _randomize_eef_pose(self) -> None:
        """Sample and apply a randomized initial EEF pose (deterministic per seed)."""
        env = self._robomim_env.env
        target_pos, target_rot = self.sample_target_pose(env, pos_delta=(0.3, 0.3, 0.05), rot_delta_deg=(30, 30, 30))
        self.set_panda_eef_pose(env, target_pos, target_rot)

    def sample_target_pose(
        self,
        env: Any,
        pos_delta: tuple[float, float, float] = (0.1, 0.1, 0.1),
        rot_delta_deg: tuple[float, float, float] = (10.0, 10.0, 10.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a target EEF pose uniformly around the current EEF pose.

        Args:
            env: robosuite environment.
            pos_delta: Maximum absolute XYZ displacement in meters.
            rot_delta_deg: Maximum absolute XYZ rotation in degrees.

        Returns:
            Tuple of (target position (3,), target rotation matrix (3, 3)).
        """
        site_id = env.robots[0].eef_site_id["right"]
        initial_pos = env.sim.data.site_xpos[site_id].copy()
        initial_rot = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()

        # Uniform Cartesian displacement
        pos_delta_arr = np.asarray(pos_delta)
        dpos = np.random.uniform(-pos_delta_arr, pos_delta_arr)
        target_pos = initial_pos + dpos

        # Uniform rotation perturbation about the initial EEF orientation
        rot_delta_rad = np.deg2rad(np.asarray(rot_delta_deg))
        drot = R.from_euler("xyz", np.random.uniform(-rot_delta_rad, rot_delta_rad))
        target_rot = initial_rot @ drot.as_matrix()

        return target_pos, target_rot

    def set_panda_eef_pose(
        self,
        env: Any,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        max_iter: int = 50,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
    ) -> None:
        """Move the robot so that its EEF reaches the requested Cartesian pose.

        Solves inverse kinematics with an iterative damped least-squares scheme
        on the arm joints, then writes the resulting joint positions. The OSC
        controller is re-synced so it starts from the new configuration instead
        of driving the EEF back to the previous goal.

        Args:
            env: robosuite / robomimic environment.
            target_pos: (3,) position in world coordinates.
            target_rot: (3, 3) rotation matrix in world coordinates.
            max_iter: Maximum number of IK iterations.
            pos_tol: Position convergence threshold (m).
            rot_tol: Rotation convergence threshold (rad).
        """
        robot = env.robots[0]
        sim = env.sim
        model, data = sim.model._model, sim.data._data

        site_id = robot.eef_site_id["right"]
        joint_idx = robot._ref_joint_pos_indexes  # arm joint qpos indices

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        damping = 1e-2

        for _ in range(max_iter):
            cur_pos = data.site_xpos[site_id]
            cur_rot = data.site_xmat[site_id].reshape(3, 3)

            # Task-space error (world frame)
            dpos = target_pos - cur_pos
            drot = R.from_matrix(target_rot @ cur_rot.T).as_rotvec()
            error = np.concatenate([dpos, drot])

            # Damped least-squares joint update for the arm joints
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            J = np.vstack([jacp, jacr])[:, joint_idx]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), error)

            data.qpos[joint_idx] += dq
            data.qvel[:] = 0.0
            sim.forward()

            if np.linalg.norm(dpos) < pos_tol and np.linalg.norm(drot) < rot_tol:
                break

        # Re-sync the OSC controller with the new configuration (also resets
        # its Cartesian goal to the current EEF pose)
        robot.part_controllers["right"].update_initial_joints(data.qpos[joint_idx])

    def step(self, action: np.ndarray):
        """Take a step in the environment.

        Args:
            action: Normalized action in ``[-1, 1]``.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        self._current_step += 1

        raw_obs, reward, done, info = self._robomim_env.step(action)
        self._episode_ended = done

        return raw_obs, reward, done, info

    def get_observation(self) -> np.ndarray:
        """Get the observation by concatenating the configured ``obs_keys``.

        Returns:
            Concatenated observation vector.
        """
        raw_obs = self._robomim_env.get_observation()
        return np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)

    def _get_info(self) -> dict[str, Any]:
        """Get additional information about the current state.

        Returns:
            Dictionary with task name, step count, and episode-ended flag.
        """
        return {
            "task_name": self.task_name,
            "step": self._current_step,
            "episode_ended": self._episode_ended,
        }

    def _configure_camera(self) -> None:
        """Pull the render camera back along its line of sight for a wider view.

        The camera looks along its local -z axis, so its +z axis points away
        from the scene; translating the camera along +z increases the distance
        to the scene without widening the field of view (which is what causes
        the barrel distortion of a large ``cam_fovy``). Applied idempotently
        from the base position, so it is safe to call on every render.
        """
        if self.camera_distance == 0:
            return

        sim = self._robomim_env.env.sim
        model = sim.model._model
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.render_camera_name)
        quat = model.cam_quat[cam_id]  # wxyz
        back_dir = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()[:, 2]
        model.cam_pos[cam_id] = self._cam_base_pos + back_dir * self.camera_distance

        # cam_pos only takes effect in the rendered image after a forward pass
        # recomputes the camera's world pose (data.cam_xpos). Without this the
        # first frame (rendered before any env.step) would use the stale pose.
        sim.forward()

    def render(self, mode: str = "rgb_array"):
        """Render the environment.

        Args:
            mode: ``"rgb_array"`` (returns an image) or ``"human"``.

        Returns:
            Rendered image (``rgb_array`` mode) or None (``human`` mode).
        """
        self._configure_camera()

        if mode == "rgb_array":
            h, w = self.render_size, self.render_size
            return self._robomim_env.render(mode=mode, height=h, width=w, camera_name=self.render_camera_name)

        elif mode == "human":
            return self._robomim_env.render(mode=mode)

        return None

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if hasattr(self._robomim_env, "close"):
            self._robomim_env.close()
        elif hasattr(self._robomim_env, "env"):
            self._robomim_env.env.close()

    def set_state(self, state: Any) -> None:
        """Set the environment to a specific state (for reproducibility)."""
        if hasattr(self._robomim_env, "set_state"):
            self._robomim_env.set_state(state)

    def get_state(self) -> Optional[Any]:
        """Get the current environment state (for reproducibility)."""
        if hasattr(self._robomim_env, "get_state"):
            return self._robomim_env.get_state()
        return None
