"""Demo-data loading and feature construction for the robomimic Lift use-cases.

The demonstration dataset (``datasets/lift_ph_v15.hdf5``) is **not** shipped
with this repository. It is generated from the official robomimic v1.5 *Lift*
(PH model) low-dim dataset (``low_dim_v15.hdf5``) by replaying the demos in
the robosuite simulation with :mod:`examples.robomim.convert_dataset`; see
the README for the full instructions.

Both use-cases load the same robomimic *Lift* HDF5 demonstration format and
build the ``(X, Y)`` feature matrices that the Gaussian process is trained on:

- **Force-conditioned** (single skill):
  ``X = [contact forces (2), EEF position (3), EEF rotvec in home frame (3)]``
  and ``Y = [twist in home frame (6), gripper (1)]``. Each demo is rigidly
  aligned so the cube's initial pose matches the canonical inference pose
  (symmetry-aware, see below).

- **Object-centric** (two skills, split by grasp status):
  the grasp skill is expressed in a symmetry-canonicalized object-centric
  frame (anchored at the cube), the lift skill in the world frame.

The cube has 4-fold z-symmetry (90 deg). Instead of rotating a demo by the
cube's full yaw, we rotate by the *residual* (<= 45 deg) to the nearest 90-deg
multiple. This keeps the EEF orientation within +/-45 deg of the home pose, so
the rotvec relative to the home pose stays small and avoids the +/-180 deg
rotvec singularity.

Public entry points (used by the entry scripts *and* the BBO black-box
functions, so there is a single source of truth for the data pipeline):

- :func:`load_force_conditioned_data`
- :func:`load_object_centric_data`
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import robomimic.utils.file_utils as FileUtils
from scipy.spatial.transform import Rotation as R

from examples.robomim.utils import controller_output_range

# Canonical cube position at inference time for the force-conditioned use-case
# (the cube is placed at a fixed pose: x = y = 0, z from the environment model).
FC_CUBE_HOME_POS = np.array([0.0, 0.0, 0.83082212])

# The augmented dataset is generated from the official robomimic dataset
# (not shipped with this repository)
ROBOMIMIC_DATASET_URL = "https://huggingface.co/datasets/robomimic/robomimic_datasets/tree/main/v1.5/lift/ph"
# Observation keys added by the conversion (examples/robomim/convert_dataset.py)
REQUIRED_OBS_KEYS = ("normal_forces", "grasp_status")


def check_robomim_dataset(dataset_path: str | Path) -> Path:
    """Validate the location of the converted robomimic demonstration dataset.

    Args:
        dataset_path: Path to the converted robomimic HDF5 dataset.

    Returns:
        The validated path as a :class:`~pathlib.Path`.

    Raises:
        FileNotFoundError: If the file is missing or does not contain the
            observation keys added by the conversion. The message explains how
            to obtain the original dataset and convert it.
    """
    dataset_path = Path(dataset_path)
    convert_command = "python examples/robomim/convert_dataset.py path/to/low_dim_v15.hdf5"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Robomimic demonstration dataset not found at {dataset_path}.\n"
            f"It is not distributed with this repository. Download the official\n"
            f"robomimic v1.5 Lift (PH) low-dim dataset (file: low_dim_v15.hdf5) from\n"
            f"    {ROBOMIMIC_DATASET_URL}\n"
            f"and convert it with:\n"
            f"    {convert_command}\n"
            f"(or pass --dataset pointing at your own converted file)."
        )
    with h5py.File(dataset_path, "r") as f:
        group = f["data"] if "data" in f else f
        demo_keys = [k for k in group.keys() if k.startswith("demo_")]
        obs_group = group[demo_keys[0]].get("obs", group[demo_keys[0]]) if demo_keys else None
        missing = [k for k in REQUIRED_OBS_KEYS if obs_group is None or k not in obs_group]
    if missing:
        raise FileNotFoundError(
            f"{dataset_path} does not contain the observation key(s): {', '.join(missing)}.\n"
            f"It does not look like a converted CAPOF demonstration dataset.\n"
            f"Convert the official robomimic v1.5 Lift (PH) low-dim dataset with:\n"
            f"    {convert_command}"
        )
    return dataset_path


def symmetry_targets() -> list[np.ndarray]:
    """Return the 4-fold z-symmetry rotations of the cube (0/90/180/270 deg)."""
    return [R.from_euler("z", j * 90, degrees=True).as_matrix() for j in range(4)]


def rotvec_in_representation_base(rotation_matrices: np.ndarray, R0: np.ndarray) -> np.ndarray:
    """Express a batch of rotations as rotvecs relative to the home rotation ``R0``.

    Args:
        rotation_matrices: (n, 3, 3) rotation matrices.
        R0: (3, 3) home rotation.

    Returns:
        (n, 3) rotation vectors of ``R0^{-1} @ R_i``.
    """
    R0_inv = np.linalg.inv(R0)
    R_in_base = np.array([R0_inv @ Rm for Rm in rotation_matrices])
    return R.from_matrix(R_in_base).as_rotvec()


def vel_twists_in_representation_base(vel_twists: np.ndarray, R0: np.ndarray) -> np.ndarray:
    """Map the orientation part of a batch of twists into the home frame.

    Args:
        vel_twists: (n, 6) twists ``[v, omega]``.
        R0: (3, 3) home rotation.

    Returns:
        (n, 6) copy of the input with ``omega`` rotated by ``R0^{-T}``.
    """
    out = vel_twists.copy()
    out[:, 3:6] = vel_twists[:, 3:6] @ np.linalg.inv(R0).T
    return out


def to_object_centric_frame(
    frames: np.ndarray, obj_frames: np.ndarray, actions: np.ndarray, S_targets: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform world-frame EEF frames/actions into the object-centric frame.

    For each frame, the object-centric frame is anchored at the object position
    and oriented by the residual rotation ``R_align`` (<= 45 deg) that brings
    the object's orientation onto the nearest 90-deg multiple (the cube has
    4-fold z-symmetry). This makes the representation invariant to the object's
    pose and keeps the EEF orientation within +/-45 deg of the home pose (no
    rotvec singularity).

    Args:
        frames: (n, 4, 4) EEF frames in the world frame.
        obj_frames: (n, 4, 4) object frames in the world frame.
        actions: (n, >=6) actions whose first 6 columns are the twist.
        S_targets: Symmetry target rotations from :func:`symmetry_targets`.

    Returns:
        Tuple of:
        - p_rep: (n, 3) EEF position in the object-centric frame.
        - R_rep: (n, 3, 3) EEF orientation in the object-centric frame.
        - vel_rep: (n, 6) twist ``[v, omega]`` in the object-centric frame.
    """
    n = len(frames)
    p_rep = np.zeros((n, 3))
    R_rep = np.zeros((n, 3, 3))
    vel_rep = np.zeros((n, 6))

    for idx in range(n):
        # Object orientation and position at this frame
        Q0 = obj_frames[idx, :3, :3]
        p_obj = obj_frames[idx, :3, 3]

        # Pick the 90-deg symmetric target closest to the object orientation;
        # R_align is the (<= 45 deg) residual rotation that canonicalizes it
        angles = [R.from_matrix(S @ Q0.T).magnitude() for S in S_targets]
        R_align = S_targets[int(np.argmin(angles))] @ Q0.T

        # EEF position and orientation in the object-centric frame
        p_rep[idx] = R_align @ (frames[idx, :3, 3] - p_obj)
        R_rep[idx] = R_align @ frames[idx, :3, :3]

        # Twist (linear + angular part) in the object-centric frame
        vel_rep[idx, :3] = R_align @ actions[idx, :3]
        vel_rep[idx, 3:6] = R_align @ actions[idx, 3:6]

    return p_rep, R_rep, vel_rep


# ---------------------------------------------------------------------------
# Shared episode-level extraction
# ---------------------------------------------------------------------------


def _iter_episodes(group: h5py.Group) -> Iterator[tuple[str, h5py.Group, h5py.Group]]:
    """Yield ``(episode_key, episode_group, obs_group)`` for all demos in a group."""
    ep_keys = [k for k in group.keys() if k.startswith("demo_")]
    ep_keys.sort(key=lambda x: int(x.split("_")[-1]))
    for ep_key in ep_keys:
        ep_group = group[ep_key]
        obs_group = ep_group["obs"] if "obs" in ep_group else ep_group
        yield ep_key, ep_group, obs_group


def _first_available(obs_group: h5py.Group, keys: tuple[str, ...]) -> Optional[np.ndarray]:
    """Return the first dataset found in ``obs_group`` among ``keys`` (or None)."""
    for key in keys:
        if key in obs_group:
            return obs_group[key][:]
    return None


def _extract_eef(obs_group: h5py.Group) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract EEF position and quaternion, trying common robomimic key names."""
    eef_pos = _first_available(obs_group, ("robot0_eef_pos", "eef_pos", "ee_pos", "robot0_eef_xpos"))
    eef_quat = _first_available(obs_group, ("robot0_eef_quat", "eef_quat", "ee_rot", "robot0_eef_xquat"))
    return eef_pos, eef_quat


def _extract_object(obs_group: h5py.Group, task_name: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract object position and quaternion.

    The layout differs by task: for *Lift* the object state is
    ``[pos (3), quat (4), ...]``; for other tasks the gripper state comes
    first, so the object state starts at index 7.
    """
    if "object" not in obs_group:
        return None, None
    obj_data = obs_group["object"][:]
    if task_name == "Lift":
        return obj_data[:, :3], obj_data[:, 3:7]
    return obj_data[:, 7:10], obj_data[:, 10:]


def _pose_to_frames(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Build (n, 4, 4) homogeneous frames from positions and quaternions."""
    n = len(pos)
    frames = np.zeros((n, 4, 4))
    frames[:, :3, 3] = pos
    frames[:, :3, :3] = R.from_quat(quat).as_matrix()
    frames[:, 3, 3] = 1.0
    return frames


def _load_actions(ep_group: h5py.Group, Y_scale: np.ndarray, Y_offset: np.ndarray) -> Optional[np.ndarray]:
    """Extract, denormalize, and gripper-ramp the actions of one episode.

    Args:
        ep_group: Episode group (must contain an ``actions`` dataset).
        Y_scale: Half-range of the controller output (denormalization scale).
        Y_offset: Midpoint of the controller output (denormalization offset).

    Returns:
        (n, 7) denormalized actions ``[delta_pos, delta_rot, gripper]``, or
        None if the episode has no actions.
    """
    if "actions" not in ep_group:
        return None

    ep_actions = ep_group["actions"][:]

    # Denormalize the 6D delta part (the gripper stays normalized)
    ep_actions[:, :6] = ep_actions[:, :6] * Y_scale + Y_offset

    # Ensure actions are 7D (pad with a zero gripper command if 6D)
    if ep_actions.shape[1] == 6:
        gripper_cmd = np.zeros((ep_actions.shape[0], 1))
        ep_actions = np.concatenate([ep_actions, gripper_cmd], axis=1)

    # Replace the single-frame close step with a linear closing ramp anchored
    # to the end of the grasp phase (GP-friendly target)
    grasp_start = np.argmax(ep_actions[:, -1] == 1)
    ep_actions[grasp_start - 6 : grasp_start, -1] = np.linspace(-1, 1, 6)

    return ep_actions


def _episode_indices(lengths: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-episode start/end indices from the concatenated lengths."""
    from itertools import accumulate

    indices_start_points = np.array([0] + list(accumulate(lengths[:-1])))
    indices_end_points = np.array([x - 1 for x in accumulate(lengths)])
    return indices_start_points, indices_end_points


# ---------------------------------------------------------------------------
# Use-case data builders
# ---------------------------------------------------------------------------


def load_force_conditioned_data(dataset_path: str) -> dict[str, Any]:
    """Load the force-conditioned demo data and build the (X, Y) feature matrices.

    Each demo is rigidly aligned so that the cube's initial pose matches the
    nearest symmetric equivalent of the canonical inference pose
    (:data:`FC_CUBE_HOME_POS`, identity orientation).

    Args:
        dataset_path: Path to the robomimic HDF5 dataset
            (e.g. ``examples/robomim/datasets/lift_ph_v15.hdf5``).

    Returns:
        Dictionary with:
        - ``X``: (n, 8) features ``[forces, eef_pos, eef_rotvec_home]``.
        - ``Y``: (n, 7) targets ``[twist_home, gripper]``.
        - ``goal_poses``: (n_demos, 8) X rows of the last frame of each demo.
        - ``R0``: (3, 3) EEF home rotation (first frame of the first demo).
        - ``Y_scale`` / ``Y_offset``: controller output half-range / midpoint.
        - ``dataset_path``: the input path (for environment creation).

    Raises:
        FileNotFoundError: If the dataset is missing or was not converted
            (see :func:`check_robomim_dataset`).
    """
    check_robomim_dataset(dataset_path)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    task_name = env_meta["env_name"]
    output_min, output_max = controller_output_range(env_meta)
    Y_scale = (output_max - output_min) / 2.0
    Y_offset = (output_max + output_min) / 2.0

    frames_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    forces_list: list[np.ndarray] = []
    lengths: list[int] = []

    S_targets = symmetry_targets()

    with h5py.File(dataset_path, "r") as f:
        group = f["data"] if "data" in f else f

        for ep_key, ep_group, obs_group in _iter_episodes(group):
            eef_pos, eef_quat = _extract_eef(obs_group)
            if eef_pos is None or eef_quat is None:
                print(f"Warning: Could not find EEF position/quat in {ep_key}")
                continue

            obj_pos, obj_quat = _extract_object(obs_group, task_name)
            if obj_pos is None or obj_quat is None:
                print(f"Warning: Could not find object data in {ep_key}")
                continue

            ep_actions = _load_actions(ep_group, Y_scale, Y_offset)
            if ep_actions is None:
                print(f"Warning: Could not find actions in {ep_key}")
                continue

            if "normal_forces" not in obs_group:
                print(f"Warning: Could not find normal_forces in {ep_key}")
                continue
            ep_forces = obs_group["normal_forces"][:]

            ep_frames = _pose_to_frames(eef_pos, eef_quat)

            # Symmetry-aware rigid cube offset: rotate + translate the demo so
            # that the initial cube pose matches the NEAREST symmetric
            # equivalent of the canonical inference pose. The cube has 4-fold
            # z-symmetry (90 deg), so we rotate only by the residual (<= 45
            # deg) instead of the full cube yaw. This keeps the EEF orientation
            # within +/-45 deg of the home pose, so the R0 reorientation
            # (rotvec relative to the home) stays small and avoids the +-180
            # deg rotvec singularity.
            Q0 = R.from_quat(obj_quat[0]).as_matrix()
            angles = [R.from_matrix(S @ Q0.T).magnitude() for S in S_targets]
            R_align = S_targets[int(np.argmin(angles))] @ Q0.T
            t_align = FC_CUBE_HOME_POS - R_align @ obj_pos[0]
            ep_frames[:, :3, 3] = (R_align @ ep_frames[:, :3, 3].T).T + t_align
            ep_frames[:, :3, :3] = np.array([R_align @ Rm for Rm in ep_frames[:, :3, :3]])
            ep_actions[:, :3] = (R_align @ ep_actions[:, :3].T).T
            ep_actions[:, 3:6] = (R_align @ ep_actions[:, 3:6].T).T

            frames_list.append(ep_frames)
            action_list.append(ep_actions)
            forces_list.append(ep_forces)
            lengths.append(len(eef_pos))

    frames = np.concatenate(frames_list, axis=0)
    actions = np.concatenate(action_list, axis=0)
    forces = np.concatenate(forces_list, axis=0)
    _, indices_end_points = _episode_indices(lengths)

    R0 = frames[0, :3, :3]

    X = np.concatenate(
        (forces, frames[:, :3, 3], rotvec_in_representation_base(frames[:, :3, :3], R0)),
        axis=-1,
    )
    Y = np.concatenate([vel_twists_in_representation_base(actions[:, :-1], R0), actions[:, -1:]], axis=-1)
    goal_poses = X[indices_end_points]

    return {
        "X": X,
        "Y": Y,
        "goal_poses": goal_poses,
        "R0": R0,
        "Y_scale": Y_scale,
        "Y_offset": Y_offset,
        "dataset_path": dataset_path,
    }


def load_object_centric_data(dataset_path: str) -> dict[str, Any]:
    """Load the object-centric demo data and build the (X, Y) feature matrices.

    Each demo is split into a grasp phase (``grasp_status == 0``) and a lift
    phase (``grasp_status != 0``), producing one skill each. The grasp skill is
    expressed in the symmetry-canonicalized object-centric frame; the lift
    skill in the world frame.

    Args:
        dataset_path: Path to the robomimic HDF5 dataset
            (e.g. ``examples/robomim/datasets/lift_ph_v15.hdf5``).

    Returns:
        Dictionary with:
        - ``Xs``: list of 2 arrays, ``[grasp (n_g, 7), lift (n_l, 7)]``.
        - ``Ys``: list of 2 arrays, ``[grasp (n_g, 7), lift (n_l, 7)]``.
        - ``goal_poses``: list of 2 arrays (last X row of each skill).
        - ``R0s``: list of 2 home rotations ``[grasp, lift]``.
        - ``Y_scale`` / ``Y_offset``: controller output half-range / midpoint.
        - ``dataset_path``: the input path (for environment creation).

    Raises:
        FileNotFoundError: If the dataset is missing or was not converted
            (see :func:`check_robomim_dataset`).
    """
    check_robomim_dataset(dataset_path)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    task_name = env_meta["env_name"]
    output_min, output_max = controller_output_range(env_meta)
    Y_scale = (output_max - output_min) / 2.0
    Y_offset = (output_max + output_min) / 2.0

    skill_lists: dict[str, list[np.ndarray]] = {
        "grasp_frames": [],
        "lift_frames": [],
        "grasp_obj_frames": [],
        "lift_obj_frames": [],
        "grasp_actions": [],
        "lift_actions": [],
    }
    grasp_lengths: list[int] = []
    lift_lengths: list[int] = []

    with h5py.File(dataset_path, "r") as f:
        group = f["data"] if "data" in f else f

        for ep_key, ep_group, obs_group in _iter_episodes(group):
            eef_pos, eef_quat = _extract_eef(obs_group)
            if eef_pos is None or eef_quat is None:
                print(f"Warning: Could not find EEF position/quat in {ep_key}")
                continue

            obj_pos, obj_quat = _extract_object(obs_group, task_name)
            if obj_pos is None or obj_quat is None:
                print(f"Warning: Could not find object data in {ep_key}")
                continue

            ep_actions = _load_actions(ep_group, Y_scale, Y_offset)
            if ep_actions is None:
                print(f"Warning: Could not find actions in {ep_key}")
                continue

            if "grasp_status" not in obs_group:
                print(f"Warning: Could not find grasp_status in {ep_key}")
                continue
            ep_grasp_status = obs_group["grasp_status"][:]

            ep_frames = _pose_to_frames(eef_pos, eef_quat)
            ep_obj_frames = _pose_to_frames(obj_pos, obj_quat)

            mask = ep_grasp_status == 0

            skill_lists["grasp_frames"].append(ep_frames[mask])
            skill_lists["lift_frames"].append(ep_frames[~mask])
            skill_lists["grasp_obj_frames"].append(ep_obj_frames[mask])
            skill_lists["lift_obj_frames"].append(ep_obj_frames[~mask])
            skill_lists["grasp_actions"].append(ep_actions[mask])
            skill_lists["lift_actions"].append(ep_actions[~mask])
            grasp_lengths.append(len(ep_frames[mask]))
            lift_lengths.append(len(ep_frames[~mask]))

    frames = [np.concatenate(skill_lists["grasp_frames"]), np.concatenate(skill_lists["lift_frames"])]
    obj_frames = [np.concatenate(skill_lists["grasp_obj_frames"]), np.concatenate(skill_lists["lift_obj_frames"])]
    actions = [np.concatenate(skill_lists["grasp_actions"]), np.concatenate(skill_lists["lift_actions"])]

    _, grasp_end_points = _episode_indices(grasp_lengths)
    _, lift_end_points = _episode_indices(lift_lengths)

    S_targets = symmetry_targets()

    # Object-centric transform for the grasp skill
    p_rep, R_rep, vel_rep = to_object_centric_frame(frames[0], obj_frames[0], actions[0], S_targets)

    # EEF home orientation per skill (first frame)
    R0s = [R_rep[0], frames[1][0, :3, :3]]

    Xs = [
        np.concatenate((np.zeros((len(frames[0]), 1)), p_rep, rotvec_in_representation_base(R_rep, R0s[0])), axis=-1),
        np.concatenate(
            (np.ones((len(frames[1]), 1)), frames[1][:, :3, 3], rotvec_in_representation_base(frames[1][:, :3, :3], R0s[1])),
            axis=-1,
        ),
    ]
    Ys = [
        np.concatenate([vel_twists_in_representation_base(vel_rep, R0s[0]), actions[0][:, -1:]], axis=-1),
        np.concatenate([vel_twists_in_representation_base(actions[1][:, :-1], R0s[1]), actions[1][:, -1:]], axis=-1),
    ]
    goal_poses = [Xs[0][grasp_end_points], Xs[1][lift_end_points]]

    return {
        "Xs": Xs,
        "Ys": Ys,
        "goal_poses": goal_poses,
        "R0s": R0s,
        "Y_scale": Y_scale,
        "Y_offset": Y_offset,
        "dataset_path": dataset_path,
    }
