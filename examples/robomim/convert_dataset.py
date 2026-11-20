#!/usr/bin/env python3
"""Convert the robomimic v1.5 Lift (PH) low-dim dataset into the CAPOF dataset.

The CAPOF robomimic use-cases consume an augmented version of the official
robomimic *Lift* (PH model) v1.5 low-dim demonstration dataset:

``datasets/lift_ph_v15.hdf5`` — a byte copy of the original ``low_dim_v15.hdf5``
plus two extra observation keys computed by replaying each demo in the
robosuite simulation:

- ``obs/normal_forces`` of shape (T, 2): the summed normal contact forces
  between the gripper finger pads and the cube (used by the force-conditioned
  use-case).
- ``obs/grasp_status`` of shape (T,): the per-frame grasp success flag (used
  by the object-centric use-case).

All other data (states, actions, rewards, ...) is untouched. Existing output
files are overwritten.

The original dataset is part of the robomimic dataset repository on
HuggingFace (v1.5, lift, PH model):

    https://huggingface.co/datasets/robomimic/robomimic_datasets/tree/main/v1.5/lift/ph

Requirements: a robomimic/robosuite/mujoco installation (see
``requirements.txt``) and a working MuJoCo offscreen context
(``MUJOCO_GL=egl`` on headless machines).

Usage (from the repository root):
    python examples/robomim/convert_dataset.py path/to/low_dim_v15.hdf5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import h5py
import mujoco
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

# Output location (where the entry scripts look for the dataset by default)
_PACKAGE_DIR = Path(__file__).resolve().parent
DATASET = _PACKAGE_DIR / "datasets" / "lift_ph_v15.hdf5"

# Acceptance criterion for the demo replay: both fingers must report at least
# this normal force (in N) on the final frame
FORCE_THRESHOLD = 10.0

# Maximum number of environment (re)creation attempts per demo
MAX_ATTEMPTS = 100


def get_gripper_cube_normal_forces(env: Any) -> dict[str, float]:
    """Sum the normal contact forces between the gripper finger pads and the cube.

    Scans all current MuJoCo contacts and accumulates the absolute x-force
    component of every contact between the cube and one of the two finger
    pad geoms.

    Args:
        env: robomimic environment wrapping the robosuite simulation.

    Returns:
        Dictionary with the summed force per finger: ``{"finger1": F1, "finger2": F2}``.
    """
    sim = env.env.sim
    model = sim.model._model
    data = sim.data._data

    finger_names = {
        "finger1": "gripper0_right_finger1_pad_collision",
        "finger2": "gripper0_right_finger2_pad_collision",
    }

    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_g0")
    finger_ids = {finger: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for finger, name in finger_names.items()}

    forces = {
        "finger1": 0.0,
        "finger2": 0.0,
    }

    for i in range(data.ncon):
        contact = data.contact[i]

        g1 = contact.geom1
        g2 = contact.geom2

        if g1 == cube_id:
            finger_geom = g2
        elif g2 == cube_id:
            finger_geom = g1
        else:
            continue

        for finger, geom_id in finger_ids.items():
            if finger_geom == geom_id:
                wrench = np.zeros(6)

                mujoco.mj_contactForce(model, data, i, wrench)

                forces[finger] += abs(wrench[0])
                break

    return forces


def _replay_episode(env_meta: dict[str, Any], states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replay one demo in the robosuite simulation and compute the extra observations.

    Both ``obs/normal_forces`` and ``obs/grasp_status`` are derived from the
    same replay: the grasp flag is read from the contacts established by
    ``reset_to`` (which forwards the simulation after setting the state), and
    the normal forces from the contacts after the explicit ``sim.forward()``.

    A fresh environment is created for each attempt, seeded with the attempt
    index so that a given attempt is reproducible. The replay is accepted once
    both fingers report at least :data:`FORCE_THRESHOLD` N of normal force on
    the final frame.

    Args:
        env_meta: robomimic environment metadata loaded from the dataset.
        states: (T, 32) per-frame simulation states of the demo.

    Returns:
        Tuple of:
        - normal_forces: (T, 2) float32 summed normal forces per finger.
        - grasp_status: (T,) uint8 per-frame grasp success flag.

    Raises:
        RuntimeError: If the acceptance criterion is not met within
            :data:`MAX_ATTEMPTS` attempts.
    """
    T = len(states)
    attempt = 0

    while True:
        np.random.seed(attempt)

        env = EnvUtils.create_env_from_metadata(
            env_meta,
            render=False,
            render_offscreen=True,
        )

        normal_forces = np.zeros((T, 2), dtype=np.float32)
        grasp_status = np.zeros(T, dtype=np.uint8)

        # Replay the demo frame by frame
        env.reset_to({"states": states[0]})

        for t in range(T):
            env.reset_to({"states": states[t]})

            is_grasped = env.env._check_grasp(
                gripper=env.env.robots[0].gripper["right"],
                object_geoms=env.env.cube.contact_geoms,
            )
            grasp_status[t] = int(is_grasped)

            env.env.sim.forward()
            forces = get_gripper_cube_normal_forces(env)

            normal_forces[t] = [
                forces["finger1"],
                forces["finger2"],
            ]

        print(f"  final forces: {normal_forces[-1]} (attempt {attempt})")

        # Acceptance criterion: BOTH forces must be >= FORCE_THRESHOLD
        if normal_forces[-1, 0] >= FORCE_THRESHOLD and normal_forces[-1, 1] >= FORCE_THRESHOLD:
            print("  Accepted!")
            return normal_forces, grasp_status

        if attempt + 1 >= MAX_ATTEMPTS:
            raise RuntimeError(
                f"Demo not accepted after {MAX_ATTEMPTS} attempts "
                f"(final forces {normal_forces[-1]}, threshold {FORCE_THRESHOLD} N per finger)"
            )

        attempt += 1


def _convert(input_path: Path, output_path: Path) -> None:
    """Convert the input dataset (byte copy of the input plus extra obs keys).

    Args:
        input_path: Original robomimic ``low_dim_v15.hdf5`` file.
        output_path: Output dataset path (overwritten if it exists).
    """
    print(f"Converting {input_path.name} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy the original file first; the extra observations are appended in place
    shutil.copy(input_path, output_path)

    # Init obs utils (required by robomimic)
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=["agentview_image"],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    # Load env meta (now safe because we copied the file)
    env_meta = FileUtils.get_env_metadata_from_dataset(str(output_path))

    with h5py.File(output_path, "a") as f:
        data = f["data"]

        ep_keys = sorted((k for k in data.keys() if k.startswith("demo_")), key=lambda x: int(x.split("_")[-1]))

        for ep_key in ep_keys:
            print(f"Processing: {ep_key}")

            grp = data[ep_key]

            normal_forces, grasp_status = _replay_episode(env_meta, grp["states"][:])

            obs_out = grp["obs"] if "obs" in grp else grp.create_group("obs")

            # Save the extra observations
            obs_out.create_dataset("normal_forces", data=normal_forces)
            obs_out.create_dataset("grasp_status", data=grasp_status)

    print(f"Done: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert the robomimic v1.5 Lift (PH) low-dim dataset into the CAPOF dataset")
    parser.add_argument("input", type=str, help="Path to the original low_dim_v15.hdf5 file")

    return parser.parse_args()


def main() -> None:
    """Run the dataset conversion."""
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Error: input dataset not found: {input_path}")

    _convert(input_path, DATASET)


if __name__ == "__main__":
    main()
