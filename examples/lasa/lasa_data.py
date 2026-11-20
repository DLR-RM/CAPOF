"""Shared LASA dataset loading.

The LASA handwriting dataset consists of one ``.mat`` file per letter, each
containing a ``demos`` cell array with per-demo ``pos`` (pen positions) and
``vel`` (pen velocities) trajectories.

The dataset is not shipped with this repository (it is free for
non-commercial academic use only, see the dataset repository below).
Clone it once before using the LASA examples::

    git clone https://github.com/epfl-lasa/LASAHandwritingDataset.git examples/lasa/dataset

The ``.mat`` files then live in ``examples/lasa/dataset/DataSet``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

# The dataset must be cloned into the repository (see module docstring)
LASA_GIT_URL = "https://github.com/epfl-lasa/LASAHandwritingDataset"
LASA_CLONE_DIR = Path(__file__).resolve().parent / "dataset"
# The .mat files live in the DataSet/ subfolder of the cloned repository
DEFAULT_LASA_REPO = LASA_CLONE_DIR / "DataSet"


def check_lasa_repo(lasa_repo: str | Path) -> Path:
    """Validate the location of the LASA dataset.

    Args:
        lasa_repo: Path to the folder containing the LASA ``.mat`` files.

    Returns:
        The validated path as a :class:`~pathlib.Path`.

    Raises:
        FileNotFoundError: If the folder is missing or contains no ``.mat``
            files. The message explains how to obtain the dataset.
    """
    lasa_repo = Path(lasa_repo)
    clone_command = f"git clone {LASA_GIT_URL} {LASA_CLONE_DIR}"
    if not lasa_repo.is_dir():
        raise FileNotFoundError(
            f"LASA dataset not found at {lasa_repo}.\n"
            f"Obtain it by cloning the dataset repository:\n"
            f"    {clone_command}\n"
            f"The .mat files are then located in {DEFAULT_LASA_REPO} "
            f"(or pass --lasa-repo pointing at the DataSet folder of your clone)."
        )
    if not any(lasa_repo.glob("*.mat")):
        raise FileNotFoundError(
            f"No .mat files found in {lasa_repo}.\n"
            f"Obtain the dataset by cloning the dataset repository:\n"
            f"    {clone_command}"
        )
    return lasa_repo


def load_lasa_letter(mat_path: str | Path) -> dict[str, Any]:
    """Load one LASA letter ``.mat`` file.

    Args:
        mat_path: Path to the letter ``.mat`` file.

    Returns:
        Dictionary with:
        - ``X``: (n, 2) concatenated pen positions over all demos.
        - ``Y``: (n, 2) concatenated pen velocities over all demos.
        - ``demo_start_poses``: (n_demos, 2) first position of each demo.
        - ``goal_poses``: (n_demos, 2) last position of each demo.
    """
    raw_data = loadmat(mat_path)
    poses = [d["pos"][0][0].T for d in raw_data["demos"][0]]  # list of (n_i, 2) arrays
    velocities = [d["vel"][0][0].T for d in raw_data["demos"][0]]  # list of (n_i, 2) arrays

    return {
        "X": np.concatenate(poses, axis=0),
        "Y": np.concatenate(velocities, axis=0),
        "demo_start_poses": np.array([demo[0] for demo in poses]),
        "goal_poses": np.array([demo[-1] for demo in poses]),
    }
