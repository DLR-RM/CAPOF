"""Small shared helpers for the robomim use-cases."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def repo_root() -> Path:
    """Return the repository root (two levels above this package)."""
    return Path(__file__).resolve().parent.parent.parent


def package_dir() -> Path:
    """Return the directory of this package (``examples/robomim``)."""
    return Path(__file__).resolve().parent


def controller_output_range(env_meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract the controller output bounds (min, max) from robomimic env metadata.

    The bounds define the action range the controller executes per step, and
    are used to (de)normalize the 6D velocity commands between the CAPOF
    representation and the environment's ``[-1, 1]`` action space.

    Args:
        env_meta: Environment metadata dictionary from
            ``robomimic.utils.file_utils.get_env_metadata_from_dataset``.

    Returns:
        Tuple of (output_min, output_max), each a float32 array of shape (6,).
    """
    ctrl_cfg = env_meta["env_kwargs"]["controller_configs"]["body_parts"]["right"]
    output_min = np.array(ctrl_cfg["output_min"], dtype=np.float32)
    output_max = np.array(ctrl_cfg["output_max"], dtype=np.float32)
    return output_min, output_max


def save_video(images: Sequence[np.ndarray], path: str | Path, fps: float = 20.0) -> None:
    """Save a sequence of RGB images as an MP4 video.

    The parent directory of ``path`` is created if it does not exist.

    Args:
        images: Sequence of (H, W, 3) uint8 RGB images.
        path: Output video path (``.mp4``).
        fps: Frames per second of the output video.
    """
    images = list(images)
    if not images:
        return

    height, width = images[0].shape[:2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for img in images:
        writer.write(img[..., ::-1])  # RGB -> BGR
    writer.release()
    print(f"Video saved to {path}")
