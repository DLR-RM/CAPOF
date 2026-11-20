#!/usr/bin/env python3
"""Demonstration of CAPOF (Context-Adaptive Policy Framework) with 6D motion data.

This example generates a helical motion trajectory, converts it to 6D twist
representations, trains a Gaussian Process (GP) on the demonstrated motion,
and uses CAPOF to reproduce the learned motion through iterative integration
of velocity twists.

The workflow generates ground-truth frames and twists, trains a GP to learn
the velocity twist as a function of pose, and then demonstrates CAPOF's
ability to reproduce the motion by integrating predicted velocities back into
trajectories.

Usage (from the repository root):
    python examples/6d_example/six_dimensional_example.py            # with final plot
    python examples/6d_example/six_dimensional_example.py --no-plot  # headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm
from scipy.spatial.transform import Rotation as R

# Make the repository root importable regardless of the working directory
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from capof.training import train_capof  # noqa: E402
from six_dimensional_interface import SixDimensionalOuterControlLoop  # noqa: E402


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

    Args:
        v: Vector to normalize.

    Returns:
        Normalized vector with unit length.
    """
    return v / np.linalg.norm(v)


def make_frame(axis: np.ndarray) -> np.ndarray:
    """Create an orthonormal frame with its x-axis aligned to ``axis``.

    Args:
        axis: Direction vector for the frame x-axis (normalized internally).

    Returns:
        (3, 3) rotation matrix whose columns are the frame axes.
    """
    x = normalize(axis)

    # Pick an arbitrary vector not parallel to x
    tmp = np.array([0, 0, 1])
    if np.allclose(x, tmp):
        tmp = np.array([0, 1, 0])

    y = normalize(np.cross(tmp, x))
    z = np.cross(x, y)

    return np.column_stack((x, y, z))


def vee_se3(X: np.ndarray) -> np.ndarray:
    """Map an se(3) matrix to a 6D twist vector ``[v, omega]``.

    Args:
        X: (4, 4) homogeneous matrix whose top-left block is the skew-symmetric
            rotation part and whose top-right block is the translation.

    Returns:
        (6,) twist vector.
    """
    omega_hat = X[:3, :3]
    v = X[:3, 3]

    omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])

    return np.hstack((v, omega))


def generate_motion(axis: np.ndarray, n_steps: int = 50, length: float = 1.0, total_rotation: float = 2 * np.pi) -> np.ndarray:
    """Generate a helical motion trajectory as homogeneous transformation matrices.

    Creates a trajectory that translates along and rotates around a given
    axis, producing a helical (screw) motion path.

    Args:
        axis: Direction vector for the motion axis (normalized internally).
        n_steps: Number of trajectory steps.
        length: Total translation distance along the axis.
        total_rotation: Total rotation angle around the axis in radians.

    Returns:
        (n_steps, 4, 4) sequence of homogeneous transformation matrices.
    """
    axis = normalize(axis)
    base_R = make_frame(axis)

    frames = []

    for i in range(n_steps):
        t = i / (n_steps - 1)

        # Translation along axis
        position = t * length * axis

        # Rotation around axis
        angle = t * total_rotation
        R_rot = R.from_rotvec(axis * angle).as_matrix()

        # Rotate base frame around axis
        R_ = R_rot @ base_R

        # Build homogeneous transform
        T = np.eye(4)
        T[:3, :3] = R_
        T[:3, 3] = position

        frames.append(T)

    return np.array(frames)


def compute_twist_from_frames(frames: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Compute 6D velocity twists from a sequence of homogeneous matrices.

    Uses the matrix logarithm to convert relative motions between consecutive
    frames into se(3) twists, then scales by dt to obtain velocities.

    Args:
        frames: (n_frames, 4, 4) sequence of homogeneous transformation matrices.
        dt: Time step between frames for velocity scaling.

    Returns:
        (n_frames - 1, 6) velocity twists
        ``[v_x, v_y, v_z, omega_x, omega_y, omega_z]`` for each transition.
    """
    twists = []

    for i in range(len(frames) - 1):
        T1 = frames[i]
        T2 = frames[i + 1]

        # Relative motion
        delta_T = np.linalg.inv(T1) @ T2

        # Matrix logarithm
        xi_hat = logm(delta_T)

        # Convert to 6D twist
        xi = vee_se3(xi_hat)

        twist = xi / dt
        twists.append(twist)

    return np.array(twists)


def transform_twist(transformation_matrix: np.ndarray, vel_twist: np.ndarray, only_rotation: bool = False) -> np.ndarray:
    """Transform a 6D velocity twist between coordinate frames (adjoint map).

    Converts a velocity twist expressed in one frame to the equivalent twist
    in another frame, accounting for both rotational and translational velocity
    components. The assumed twist order is ``[v_x, v_y, v_z, omega_x, omega_y, omega_z]``.

    Args:
        transformation_matrix: (4, 4) homogeneous transformation from source to
            target frame.
        vel_twist: (6,) velocity twist in the source frame ``[v; omega]``.
        only_rotation: If True, only apply the rotational transformation
            (translation set to zero).

    Returns:
        (6,) transformed velocity twist in the target frame.
    """
    rot_mat = transformation_matrix[:3, :3]

    if only_rotation:
        translation_vec = np.zeros(3)
    else:
        translation_vec = transformation_matrix[:3, 3]

    # Convert translation vector to bracket notation for matrix multiplication
    translation_vec_mat = np.array(
        [
            [0, -translation_vec[2], translation_vec[1]],
            [translation_vec[2], 0, -translation_vec[0]],
            [-translation_vec[1], translation_vec[0], 0],
        ]
    )

    # Adjoint representation of the transformation matrix
    adjoint_transformation = np.block(
        [
            [rot_mat, translation_vec_mat @ rot_mat],
            [np.zeros((3, 3)), rot_mat],
        ]
    )

    return adjoint_transformation @ vel_twist


def plot_comparison(frames: np.ndarray, poses6d: np.ndarray, learned_frames: list[np.ndarray]) -> None:
    """Plot three views of a trajectory: from matrices, from 6D, and learned.

    Creates a 3-subplot figure showing the same trajectory represented in three
    different ways: (1) positions and orientations extracted directly from 4x4
    matrices, (2) poses represented as 6D axis-angle vectors, and (3)
    trajectories learned through CAPOF integration.

    Args:
        frames: (n, 4, 4) ground-truth homogeneous transformation matrices.
        poses6d: (n, 6) ground-truth poses as ``[position, rotvec]``.
        learned_frames: Learned homogeneous transformation matrices from CAPOF
            inference.
    """
    fig = plt.figure(figsize=(16, 6))

    # 3D projection axes (typed as Any: matplotlib stubs only cover 2D Axes)
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    axes: list[Any] = [ax1, ax2, ax3]
    titles = ["From 4x4 matrices", "From 6D (axis-angle)", "Learned integrated frames"]

    scale = 0.1
    colors = ["r", "g", "b"]

    for ax, title in zip(axes, titles):
        for i in range(len(learned_frames)):
            # --- Select data source ---
            if title == "From 6D (axis-angle)":
                p = poses6d[i][:3]
                w = poses6d[i][3:]
                R_ = R.from_rotvec(w).as_matrix()

            elif title == "Learned integrated frames":
                p = learned_frames[i][:3, 3]
                R_ = learned_frames[i][:3, :3]

            else:
                p = frames[i][:3, 3]
                R_ = frames[i][:3, :3]

            # --- Draw frame axes ---
            for j in range(3):
                ax.quiver(p[0], p[1], p[2], R_[0, j] * scale, R_[1, j] * scale, R_[2, j] * scale, color=colors[j])

        # --- Trajectory ---
        if title == "From 6D (axis-angle)":
            traj = poses6d[:, :3]
        elif title == "Learned integrated frames":
            traj = np.array([T[:3, 3] for T in learned_frames])
        else:
            traj = np.array([T[:3, 3] for T in frames])

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "k--")

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_ylim3d([-0.4, 0.4])
        ax.set_zlim3d([-0.4, 0.4])

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CAPOF 6D motion reproduction example")
    parser.add_argument("--no-plot", action="store_true", help="Skip the final matplotlib figure (headless)")
    parser.add_argument("--length", type=float, default=1.0, help="Total translation distance along the axis")
    parser.add_argument("--rotation", type=float, default=2 * np.pi, help="Total rotation around the axis (rad)")
    return parser.parse_args()


def main() -> None:
    """Generate a helical motion, train a GP on it, and reproduce it with CAPOF."""
    args = parse_args()
    dt = 1.0

    # Generate motion
    frames = generate_motion(axis=np.array([1, 0, 0]), n_steps=50, length=args.length, total_rotation=args.rotation)
    vel_twist = compute_twist_from_frames(frames.copy(), dt=dt)

    # Append zero twist (the last frame has no outgoing velocity)
    zero_twist = np.zeros((1, 6))
    vel_twist = np.vstack((vel_twist, zero_twist))

    # Calculate TCP velocity twist in space coordinate frame
    vel_twist = np.array(
        [transform_twist(frames.copy()[k], vel_twist.copy()[k], only_rotation=True) for k in range(vel_twist.shape[0])]
    )

    # Pre-process input data: poses as [position, rotvec]
    poses6d = np.concatenate((frames[:, :3, 3], R.from_matrix(frames[:, :3, :3]).as_rotvec()), axis=-1)

    # Hyperparameters
    length_scale = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    alpha = 0.01  # Noise variance
    K_sp = np.array([0.1, 0.1])  # Stabilizing policy gains
    K_gap = np.array([0.1, 0.1])  # Goal-attractor gains

    # Fit the GP and build the CAPOF policy (n_regression_points=len keeps all demos)
    capof = train_capof(
        poses6d,
        vel_twist,
        goal_poses=poses6d[-1:].copy(),
        K_sp=K_sp,
        K_gap=K_gap,
        alpha=alpha,
        kernel_length_vec=length_scale,
        c_dim=0,
        x_dim=3,
        phi_dim=3,
        x_dot_dim=3,
        phi_dot_dim=3,
        mu_add_dim=0,
        n_regression_points=len(poses6d),
    )

    # Create outer control loop
    sim = SixDimensionalOuterControlLoop(poses6d, capof, np.zeros(6), control_rate=1 / dt, loop_limit=50)

    # Set start pose (offset from the demonstrated start position)
    pose = frames.copy()[0]
    pose[:3, 3] = np.array([0.0, 0.3, 0.3])

    # Inference
    print("Start loop")
    sim.run_loop(pose.copy())

    # Plot results
    if not args.no_plot:
        plot_comparison(frames, poses6d, sim.visited_states)


if __name__ == "__main__":
    main()
