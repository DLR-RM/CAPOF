#!/usr/bin/env python3
"""LASA handwriting demonstration using CAPOF.

This example runs the LASA handwriting dataset through the CAPOF
(Context-Adaptive Policy Framework) system.

The script iterates over the demonstration letters, trains the selected policy,
creates a CAPOF instance combining the LfD policy with stabilizing and goal
attractor policies, and evaluates performance via the outer control loop from
random start points. Results include success rate, iteration count, and the
mean distance to the demonstrated trajectories (MMD).

The dataset is not part of the repository (it is free for non-commercial
academic use only). Clone it once before running this example:

    git clone https://github.com/epfl-lasa/LASAHandwritingDataset.git examples/lasa/dataset

Usage (from the repository root):
    python examples/lasa/lasa.py                          # animated visualization
    python examples/lasa/lasa.py --mode headless          # no plots (fast)
    python examples/lasa/lasa.py --letters Angle Line     # run a subset of letters
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make the repository root importable regardless of the working directory
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from capof.training import train_capof  # noqa: E402
from lasa_data import DEFAULT_LASA_REPO, check_lasa_repo, load_lasa_letter  # noqa: E402
from lasa_interface import LasaOuterControlLoop  # noqa: E402


def scale_relative_hyperparameters(
    X: np.ndarray, Y: np.ndarray, alpha: float, kernel_length_vec: np.ndarray, K_sp: float, K_gap: float
) -> tuple[float, np.ndarray, float, float]:
    """Scale hyperparameters relative to the data ranges.

    Scales alpha, kernel lengths, and policy gains by the data statistics,
    making hyperparameter choices more robust across different datasets and
    scales.

    Args:
        X: (n_samples, n_features) input demonstration data (positions).
        Y: (n_samples, n_targets) output demonstration data (velocities).
        alpha: Relative noise variance parameter.
        kernel_length_vec: Relative GP kernel length scale vector.
        K_sp: Relative stabilizing policy gain.
        K_gap: Relative goal attractor policy gain.

    Returns:
        Scaled (alpha, kernel_length_vec, K_sp, K_gap).
    """
    alpha *= np.max(np.var(Y, axis=0))

    max_abs_velocity = np.max(np.abs(Y))
    K_sp *= max_abs_velocity
    K_gap *= max_abs_velocity

    kernel_length_vec *= np.max(X, axis=0) - np.min(X, axis=0)
    return alpha, kernel_length_vec, K_sp, K_gap


def create_start_points(X: np.ndarray, num: int = 10) -> np.ndarray:
    """Create random starting points within the bounds of the demonstration data.

    Args:
        X: (n_samples, 2) demonstration input data (2D positions).
        num: Number of random starting points.

    Returns:
        (num, 2) randomly sampled start points within the data bounds.
    """
    start_points = np.random.rand(num, 2)
    start_points[:, 0] = start_points[:, 0] * (np.max(X, axis=0) - np.min(X, axis=0))[0] + np.min(X, axis=0)[0]
    start_points[:, 1] = start_points[:, 1] * (np.max(X, axis=0) - np.min(X, axis=0))[1] + np.min(X, axis=0)[1]
    return start_points


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CAPOF LASA handwriting example")

    # Data
    parser.add_argument(
        "--lasa-repo", type=str, default=str(DEFAULT_LASA_REPO), help="Path to the folder with the LASA .mat files"
    )
    parser.add_argument(
        "--letters", type=str, nargs="*", default=None, help="Run only these letters (folder names without .mat; default: all)"
    )

    # Execution mode
    parser.add_argument(
        "--mode",
        type=str,
        default="animate",
        choices=["animate", "headless"],
        help="animate: show a matplotlib animation per start point; headless: run_loop only (no plots)",
    )
    parser.add_argument("--n-starts", type=int, default=10, help="Number of random start points per letter")

    # Control loop
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random in [0, 100))")
    parser.add_argument("--loop-limit", type=int, default=500, help="Maximum number of control iterations")
    parser.add_argument("--threshold", type=float, default=0.01, help="Relative goal-reach threshold")
    parser.add_argument("--control-rate", type=float, default=20, help="Control loop frequency in Hz")
    parser.add_argument(
        "--policy-combination",
        type=str,
        default="MoE",
        choices=["LfD", "Stabilizing", "Goal", "LfD + Stabilizing", "LfD + Goal", "MoE"],
        help="Policy combination method",
    )

    # GP / CAPOF hyperparameters (relative values; scaled by the data ranges)
    parser.add_argument("--n-regression", type=int, default=500, help="Number of regression points sampled from the demos")
    parser.add_argument("--alpha", type=float, default=0.0016, help="Relative GP noise variance")
    parser.add_argument("--l-x", type=float, default=0.0763, help="Relative kernel length scale for position")
    parser.add_argument("--k-sp", type=float, default=0.7889, help="Relative stabilizing gain")
    parser.add_argument("--k-gap", type=float, default=0.5826, help="Relative goal-attractor gain")

    return parser.parse_args()


def main() -> None:
    """Run the LASA handwriting experiments with CAPOF."""
    args = parse_args()

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 100))
    np.random.seed(seed)
    print(f"LASA | seed={seed} | mode={args.mode} | letters={args.letters or 'all'}")

    try:
        lasa_repo = check_lasa_repo(args.lasa_repo)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    folders = sorted(f for f in os.listdir(lasa_repo) if f.endswith(".mat"))
    if args.letters:
        wanted = set(args.letters)
        folders = [f for f in folders if Path(f).stem in wanted]
        missing = wanted - {Path(f).stem for f in folders}
        if missing:
            print(f"Warning: letters not found in {lasa_repo}: {sorted(missing)}")

    # Per-letter evaluation results
    overall_success: dict[str, list[bool]] = {f: [] for f in folders}
    overall_iterations: dict[str, list[int]] = {f: [] for f in folders}
    overall_MMD: dict[str, list[float]] = {f: [] for f in folders}

    # Iterate over letters
    for folder_name in folders:
        print(folder_name)

        # Load the letter's demonstration data
        letter = load_lasa_letter(lasa_repo / folder_name)
        X = letter["X"]
        Y = letter["Y"]
        demo_start_poses = letter["demo_start_poses"]
        goal_poses = letter["goal_poses"]

        # Scale relative parameters and train the CAPOF policy
        alpha_, kernel_length_vec_, K_sp_, K_gap_ = scale_relative_hyperparameters(
            X, Y, args.alpha, np.array([args.l_x, args.l_x]), args.k_sp, args.k_gap
        )
        policy = train_capof(
            X,
            Y,
            goal_poses,
            [K_sp_],
            [K_gap_],
            alpha_,
            kernel_length_vec_,
            c_dim=0,
            x_dim=2,
            phi_dim=0,
            x_dot_dim=2,
            phi_dot_dim=0,
            mu_add_dim=0,
            n_regression_points=args.n_regression,
        )

        sim = LasaOuterControlLoop(
            X,
            policy,
            np.zeros(2),
            demo_start_poses,
            control_rate=args.control_rate,
            loop_limit=args.loop_limit,
            threshold=args.threshold,
            policy_combination=args.policy_combination,
            animate=(args.mode == "animate"),
        )

        # Random start points within the data bounds
        start_poses = create_start_points(X, num=args.n_starts)

        for index in range(len(start_poses)):
            # Set start position
            x = start_poses[index].copy()

            # Run the control loop (with or without animation)
            if args.mode == "animate":
                sim.animate(x)
                plt.show()
                plt.close(sim.fig)
            else:
                sim.run_loop(x)

            # Mean of the minimum distance from each visited state to the demonstrations
            min_distances = np.mean(np.min(np.linalg.norm(np.array(sim.visited_states)[:, np.newaxis] - X, axis=-1), axis=1))

            # Collect success rate, iterations and distance to data
            overall_success[folder_name].append(sim.success)
            overall_iterations[folder_name].append(len(sim.visited_states))
            overall_MMD[folder_name].append(min_distances)

        print(f"Letter success:    {np.mean(overall_success[folder_name]):.3f}")
        print(f"Letter iterations: {np.mean(overall_iterations[folder_name]):.1f}")
        print(f"Letter MMD:        {np.mean(overall_MMD[folder_name]):.3f}")

    all_success = [v for vals in overall_success.values() for v in vals]
    all_iterations = [v for vals in overall_iterations.values() for v in vals]
    all_MMD = [v for vals in overall_MMD.values() for v in vals]
    print("Overall success: ", np.mean(all_success))
    print("Overall iterations: ", np.mean(all_iterations))
    print("Overall MMD: ", np.mean(all_MMD))


if __name__ == "__main__":
    main()
