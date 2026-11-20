#!/usr/bin/env python3
"""Force-conditioned robomimic Lift use-case.

Trains a CAPOF policy whose input is conditioned on the two gripper-cube
contact forces, then executes it in the robomimic *Lift* environment (the
cube starts at a fixed pose).

The demonstration dataset is not part of the repository. Generate it once from
the official robomimic v1.5 *Lift* (PH model) low-dim dataset with
``examples/robomim/convert_dataset.py``; by default it is then expected at
``examples/robomim/datasets/lift_ph_v15.hdf5`` (pass ``--dataset`` to use a
different location). See the README for the full instructions.

Usage (from the repository root):
    python examples/robomim/force_conditioned.py                     # headless, video saved
    python examples/robomim/force_conditioned.py --live-render       # interactive viewer
    python examples/robomim/force_conditioned.py --seed 7 --loop-limit 250

In live mode, press ``space`` to re-place the cube and ``r`` to restart the
episode with a fresh seed; close the viewer window to stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the repository root importable regardless of the working directory
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from capof.training import train_capof  # noqa: E402
from examples.robomim.data_processing import load_force_conditioned_data  # noqa: E402
from examples.robomim.robomim_interface import ForceConditionedLoop  # noqa: E402
from examples.robomim.utils import package_dir, save_video  # noqa: E402

DEFAULT_DATASET = package_dir() / "datasets" / "lift_ph_v15.hdf5"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CAPOF force-conditioned robomimic Lift use-case")

    # Data / environment
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET), help="Path to the robomimic HDF5 dataset")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed (default: random in [0, 100))")
    parser.add_argument("--live-render", action="store_true", help="Open a live MuJoCo viewer window")
    parser.add_argument(
        "--loop-limit",
        type=int,
        default=250,
        help="Maximum number of control steps per episode (ignored in --live-render mode)",
    )
    parser.add_argument("--control-rate", type=float, default=20, help="Control loop frequency in Hz")
    parser.add_argument("--threshold", type=float, default=0.01, help="Relative goal-reach threshold")
    parser.add_argument(
        "--policy-combination",
        type=str,
        default="MoE",
        choices=["LfD", "Stabilizing", "Goal", "LfD + Stabilizing", "LfD + Goal", "MoE"],
        help="Policy combination method",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(package_dir() / "results" / "force_conditioned"),
        help="Directory for saved videos",
    )

    # GP / CAPOF hyperparameters (relative values; scaled by the data ranges)
    parser.add_argument("--n-regression", type=int, default=500, help="Number of regression points sampled from the demos")
    parser.add_argument("--alpha", type=float, default=0.0917, help="Relative GP noise variance")
    parser.add_argument("--l-c", type=float, default=0.4575, help="Relative kernel length scale for context (forces)")
    parser.add_argument("--l-x", type=float, default=0.1545, help="Relative kernel length scale for position")
    parser.add_argument("--l-phi", type=float, default=0.4488, help="Relative kernel length scale for orientation")
    parser.add_argument("--k-sp-x", type=float, default=0.5454, help="Relative stabilizing gain for position")
    parser.add_argument("--k-sp-phi", type=float, default=0.5961, help="Relative stabilizing gain for orientation")
    parser.add_argument("--k-gap-x", type=float, default=0.3291, help="Relative goal-attractor gain for position")
    parser.add_argument("--k-gap-phi", type=float, default=0.8791, help="Relative goal-attractor gain for orientation")

    return parser.parse_args()


def main() -> None:
    """Train the force-conditioned policy and run one episode."""
    args = parse_args()

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 100))
    if args.live_render:
        # loop_limit is ignored in live-render mode (the loop runs until the viewer is closed)
        print(f"Force-conditioned Lift | seed={seed} | live_render=True")
    else:
        print(f"Force-conditioned Lift | seed={seed} | live_render=False | loop_limit={args.loop_limit}")

    # Load demo data and build the (X, Y) feature matrices
    data = load_force_conditioned_data(args.dataset)
    X, Y = data["X"], data["Y"]
    R0 = data["R0"]
    Y_scale = data["Y_scale"]

    # Scale the relative hyperparameters by the data ranges
    K_sp = np.array([args.k_sp_x, args.k_sp_phi]) * np.array([np.max(Y_scale[:3]), np.max(Y_scale[3:])])
    K_gap = np.array([args.k_gap_x, args.k_gap_phi]) * np.array([np.max(Y_scale[:3]), np.max(Y_scale[3:])])
    alpha = args.alpha * np.max(np.var(Y, axis=0))
    kernel_length_vec = np.array([args.l_c, args.l_c, args.l_x, args.l_x, args.l_x, args.l_phi, args.l_phi, args.l_phi])
    kernel_length_vec *= np.max(X, axis=0) - np.min(X, axis=0)

    # Fit the GP and build the CAPOF policy
    capof = train_capof(
        X,
        Y,
        data["goal_poses"],
        K_sp,
        K_gap,
        alpha,
        kernel_length_vec,
        c_dim=2,
        x_dim=3,
        phi_dim=3,
        x_dot_dim=3,
        phi_dot_dim=3,
        mu_add_dim=1,
        n_regression_points=args.n_regression,
    )

    # Run the control loop
    sim = ForceConditionedLoop(
        X,
        capof,
        np.zeros(6),
        seed,
        data["dataset_path"],
        R0,
        control_rate=args.control_rate,
        loop_limit=args.loop_limit,
        threshold=args.threshold,
        policy_combination=args.policy_combination,
        record_images=not args.live_render,
        live_render=args.live_render,
    )
    sim.run_loop()

    # Print the final reward (parsed by evaluate.sh)
    print("Score: ", sim.rewards[-1])

    # Save the episode video (headless mode only)
    if sim.imgs:
        video_path = Path(args.results_dir) / sim.env.task_name / f"{sim.env._seed}.mp4"
        save_video(sim.imgs, video_path, fps=20.0)

    if args.live_render:
        sim.env.close()


if __name__ == "__main__":
    main()
