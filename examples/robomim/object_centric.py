#!/usr/bin/env python3
"""Object-centric robomimic Lift use-case (two skills: grasp + lift).

Trains two CAPOF policies — a grasp skill in a symmetry-canonicalized
object-centric frame and a lift skill in the world frame — and executes them
in the robomimic *Lift* environment (the cube starts at a random pose). The
active skill is switched online by the environment's grasp status.

The demonstration dataset is not part of the repository. Generate it once from
the official robomimic v1.5 *Lift* (PH model) low-dim dataset with
``examples/robomim/convert_dataset.py``; by default it is then expected at
``examples/robomim/datasets/lift_ph_v15.hdf5`` (pass ``--dataset`` to
use a different location). See the README for the full instructions.

Usage (from the repository root):
    python examples/robomim/object_centric.py                 # headless, video saved
    python examples/robomim/object_centric.py --live-render   # interactive viewer
    python examples/robomim/object_centric.py --seed 7 --loop-limit 250

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
from examples.robomim.data_processing import load_object_centric_data  # noqa: E402
from examples.robomim.robomim_interface import ObjectCentricLoop  # noqa: E402
from examples.robomim.utils import package_dir, save_video  # noqa: E402

DEFAULT_DATASET = package_dir() / "datasets" / "lift_ph_v15.hdf5"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CAPOF object-centric robomimic Lift use-case (grasp + lift)")

    # Data / environment
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET), help="Path to the robomimic HDF5 dataset")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed (default: random in [0, 100))")
    parser.add_argument("--live-render", action="store_true", help="Open a live MuJoCo viewer window (interactive)")
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
        "--results-dir", type=str, default=str(package_dir() / "results" / "object_centric"), help="Directory for saved videos"
    )

    # GP / CAPOF hyperparameters (relative values; scaled by the data ranges)
    parser.add_argument("--n-regression", type=int, default=500, help="Number of regression points sampled per skill")
    parser.add_argument("--alpha", type=float, default=0.0981, help="Relative GP noise variance")
    parser.add_argument("--l-c", type=float, default=0.2514, help="Relative kernel length scale for context (grasp status)")
    parser.add_argument("--l-x", type=float, default=0.1163, help="Relative kernel length scale for position")
    parser.add_argument("--l-phi", type=float, default=0.3564, help="Relative kernel length scale for orientation")
    parser.add_argument("--k-sp-x", type=float, default=0.8087, help="Relative stabilizing gain for position")
    parser.add_argument("--k-sp-phi", type=float, default=0.3634, help="Relative stabilizing gain for orientation")
    parser.add_argument("--k-gap-x", type=float, default=0.8761, help="Relative goal-attractor gain for position")
    parser.add_argument("--k-gap-phi", type=float, default=0.4386, help="Relative goal-attractor gain for orientation")

    return parser.parse_args()


def main() -> None:
    """Train the grasp + lift policies and run one episode."""
    args = parse_args()

    seed = args.seed if args.seed is not None else int(np.random.randint(0, 100))
    if args.live_render:
        # loop_limit is ignored in live-render mode (the loop runs until the viewer is closed)
        print(f"Object-centric Lift | seed={seed} | live_render=True")
    else:
        print(f"Object-centric Lift | seed={seed} | live_render=False | loop_limit={args.loop_limit}")

    # Load demo data and build the per-skill (X, Y) feature matrices
    data = load_object_centric_data(args.dataset)
    Xs, Ys = data["Xs"], data["Ys"]
    R0s = data["R0s"]
    Y_scale = data["Y_scale"]

    # Scale the relative gains by the controller output ranges
    K_sp = np.array([args.k_sp_x, args.k_sp_phi]) * np.array([np.max(Y_scale[:3]), np.max(Y_scale[3:])])
    K_gap = np.array([args.k_gap_x, args.k_gap_phi]) * np.array([np.max(Y_scale[:3]), np.max(Y_scale[3:])])

    # Fit one GP per skill (grasp, lift) with per-skill relative scaling
    capofs = []
    for i in range(2):
        alpha_ = args.alpha * np.max(np.var(Ys[i], axis=0))
        scale = np.max(Xs[i], axis=0) - np.min(Xs[i], axis=0)
        scale = np.where(scale == 0, 1.0, scale)  # prevent zero length scales
        kernel_length_vec_ = np.array([args.l_c, args.l_x, args.l_x, args.l_x, args.l_phi, args.l_phi, args.l_phi]) * scale

        capofs.append(
            train_capof(
                Xs[i],
                Ys[i],
                data["goal_poses"][i],
                K_sp,
                K_gap,
                alpha_,
                kernel_length_vec_,
                c_dim=1,
                x_dim=3,
                phi_dim=3,
                x_dot_dim=3,
                phi_dot_dim=3,
                mu_add_dim=1,
                n_regression_points=args.n_regression,
            )
        )

    # Run the control loop
    sim = ObjectCentricLoop(
        Xs,
        capofs,
        np.zeros(6),
        seed,
        data["dataset_path"],
        R0s,
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
