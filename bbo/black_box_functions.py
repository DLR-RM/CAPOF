#!/usr/bin/env python3
"""Black-box functions for GP hyperparameter optimization (BBO).

Optimizes the CAPOF GP hyperparameters:
  - alpha      : GP noise variance
  - l/x        : kernel length scale for x (position) dimensions
  - l/c        : kernel length scale for c (context) dimensions
  - l/phi      : kernel length scale for phi (orientation) dimensions
  - K_sp/x     : stabilizing policy gain for x
  - K_sp/phi   : stabilizing policy gain for phi
  - K_gap/x    : goal attractor gain for x
  - K_gap/phi  : goal attractor gain for phi

All parameters are relative and scaled by the data span inside each
black-box function (the l/*, K_sp/*, K_gap/* parameters are searched in
[1e-2, 1]; alpha in [1e-4, 1e-1]).

Supported tasks:
  - lasa                    : 2D handwriting (c_dim=0, x_dim=2, phi_dim=0)
  - robomim_object_centric  : 7D lift (c_dim=1, x_dim=3, phi_dim=3, two skills)
  - robomim_force_conditioned: 8D lift (c_dim=2, x_dim=3, phi_dim=3, single skill)

The data pipelines and policy training are shared with the entry scripts
(``examples/robomim/data_processing.py``, ``capof/training.py``), so the
optimized hyperparameters can be pasted directly into the corresponding
entry script.
"""

import gc
import logging
import os
import sys
from typing import Any, Dict

import numpy as np

# Ensure repo root is on the path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from capof.training import train_capof  # noqa: E402
from examples.lasa.lasa_data import DEFAULT_LASA_REPO, check_lasa_repo, load_lasa_letter  # noqa: E402
from examples.robomim.data_processing import load_force_conditioned_data, load_object_centric_data  # noqa: E402
from examples.robomim.robomim_interface import ForceConditionedLoop, ObjectCentricLoop  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(processName)s-%(process)d] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Datasets
# The LASA dataset is not shipped with the repository; it must be cloned into
# examples/lasa/dataset (see examples/lasa/lasa_data.py for the clone command)
# The robomimic dataset is not shipped either; it must be generated with
# examples/robomim/convert_dataset.py (see the README for the download source)
# ============================================================================
LASA_REPO = str(DEFAULT_LASA_REPO)
ROBOMIM_DATASET = os.path.join(_REPO_ROOT, "examples", "robomim", "datasets", "lift_ph_v15.hdf5")

# ============================================================================
# Data caches (loaded once per process, reused across trials)
# ============================================================================
_data_cache: Dict[str, Any] = {}

# ============================================================================
# LASA
# ============================================================================
LASA_N_LETTERS = 5  # Number of letters to use per trial
LASA_N_STARTS = 10  # Number of start points per letter
LASA_N_REGRESSION = 500
LASA_LOOP_LIMIT = 500


def _get_lasa_data() -> Dict[str, Any]:
    """Load and cache a random subset of LASA letters."""
    if "lasa" in _data_cache:
        return _data_cache["lasa"]

    check_lasa_repo(LASA_REPO)

    all_folders = sorted(f for f in os.listdir(LASA_REPO) if f.endswith(".mat"))
    chosen = np.random.choice(all_folders, LASA_N_LETTERS, replace=False)

    letters = []
    for folder_name in chosen:
        letter = load_lasa_letter(os.path.join(LASA_REPO, folder_name))

        # Start points: random within data bounds
        span_x = np.max(letter["X"], axis=0) - np.min(letter["X"], axis=0)
        start_points = np.random.rand(LASA_N_STARTS, 2) * span_x + np.min(letter["X"], axis=0)

        letters.append({**letter, "start_points": start_points})

    data = {"letters": letters}
    _data_cache["lasa"] = data
    logger.info(f"LASA data loaded: {len(letters)} letters")
    return data


def lasa_black_box_function(params: Dict[str, float], learning_seed: int = 0) -> float:
    """LASA black-box function.

    Args:
        params: dict with keys: alpha, l/x, K_sp/x, K_gap/x
        learning_seed: seed for random start points

    Returns:
        Average loss over letters and start points.
    """
    from examples.lasa.lasa_interface import LasaOuterControlLoop

    np.random.seed(learning_seed)
    data = _get_lasa_data()

    all_losses = []
    all_success = []
    all_mmd = []
    all_iterations = []

    for letter in data["letters"]:
        X, Y = letter["X"], letter["Y"]
        goal_poses = letter["goal_poses"]
        start_points = letter["start_points"]

        # Scale relative hyperparameters per-letter
        alpha = params["alpha"] * np.max(np.var(Y, axis=0))
        max_abs_velocity = np.max(np.abs(Y))
        K_sp_x = params["K_sp/x"] * max_abs_velocity
        K_gap_x = params["K_gap/x"] * max_abs_velocity
        span_X = np.max(X, axis=0) - np.min(X, axis=0)
        kernel_length_vec = np.array([params["l/x"], params["l/x"]]) * span_X

        policy = train_capof(
            X,
            Y,
            goal_poses,
            [K_sp_x],
            [K_gap_x],
            alpha,
            kernel_length_vec,
            c_dim=0,
            x_dim=2,
            phi_dim=0,
            x_dot_dim=2,
            phi_dot_dim=0,
            mu_add_dim=0,
            n_regression_points=LASA_N_REGRESSION,
        )

        span = np.linalg.norm(span_X)

        # Create sim once per letter (headless)
        sim = LasaOuterControlLoop(
            X,
            policy,
            np.zeros(2),
            letter["demo_start_poses"],
            control_rate=20,
            loop_limit=LASA_LOOP_LIMIT,
            threshold=0.01,
            policy_combination="MoE",
        )

        for start in start_points:
            sim.run_loop(start.copy())

            success = sim.success
            iterations = len(sim.visited_states)
            mmd = min(
                np.mean(np.min(np.linalg.norm(np.array(sim.visited_states)[:, np.newaxis] - X, axis=-1), axis=1)) / span, 1.0
            )

            loss = (1 - success) + iterations / LASA_LOOP_LIMIT + mmd
            all_losses.append(loss)
            all_success.append(success)
            all_mmd.append(mmd)
            all_iterations.append(iterations)

    lasa_black_box_function.last_metrics = {  # type: ignore[attr-defined]
        "avg_success": float(np.mean(all_success)),
        "avg_mmd": float(np.mean(all_mmd)),
        "avg_iterations": float(np.mean(all_iterations)),
    }
    return float(np.mean(all_losses))


# ============================================================================
# RoboMimic Object-Centric (two-skill: grasp + lift)
# ============================================================================
ROBOMIM_N_SEEDS = 10  # Number of env seeds per trial
ROBOMIM_N_REGRESSION = 500
ROBOMIM_LOOP_LIMIT = 250


def _get_robomim_oc_data() -> Dict[str, Any]:
    """Load and cache RoboMimic object-centric data (shared pipeline with the entry script)."""
    if "robomim_oc" in _data_cache:
        return _data_cache["robomim_oc"]

    data = load_object_centric_data(ROBOMIM_DATASET)
    _data_cache["robomim_oc"] = data
    logger.info("RoboMimic object-centric data loaded")
    return data


def robomim_object_centric_black_box_function(params: Dict[str, float], learning_seed: int = 100) -> float:
    """RoboMimic object-centric black-box function.

    Args:
        params: dict with keys: alpha, l/x, l/c, l/phi, K_sp/x, K_sp/phi, K_gap/x, K_gap/phi
        learning_seed: base seed for env

    Returns:
        Average loss (1 - reward) over env seeds.
    """
    data = _get_robomim_oc_data()

    # Scale relative hyperparameters
    l_x = params["l/x"]
    l_c = params["l/c"]
    l_phi = params["l/phi"]
    K_sp = np.array([params["K_sp/x"] * np.max(data["Y_scale"][:3]), params["K_sp/phi"] * np.max(data["Y_scale"][3:])])
    K_gap = np.array([params["K_gap/x"] * np.max(data["Y_scale"][:3]), params["K_gap/phi"] * np.max(data["Y_scale"][3:])])

    kernel_length_vec = np.array([l_c, l_x, l_x, l_x, l_phi, l_phi, l_phi])

    X, Y = data["Xs"], data["Ys"]
    goal_poses = data["goal_poses"]
    R0 = data["R0s"]

    losses = []
    rewards_list = []
    iterations_list = []
    mmd_list = []

    for i_seed in range(ROBOMIM_N_SEEDS):
        seed = learning_seed + i_seed
        np.random.seed(seed)

        # Build two GP models (grasp + lift) with per-skill relative scaling
        capofs = []
        for i in range(2):
            alpha_ = params["alpha"] * np.max(np.var(Y[i], axis=0))
            scale = np.max(X[i], axis=0) - np.min(X[i], axis=0)
            scale = np.where(scale == 0, 1.0, scale)  # Prevent 0 multiplication for l_c
            kernel_length_vec_ = kernel_length_vec * scale

            capofs.append(
                train_capof(
                    X[i],
                    Y[i],
                    goal_poses[i],
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
                    n_regression_points=ROBOMIM_N_REGRESSION,
                )
            )

        sim = ObjectCentricLoop(
            X,
            capofs,
            np.zeros(6),
            seed,
            data["dataset_path"],
            R0,
            control_rate=20,
            loop_limit=ROBOMIM_LOOP_LIMIT,
            threshold=0.01,
            policy_combination="MoE",
            record_images=False,
        )
        sim.run_loop()

        S = sim.rewards[-1] if sim.rewards else 0.0
        iters = len(sim.visited_states)
        # MMD over the full MoE input (c + x + phi) against the demo data X.
        # The c component (index 0) of each visited state indicates which skill
        # (and thus which X mode / frame) it belongs to, so states are only
        # compared against the matching X[i]. Distances are normalized by the
        # bounding-box span of the corresponding X[i].
        visited = np.array(sim.visited_states)
        mmds = []
        for i in range(2):
            mask = visited[:, 0] == i
            if not mask.any():
                continue
            span_i = np.linalg.norm(np.max(X[i], axis=0) - np.min(X[i], axis=0))
            mmds.append(
                min(
                    np.mean(np.min(np.linalg.norm(visited[mask][:, np.newaxis, :] - X[i][np.newaxis, :, :], axis=-1), axis=1))
                    / span_i,
                    1.0,
                )
            )
        mmd = np.mean(mmds) if mmds else 0.0
        loss = (1.0 - S) + iters / ROBOMIM_LOOP_LIMIT + mmd

        rewards_list.append(S)
        iterations_list.append(iters)
        mmd_list.append(mmd)
        losses.append(loss)

        # Free the MuJoCo model and offscreen render context before the next seed
        sim.env.close()
        del sim
        gc.collect()

    robomim_object_centric_black_box_function.last_metrics = {  # type: ignore[attr-defined]
        "avg_success": float(np.mean(rewards_list)),
        "avg_iterations": float(np.mean(iterations_list)),
        "avg_mmd": float(np.mean(mmd_list)),
    }
    return float(np.mean(losses))


# ============================================================================
# RoboMimic Force-Conditioned (single skill)
# ============================================================================


def _get_robomim_fc_data() -> Dict[str, Any]:
    """Load and cache RoboMimic force-conditioned data (shared pipeline with the entry script)."""
    if "robomim_fc" in _data_cache:
        return _data_cache["robomim_fc"]

    data = load_force_conditioned_data(ROBOMIM_DATASET)
    _data_cache["robomim_fc"] = data
    logger.info("RoboMimic force-conditioned data loaded")
    return data


def robomim_force_conditioned_black_box_function(params: Dict[str, float], learning_seed: int = 100) -> float:
    """RoboMimic force-conditioned black-box function.

    Args:
        params: dict with keys: alpha, l/x, l/c, l/phi, K_sp/x, K_sp/phi, K_gap/x, K_gap/phi
        learning_seed: base seed for env

    Returns:
        Average loss (1 - reward) over env seeds.
    """
    data = _get_robomim_fc_data()

    X, Y = data["X"], data["Y"]
    goal_poses = data["goal_poses"]
    R0 = data["R0"]
    span = np.linalg.norm(np.max(X, axis=0) - np.min(X, axis=0))

    # Scale relative hyperparameters
    alpha = params["alpha"] * np.max(np.var(Y, axis=0))
    l_x = params["l/x"]
    l_c = params["l/c"]
    l_phi = params["l/phi"]
    K_sp = np.array([params["K_sp/x"] * np.max(data["Y_scale"][:3]), params["K_sp/phi"] * np.max(data["Y_scale"][3:])])
    K_gap = np.array([params["K_gap/x"] * np.max(data["Y_scale"][:3]), params["K_gap/phi"] * np.max(data["Y_scale"][3:])])

    kernel_length_vec = np.array([l_c, l_c, l_x, l_x, l_x, l_phi, l_phi, l_phi])
    kernel_length_vec *= np.max(X, axis=0) - np.min(X, axis=0)

    losses = []
    rewards_list = []
    iterations_list = []
    mmd_list = []

    for i_seed in range(ROBOMIM_N_SEEDS):
        seed = learning_seed + i_seed
        np.random.seed(seed)

        capof = train_capof(
            X,
            Y,
            goal_poses,
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
            n_regression_points=ROBOMIM_N_REGRESSION,
        )

        sim = ForceConditionedLoop(
            X,
            capof,
            np.zeros(6),
            seed,
            data["dataset_path"],
            R0,
            control_rate=20,
            loop_limit=ROBOMIM_LOOP_LIMIT,
            threshold=0.01,
            policy_combination="MoE",
            record_images=False,
        )
        sim.run_loop()
        S = sim.rewards[-1] if sim.rewards else 0.0
        iters = len(sim.visited_states)
        visited = np.array(sim.visited_states)
        mmd = min(
            np.mean(np.min(np.linalg.norm(visited[:, np.newaxis, :] - X[np.newaxis, :, :], axis=-1), axis=1)) / span, 1.0
        )
        loss = (1.0 - S) + iters / ROBOMIM_LOOP_LIMIT + mmd

        rewards_list.append(S)
        iterations_list.append(iters)
        mmd_list.append(mmd)
        losses.append(loss)

        # Free the MuJoCo model and offscreen render context before the next seed
        sim.env.close()
        del sim
        gc.collect()

    robomim_force_conditioned_black_box_function.last_metrics = {  # type: ignore[attr-defined]
        "avg_success": float(np.mean(rewards_list)),
        "avg_iterations": float(np.mean(iterations_list)),
        "avg_mmd": float(np.mean(mmd_list)),
    }
    return float(np.mean(losses))
