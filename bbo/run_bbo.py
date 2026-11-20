#!/usr/bin/env python3
"""BBO Hyperparameter Optimization for CAPOF GP hyperparameters.

Optimizes: alpha, l/x, K_sp/x, K_gap/x, l/c, l/phi, K_sp/phi, K_gap/phi
Supports: lasa, robomim_object_centric, robomim_force_conditioned

Usage:
    python bbo/run_bbo.py --task lasa --method cmaes --n-trials 200
    python bbo/run_bbo.py --task robomim_object_centric --method cmaes --n-trials 200
    python bbo/run_bbo.py --task robomim_force_conditioned --method bayes --n-trials 200

    To resume: same command with same --output-dir will continue from last trial
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
from optuna.samplers import CmaEsSampler
from skopt import gp_minimize
from skopt.space import Real

# Ensure repo root and bbo dir are on the path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "bbo") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "bbo"))

from black_box_functions import (  # noqa: E402
    lasa_black_box_function,
    robomim_force_conditioned_black_box_function,
    robomim_object_centric_black_box_function,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ============================================================================
# Task configuration
# ============================================================================

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "lasa": {
        "pbounds": {
            "alpha": (1e-4, 1e-1),
            "l/x": (1e-2, 1.0),
            "K_sp/x": (1e-2, 1.0),
            "K_gap/x": (1e-2, 1.0),
        },
        "black_box_function": lasa_black_box_function,
    },
    "robomim_object_centric": {
        "pbounds": {
            "alpha": (1e-4, 1e-1),
            "l/x": (1e-2, 1.0),
            "l/c": (1e-2, 1.0),
            "l/phi": (1e-2, 1.0),
            "K_sp/x": (1e-2, 1.0),
            "K_sp/phi": (1e-2, 1.0),
            "K_gap/x": (1e-2, 1.0),
            "K_gap/phi": (1e-2, 1.0),
        },
        "black_box_function": robomim_object_centric_black_box_function,
    },
    "robomim_force_conditioned": {
        "pbounds": {
            "alpha": (1e-4, 1e-1),
            "l/x": (1e-2, 1.0),
            "l/c": (1e-2, 1.0),
            "l/phi": (1e-2, 1.0),
            "K_sp/x": (1e-2, 1.0),
            "K_sp/phi": (1e-2, 1.0),
            "K_gap/x": (1e-2, 1.0),
            "K_gap/phi": (1e-2, 1.0),
        },
        "black_box_function": robomim_force_conditioned_black_box_function,
    },
}


# ============================================================================
# Resume / checkpoint utilities
# ============================================================================


def load_previous_optimization(output_dir: str) -> Optional[Dict]:
    """Load previous optimization state if it exists."""
    output_path = Path(output_dir)
    history_file = output_path / "optimization_history.json"
    results_file = output_path / "optimization_results.json"

    if not history_file.exists() or not results_file.exists():
        return None

    try:
        with open(history_file, "r") as f:
            history = json.load(f)
        with open(results_file, "r") as f:
            results = json.load(f)
        return {
            "history": history,
            "results": results,
            "n_completed_trials": len(history),
            "config": results.get("config", {}),
        }
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load previous optimization state: {e}")
        return None


def validate_resume_compatibility(previous_state: Dict, task: str, method: str, pbounds: Dict) -> bool:
    """Validate that we can safely resume from previous optimization."""
    config = previous_state.get("config", {})
    if config.get("task") != task:
        logger.warning(f"Task mismatch: previous={config.get('task')}, current={task}")
        return False
    if config.get("method") != method:
        logger.warning(f"Method mismatch: previous={config.get('method')}, current={method}")
        return False
    prev_pbounds = config.get("pbounds", {})
    if prev_pbounds:
        for key, (low, high) in pbounds.items():
            if key not in prev_pbounds:
                logger.warning(f"Parameter '{key}' not in previous bounds")
                return False
            prev_low, prev_high = prev_pbounds[key]
            if abs(prev_low - low) > 1e-10 or abs(prev_high - high) > 1e-10:
                logger.warning(f"Bounds mismatch for '{key}'")
                return False
    return True


def _save_optimization_results(
    best_params: Dict,
    best_objective: float,
    optimization_history: List[Dict],
    output_path: Path,
    task: str,
    method: str,
    n_trials: int,
    seed: int,
    learning_seed: int,
    pbounds: Dict[str, tuple],
):
    """Save optimization results to JSON files."""
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "task": task,
            "method": method,
            "n_trials": n_trials,
            "seed": seed,
            "learning_seed": learning_seed,
            "pbounds": {k: list(v) for k, v in pbounds.items()},
        },
        "best_params_raw": best_params,
        "best_objective": float(best_objective),
        "total_trials": len(optimization_history),
    }

    with open(output_path / "optimization_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    with open(output_path / "optimization_history.json", "w") as f:
        json.dump(optimization_history, f, indent=2)

    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "trial": len(optimization_history) - 1 if optimization_history else -1,
        "best_params": best_params,
        "best_objective": float(best_objective),
        "history_length": len(optimization_history),
        "final": True,
    }
    with open(output_path / "checkpoint.json", "w") as f:
        json.dump(checkpoint, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


# ============================================================================
# Optimization
# ============================================================================


def _cmaes_optimize(
    black_box_function: Callable,
    pbounds: Dict[str, tuple],
    n_trials: int,
    seed: int,
    learning_seed: int,
    output_dir: str,
    task: str,
) -> Dict[str, float]:
    """Run CMA-ES optimization using Optuna."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check for resume
    previous_state = load_previous_optimization(output_dir)
    optimization_history: List[Dict] = []

    if previous_state is not None and validate_resume_compatibility(previous_state, task, "cmaes", pbounds):
        n_completed = previous_state["n_completed_trials"]
        optimization_history = previous_state["history"]
        logger.info(f"Resuming from trial {n_completed} to reach {n_trials} total")

        from optuna.trial import create_trial

        # Rebuild the previous trials so they can (a) seed the study (correct
        # trial numbering + record) and (b) warm-start the CMA-ES sampler, so
        # the resumed run continues the previous search instead of restarting it.
        prev_trials = []
        for trial_info in optimization_history:
            params = trial_info["params"]
            distributions: Dict[str, optuna.distributions.BaseDistribution] = {
                str(key): optuna.distributions.FloatDistribution(pbounds[key][0], pbounds[key][1]) for key in params
            }
            prev_trials.append(
                create_trial(value=float(trial_info["combined_loss"]), params=params, distributions=distributions)
            )

        # source_trials is an experimental Optuna feature; silence its warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            sampler = CmaEsSampler(seed=seed, source_trials=prev_trials, warn_independent_sampling=False)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        for t in prev_trials:
            study.add_trial(t)

        trials_to_run = n_trials - n_completed
        if trials_to_run <= 0:
            logger.info(f"Already completed {n_completed} trials. Skipping.")
            best_idx = min(range(len(optimization_history)), key=lambda i: optimization_history[i]["combined_loss"])
            best_params = optimization_history[best_idx]["params"]
            _save_optimization_results(
                best_params,
                optimization_history[best_idx]["combined_loss"],
                optimization_history,
                output_path,
                task,
                "cmaes",
                n_trials,
                seed,
                learning_seed,
                pbounds,
            )
            return best_params
    else:
        if previous_state is not None:
            logger.warning("Previous optimization incompatible, starting fresh")
        sampler = CmaEsSampler(seed=seed, warn_independent_sampling=False)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        trials_to_run = n_trials

    best_loss = [float("inf")]
    best_params = [None]

    # Seed the running best from any previously completed trials, so a resumed
    # run tracks the overall best (not just the trials added in this run).
    if optimization_history:
        _prev_best_idx = min(range(len(optimization_history)), key=lambda i: optimization_history[i]["combined_loss"])
        best_params[0] = optimization_history[_prev_best_idx]["params"]
        best_loss[0] = optimization_history[_prev_best_idx]["combined_loss"]

    def objective(trial):
        """Evaluate one Optuna trial: sample params, run the black-box function.

        Tracks the best (loss, params) and appends a trial record to the
        in-memory history (persisted by the caller after each trial).
        """
        params = {}
        for key, (low, high) in pbounds.items():
            params[key] = trial.suggest_float(key, low, high)

        loss = black_box_function(params, learning_seed=learning_seed)

        if loss < best_loss[0]:
            best_loss[0] = loss
            best_params[0] = params

        trial_info = {
            "trial": trial.number,
            "params": params,
            "combined_loss": float(loss),
            "timestamp": datetime.now().isoformat(),
        }
        if hasattr(black_box_function, "last_metrics"):
            trial_info["metrics"] = black_box_function.last_metrics

        optimization_history.append(trial_info)
        logger.info(f"  Trial {trial.number}: loss={loss:.4f}, best={best_loss[0]:.4f}")

        # Checkpoint
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "trial": trial.number,
            "best_params": best_params[0],
            "best_objective": float(best_loss[0]),
            "history_length": len(optimization_history),
        }
        with open(output_path / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

        return loss

    logger.info(f"Running CMA-ES: {trials_to_run} trials, output={output_path.absolute()}")
    study.optimize(objective, n_trials=trials_to_run)

    # Determine the overall best from the full history (robust across resumes).
    best_idx = min(range(len(optimization_history)), key=lambda i: optimization_history[i]["combined_loss"])
    best_params_final = optimization_history[best_idx]["params"]
    best_loss_final = optimization_history[best_idx]["combined_loss"]
    logger.info(f"Optimization complete. Best value: {best_loss_final:.6f}")

    _save_optimization_results(
        best_params_final,
        best_loss_final,
        optimization_history,
        output_path,
        task,
        "cmaes",
        n_trials,
        seed,
        learning_seed,
        pbounds,
    )
    return best_params_final


def _bayesian_optimize(
    black_box_function: Callable,
    pbounds: Dict[str, tuple],
    n_trials: int,
    seed: int,
    learning_seed: int,
    output_dir: str,
    task: str,
) -> Dict[str, float]:
    """Run Bayesian Optimization using scikit-optimize."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check for resume
    previous_state = load_previous_optimization(output_dir)
    optimization_history: List[Dict] = []

    if previous_state is not None and validate_resume_compatibility(previous_state, task, "bayes", pbounds):
        resume = True
        n_completed = previous_state["n_completed_trials"]
        optimization_history = previous_state["history"]
        logger.info(f"Resuming Bayesian from trial {n_completed} to reach {n_trials} total")

        x0 = []
        for trial_info in optimization_history:
            params = trial_info["params"]
            x0.append([params.get(key, (pbounds[key][0] + pbounds[key][1]) / 2) for key in pbounds])

        trials_to_run = n_trials - n_completed
        if trials_to_run <= 0:
            logger.info(f"Already completed {n_completed} trials. Skipping.")
            best_idx = min(range(len(optimization_history)), key=lambda i: optimization_history[i]["combined_loss"])
            best_params_completed = optimization_history[best_idx]["params"]
            _save_optimization_results(
                best_params_completed,
                optimization_history[best_idx]["combined_loss"],
                optimization_history,
                output_path,
                task,
                "bayes",
                n_trials,
                seed,
                learning_seed,
                pbounds,
            )
            return best_params_completed
    else:
        resume = False
        if previous_state is not None:
            logger.warning("Previous optimization incompatible, starting fresh")
        x0 = []
        trials_to_run = n_trials

    space = [Real(low, high, prior="uniform", name=key) for key, (low, high) in pbounds.items()]
    param_names = list(pbounds.keys())

    best_params: List[Dict[str, float]] = [{}]
    best_loss = [float("inf")]

    if resume and optimization_history:
        best_idx = min(range(len(optimization_history)), key=lambda i: optimization_history[i]["combined_loss"])
        best_params[0] = optimization_history[best_idx]["params"]
        best_loss[0] = optimization_history[best_idx]["combined_loss"]

    trial_counter = [len(optimization_history) if resume else 0]

    def tracking_objective(params_list):
        """Evaluate one scikit-optimize candidate (same bookkeeping as the CMA-ES objective)."""
        params = dict(zip(param_names, params_list))
        loss = black_box_function(params, learning_seed=learning_seed)

        trial_info = {
            "trial": trial_counter[0],
            "params": params,
            "combined_loss": float(loss),
            "timestamp": datetime.now().isoformat(),
        }
        if hasattr(black_box_function, "last_metrics"):
            trial_info["metrics"] = black_box_function.last_metrics

        optimization_history.append(trial_info)
        logger.info(f"  Trial {trial_counter[0]}: loss={loss:.4f}")

        if loss < best_loss[0]:
            best_loss[0] = loss
            best_params[0] = params

        # Checkpoint every 5 trials
        if trial_counter[0] % 5 == 0:
            checkpoint = {
                "timestamp": datetime.now().isoformat(),
                "trial": trial_counter[0],
                "best_params": best_params[0],
                "best_objective": float(best_loss[0]),
                "history_length": len(optimization_history),
            }
            with open(output_path / "checkpoint.json", "w") as f:
                json.dump(checkpoint, f, indent=2)

        trial_counter[0] += 1
        return loss

    logger.info(f"Running Bayesian: {trials_to_run} trials, output={output_path.absolute()}")

    if resume:
        gp_minimize(
            tracking_objective,
            space,
            x0=x0,
            n_calls=trials_to_run,
            n_random_starts=0,
            random_state=seed,
            acq_func="EI",
            verbose=False,
        )
    else:
        np.random.seed(seed)
        x0 = [np.random.uniform(low, high) for low, high in pbounds.values()]
        gp_minimize(
            tracking_objective,
            space,
            x0=x0,
            n_calls=trials_to_run,
            n_random_starts=max(1, trials_to_run // 5),
            random_state=seed,
            acq_func="EI",
            verbose=False,
        )

    # Determine the overall best from the full history (robust across resumes).
    best_idx = min(range(len(optimization_history)), key=lambda i: optimization_history[i]["combined_loss"])
    best_params_final = optimization_history[best_idx]["params"]
    best_loss_final = optimization_history[best_idx]["combined_loss"]

    _save_optimization_results(
        best_params_final,
        best_loss_final,
        optimization_history,
        output_path,
        task,
        "bayes",
        n_trials,
        seed,
        learning_seed,
        pbounds,
    )
    return best_params_final


# ============================================================================
# Paste-ready output
# ============================================================================


def print_pasteable_params(best_params: Dict[str, float], task: str):
    """Print the best hyperparameters as a ready-to-run command line.

    The printed command uses the corresponding entry script's CLI flags, so it
    can be copied and pasted directly (run from the repository root).
    """
    print("\n" + "=" * 60)
    print(f"PASTE-READY HYPERPARAMETERS FOR: {task}")
    print("=" * 60)

    if task == "lasa":
        print(
            f"python examples/lasa/lasa.py "
            f"--alpha {best_params['alpha']:.4f} "
            f"--l-x {best_params['l/x']:.4f} "
            f"--k-sp {best_params['K_sp/x']:.4f} "
            f"--k-gap {best_params['K_gap/x']:.4f}"
        )
    elif task == "robomim_object_centric":
        print(
            f"python examples/robomim/object_centric.py "
            f"--alpha {best_params['alpha']:.4f} "
            f"--l-c {best_params.get('l/c', 0.1):.4f} "
            f"--l-x {best_params['l/x']:.4f} "
            f"--l-phi {best_params.get('l/phi', 0.1):.4f} "
            f"--k-sp-x {best_params['K_sp/x']:.4f} "
            f"--k-sp-phi {best_params.get('K_sp/phi', 0.5):.4f} "
            f"--k-gap-x {best_params['K_gap/x']:.4f} "
            f"--k-gap-phi {best_params.get('K_gap/phi', 0.5):.4f}"
        )
    elif task == "robomim_force_conditioned":
        print(
            f"python examples/robomim/force_conditioned.py "
            f"--alpha {best_params['alpha']:.4f} "
            f"--l-c {best_params.get('l/c', 0.1):.4f} "
            f"--l-x {best_params['l/x']:.4f} "
            f"--l-phi {best_params.get('l/phi', 0.1):.4f} "
            f"--k-sp-x {best_params['K_sp/x']:.4f} "
            f"--k-sp-phi {best_params.get('K_sp/phi', 0.5):.4f} "
            f"--k-gap-x {best_params['K_gap/x']:.4f} "
            f"--k-gap-phi {best_params.get('K_gap/phi', 0.5):.4f}"
        )
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Parse arguments and run the BBO optimization for the selected task/method."""
    parser = argparse.ArgumentParser(description="BBO Hyperparameter Optimization for CAPOF GP hyperparameters")
    parser.add_argument(
        "--task",
        type=str,
        default="lasa",
        choices=["lasa", "robomim_object_centric", "robomim_force_conditioned"],
    )
    parser.add_argument("--method", type=str, default="cmaes", choices=["cmaes", "bayes"])
    parser.add_argument("--n-trials", type=int, default=200, help="Number of optimization trials")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for optimizer")
    parser.add_argument("--learning-seed", type=int, default=0, help="Seed for policy training / env")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: bbo/bbo_optimization_results_{task})",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"bbo/bbo_optimization_results_{args.task}"

    config = TASK_CONFIG[args.task]
    pbounds = config["pbounds"]
    bb_func = config["black_box_function"]

    logger.info("=" * 60)
    logger.info("BBO GP HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Task:    {args.task}")
    logger.info(f"Method:  {args.method}")
    logger.info(f"Trials:  {args.n_trials}")
    logger.info(f"Seed:    {args.seed}")
    logger.info(f"Learn:   {args.learning_seed}")
    logger.info(f"Output:  {args.output_dir}")
    logger.info(f"Params:  {list(pbounds.keys())}")
    logger.info("=" * 60)

    if args.method == "cmaes":
        best_params = _cmaes_optimize(
            bb_func, pbounds, args.n_trials, args.seed, args.learning_seed, args.output_dir, args.task
        )
    else:
        best_params = _bayesian_optimize(
            bb_func, pbounds, args.n_trials, args.seed, args.learning_seed, args.output_dir, args.task
        )

    logger.info("\nBest parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value:.6f}")

    print_pasteable_params(best_params, args.task)


if __name__ == "__main__":
    main()
