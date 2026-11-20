#!/usr/bin/env python3
"""Analysis and visualization scripts for BBO optimization results.

This script provides tools to analyze and visualize the results from
the unified BBO optimization framework.

Usage (from the repository root):
    python bbo/analyze_bbo_results.py --results-dir bbo/bbo_optimization_results
    python bbo/analyze_bbo_results.py --results-dir bbo/bbo_optimization_results --plot
    python bbo/analyze_bbo_results.py --results-dir bbo/bbo_optimization_results --export-config
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load optimization results from directory."""
    results_path = Path(results_dir) / "optimization_results.json"
    history_path = Path(results_dir) / "optimization_history.json"
    checkpoint_path = Path(results_dir) / "checkpoint.json"

    results = {}

    if results_path.exists():
        with open(results_path, "r") as f:
            results["final"] = json.load(f)

    if history_path.exists():
        with open(history_path, "r") as f:
            results["history"] = json.load(f)

    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            results["checkpoint"] = json.load(f)

    return results


def _get_task_list(final_config: Dict) -> str:
    """Get task list from config, handling both 'task' and 'tasks' keys."""
    if "tasks" in final_config:
        return ", ".join(final_config["tasks"]) if isinstance(final_config["tasks"], list) else final_config["tasks"]
    elif "task" in final_config:
        return final_config["task"]
    else:
        return "unknown"


def print_summary(results: Dict[str, Any]):
    """Print summary of optimization results."""
    print("=" * 60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 60)

    if "final" in results:
        final = results["final"]
        print("\nConfiguration:")
        print(f"  Task: {_get_task_list(final['config'])}")
        print(f"  Method: {final['config']['method']}")
        print(f"  Total trials: {final['total_trials']}")
        print(f"  Best objective: {final['best_objective']:.4f}")

        print("\nBest Parameters:")
        for param, value in final["best_params_raw"].items():
            print(f"  {param}: {value}")

    if "history" in results:
        history = results["history"]
        print("\nOptimization History:")
        print(f"  Total trials recorded: {len(history)}")

        if len(history) > 0:
            losses = [h["combined_loss"] for h in history]
            print(f"  Initial loss: {losses[0]:.4f}")
            print(f"  Final loss: {losses[-1]:.4f}")
            print(f"  Best loss: {min(losses):.4f}")
            print(f"  Improvement: {(losses[0] - min(losses)) / losses[0] * 100:.1f}%")

    if "checkpoint" in results:
        checkpoint = results["checkpoint"]
        print("\nLatest Checkpoint:")
        print(f"  Timestamp: {checkpoint['timestamp']}")
        print(f"  Best objective: {checkpoint['best_objective']:.4f}")

    print("\n" + "=" * 60)


def plot_results(results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot optimization results including task-specific metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    if "history" not in results or len(results["history"]) == 0:
        print("No optimization history found for plotting")
        return

    history = results["history"]
    trials = [h["trial"] for h in history]
    losses = [h["combined_loss"] for h in history]

    # Running best
    running_best = np.minimum.accumulate(losses)

    # All tasks (lasa, robomim FC/OC) report the same metrics:
    # avg_success / avg_mmd / avg_iterations.
    has_metrics = "metrics" in history[0]

    # 3x2 when metrics are available (loss + success/mmd/combined), else 2x2 (loss only)
    n_rows = 3 if has_metrics else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 12 if has_metrics else 10))

    # Plot 1: Loss over trials (always present)
    ax1 = axes[0, 0]
    ax1.plot(trials, losses, "b-", alpha=0.5, label="Trial loss", marker="o", markersize=3)
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Loss")
    ax1.set_title("Optimization Progress (Combined Loss)")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Running best convergence
    ax2 = axes[0, 1]
    ax2.plot(trials, running_best, "g-", linewidth=2, marker="s", markersize=3, label="Running best")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Best Loss")
    ax2.set_title("Convergence (Best Loss So Far)")
    ax2.legend()
    ax2.grid(True)

    if has_metrics:
        # All current tasks report the same metrics, so these are always present.
        successes = [h["metrics"].get("avg_success", np.nan) for h in history]
        mmds = [h["metrics"].get("avg_mmd", np.nan) for h in history]

        # Plot 3: Success rate
        ax3 = axes[1, 0]
        ax3.plot(trials, successes, "b-", alpha=0.7, label="Success rate", marker="o", markersize=3)
        ax3.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="Perfect (1.0)")
        ax3.set_xlabel("Trial")
        ax3.set_ylabel("Success Rate")
        ax3.set_title("Success Rate Over Trials")
        ax3.legend()
        ax3.grid(True)
        ax3.set_ylim(0, 1.1)

        # Plot 4: MMD
        ax4 = axes[1, 1]
        ax4.plot(trials, mmds, "g-", alpha=0.7, label="MMD", marker="^", markersize=3)
        ax4.set_xlabel("Trial")
        ax4.set_ylabel("MMD")
        ax4.set_title("Mean Minimum Distance")
        ax4.legend()
        ax4.grid(True)

        # Plot 5: Combined metrics view
        ax5 = axes[2, 0]
        ax5.plot(trials, [1 - s for s in successes], "b-", alpha=0.5, label="1-Success", marker="o", markersize=2)
        ax5.plot(trials, mmds, "g-", alpha=0.5, label="MMD", marker="^", markersize=2)
        ax5.set_xlabel("Trial")
        ax5.set_ylabel("Metric Value")
        ax5.set_title("Inverse Success vs MMD")
        ax5.legend()
        ax5.grid(True)

        # Hide the unused 3x2 slot
        axes[2, 1].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_parameter_evolution(results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot parameter evolution over trials."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    if "history" not in results or len(results["history"]) == 0:
        print("No optimization history found for plotting")
        return

    history = results["history"]
    trials = [h["trial"] for h in history]

    # Get all parameters
    all_params = set()
    for h in history:
        all_params.update(h["params"].keys())

    if not all_params:
        print("No parameters found in history")
        return

    # Create figure
    n_params = len(all_params)
    fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(4 * ((n_params + 1) // 2), 3 * 2))
    axes = np.array(axes).flatten() if n_params > 1 else [axes]

    for i, param in enumerate(sorted(all_params)):
        values = [h["params"].get(param, np.nan) for h in history]
        axes[i].plot(trials, values, "b-", alpha=0.7, marker="o", markersize=3)
        axes[i].set_xlabel("Trial")
        axes[i].set_ylabel("Value")
        axes[i].set_title(f"{param}")
        axes[i].grid(True)

    # Hide any unused subplots
    for i in range(len(all_params), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Parameter evolution plot saved to: {save_path}")
    else:
        plt.show()


def export_config(results: Dict[str, Any], output_path: str):
    """Export best parameters as a configuration file."""
    if "final" not in results:
        print("No final results found")
        return

    final = results["final"]

    config = {
        "moe_hyperparameters": final["best_params_raw"],
        "optimization_info": {
            "method": final["config"]["method"],
            "n_trials": final["config"]["n_trials"],
            "best_objective": final["best_objective"],
            "timestamp": final["timestamp"],
        },
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration exported to: {output_path}")


def compare_runs(results_dirs: List[str]):
    """Compare multiple optimization runs."""
    print("=" * 60)
    print("COMPARING OPTIMIZATION RUNS")
    print("=" * 60)

    for results_dir in results_dirs:
        results = load_results(results_dir)
        print(f"\n{results_dir}:")

        if "final" in results:
            final = results["final"]
            print(f"  Task: {_get_task_list(final['config'])}")
            print(f"  Method: {final['config']['method']}")
            print(f"  Trials: {final['total_trials']}")
            print(f"  Best loss: {final['best_objective']:.4f}")
        else:
            print("  No results found")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze and visualize BBO optimization results")

    parser.add_argument(
        "--results-dir", type=str, default="bbo/bbo_optimization_results", help="Directory containing optimization results"
    )

    parser.add_argument("--plot", action="store_true", help="Generate optimization progress plots (loss, metrics)")

    parser.add_argument("--plot-params", action="store_true", help="Generate parameter evolution plots")

    parser.add_argument("--plot-save", type=str, default=None, help="Save plot to file (instead of displaying)")

    parser.add_argument("--export-config", type=str, default=None, help="Export best parameters to JSON file")

    parser.add_argument("--compare", nargs="+", default=None, help="Compare multiple optimization runs")

    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
        return

    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)

    if not results:
        print(f"ERROR: No results found in {args.results_dir}")
        return

    # Print summary
    print_summary(results)

    # Generate optimization progress plots
    if args.plot:
        print("\nGenerating optimization progress plots...")
        plot_save_path = f"{args.plot_save}_progress.png" if args.plot_save else str(Path(args.results_dir) / "progress.png")
        plot_results(results, plot_save_path)

    # Generate parameter evolution plots
    if args.plot_params:
        print("\nGenerating parameter evolution plots...")
        plot_save_path = f"{args.plot_save}_params.png" if args.plot_save else str(Path(args.results_dir) / "params.png")
        plot_parameter_evolution(results, plot_save_path)

    # Export configuration
    if args.export_config:
        print("\nExporting configuration...")
        export_config(results, args.export_config)


if __name__ == "__main__":
    main()
