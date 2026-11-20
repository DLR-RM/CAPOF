"""LASA handwriting outer control loop for CAPOF visualization and execution.

This module provides the LasaOuterControlLoop class, which extends OuterControlLoop
for the LASA handwriting domain, along with helper functions for creating
visualization grids and plotting vector fields.

The LasaOuterControlLoop interfaces CAPOF policies with a simulated 2D robotic
system, visualizing demonstrated trajectories, learned vector fields, and
mixing coefficients for different policy combinations.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from capof.outer_control_loop import OuterControlLoop


class LasaOuterControlLoop(OuterControlLoop):
    """Outer control loop specialized for LASA 2D handwriting trajectories.

    Extends OuterControlLoop with LASA-specific initialization, creating
    visualization grids for vector fields and mixing coefficient contours.
    Supports both headless execution via :meth:`run_loop` and animated
    visualization via :meth:`animate`.

    Args:
        X: ndarray of shape (n_samples, 2). Demonstration input data (2D positions).
        capof: The CAPOF policy instance combining LfD with stabilizing and goal
            attractors.
        offset: ndarray of shape (2,). Offset added to success thresholds in each
            dimension.
        start_poses: ndarray of shape (n_starts, 2). Starting poses for evaluation
            trajectories.
        control_rate: int (default 20). Control loop frequency in Hz (determines
            loop_time per iteration).
        loop_limit: int (default 500). Maximum number of control iterations before
            forced termination.
        threshold: float (default 0.01). Relative distance threshold for goal reach
            detection.
        policy_combination: str (default "MoE"). Policy combination method: 'LfD',
            'Stabilizing', 'Goal', 'LfD + Stabilizing', 'LfD + Goal', or 'MoE'.
        animate: bool (default False). If True, precompute the vector-field /
            mixing-coefficient grids used by :meth:`animate` (not needed for headless
            :meth:`run_loop`).
    """

    def __init__(
        self,
        X: np.ndarray,
        capof: Any,
        offset: np.ndarray,
        start_poses: np.ndarray,
        control_rate: float = 20,
        loop_limit: int = 500,
        threshold: float = 0.01,
        policy_combination: str = "MoE",
        animate: bool = False,
    ) -> None:
        super().__init__(X, capof, offset, loop_limit=loop_limit, threshold=threshold, policy_combination=policy_combination)

        self.start_poses = start_poses
        self.loop_time = 1 / control_rate

        if animate:
            # Precompute the vector fields and mixing coefficients for plotting
            grids = create_grids(self.capof.model.X_train_)
            self.grids = fill_grids(self.capof, self.policy_combination, grids)

    def run_loop(self, initial_position: np.ndarray) -> None:
        """Execute the control loop without visualization (headless mode).

        Runs the full state-based execution loop, logging visited states for
        later analysis but without creating any plots or animation.

        Args:
            initial_position: array-like of shape (2,). Starting position for the
                TCP frame.
        """
        self.visited_states: list[np.ndarray] = []

        # Initialize process parameters
        self._parameter_initialization()

        self.tcp_frame = np.eye(4)
        self.tcp_frame[: self.capof.x_dim, 3] = initial_position

        self._state_based_execution_loop()

    def animate(self, initial_position: np.ndarray) -> None:
        """Set up animated visualization of the control loop execution.

        Creates a matplotlib figure with vector field contours, streamlines,
        demonstration points, start poses, and goal poses, then runs the
        control loop with animation. The animation is stored in self.anim
        and should be saved or displayed by the caller.

        Args:
            initial_position: array-like of shape (2,). Starting position for the
                TCP frame.
        """
        self.point_artist: list[Any] = []
        self.visited_states = []

        self.fig = plt.figure(figsize=(6.5, 5))
        self.axis = self.fig.add_subplot(1, 1, 1)

        # Set labels
        self.axis.set_xlabel("$x$")
        self.axis.set_ylabel("$y$")
        self.axis.set_aspect("equal")

        # Set axis limits
        x_lim, y_lim = get_limits(self.capof.model.X_train_)
        self.axis.set_xlim(x_lim)
        self.axis.set_ylim(y_lim)

        # Plot vector fields and mixing coefficients
        plot_vector_fields(self.fig, self.axis, self.grids, self.policy_combination)

        # Plot demonstrated data
        self.axis.plot(
            self.capof.model.X_train_[:, 0],
            self.capof.model.X_train_[:, 1],
            "k.",
            markersize=2.0,
            zorder=-1,
            label="Demonstrations",
        )
        self.axis.plot(self.start_poses[:, 0], self.start_poses[:, 1], "g*", markersize=8.0, label="Start poses")
        self.axis.plot(self.capof.goal_poses[:, 0], self.capof.goal_poses[:, 1], "r*", markersize=8.0, label="Goal poses")

        # Initialize process parameters
        self._parameter_initialization()

        self.tcp_frame = np.eye(4)
        self.tcp_frame[: self.capof.x_dim, 3] = initial_position

        # Animation loop (frames is a generator function; matplotlib calls it lazily)
        self.anim = FuncAnimation(self.fig, self._update, frames=self._gen, repeat=False, save_count=self.loop_limit)  # type: ignore[arg-type]

    def _read_current_tcp_frame(self) -> None:
        """No-op: the TCP frame is the 2D pen position, set by the loop itself."""

    def _preprocess_data(self) -> None:
        """No-op: 2D positions need no frame transform."""

    def _postprocess_data(self) -> None:
        """No-op: 2D velocities need no frame transform."""

    def _check_for_limits(self) -> None:
        """No-op: the 2D pen has no actuation limits."""

    def _publish_vel_info(self) -> None:
        """Log visited states and update point artist for animation.

        Appends the current TCP position to visited_states for success
        evaluation and, if in animation mode, records the point artist
        for trajectory visualization.
        """
        self.visited_states.append(self.tcp_frame[: self.capof.x_dim, 3].copy())

        if hasattr(self, "point_artist"):
            self.point_artist.append(self.axis.plot(self.tcp_frame[0, 3], self.tcp_frame[1, 3], "ro", markersize=3.0))

    def _send_command_to_robot(self) -> None:
        """Send velocity command to robot by integrating twist into TCP frame.

        Updates the TCP frame position by integrating the computed velocity
        twist over the loop time step.
        """
        self.tcp_frame[: self.capof.x_dim, 3] += self.vel_twist[: self.capof.x_dim] * self.loop_time

    def _safe_shutdown(self) -> None:
        """No-op: nothing to shut down in simulation."""

    def _check_for_external_stop(self) -> None:
        """No-op: no external stop mechanism in simulation."""


def get_limits(X: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute axis limits for 2D data visualization with padding.

    Calculates equal-width limits for both axes based on the maximum span
    of the data, adding 10% padding on each side.

    Args:
        X: ndarray of shape (n_samples, 2). 2D data points.

    Returns:
        tuple: (x_lim, y_lim) each as (min, max) tuples.
    """
    max_span = np.max(np.max(X, axis=0) - np.min(X, axis=0))
    x_lim = (
        np.min(X, axis=0)[0] - (max_span - (np.max(X, axis=0) - np.min(X, axis=0))[0]) / 2 - 0.1 * max_span,
        np.max(X, axis=0)[0] + (max_span - (np.max(X, axis=0) - np.min(X, axis=0))[0]) / 2 + 0.1 * max_span,
    )
    y_lim = (
        np.min(X, axis=0)[1] - (max_span - (np.max(X, axis=0) - np.min(X, axis=0))[1]) / 2 - 0.1 * max_span,
        np.max(X, axis=0)[1] + (max_span - (np.max(X, axis=0) - np.min(X, axis=0))[1]) / 2 + 0.1 * max_span,
    )
    return x_lim, y_lim


def create_grids(
    X: np.ndarray, density: float = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create grid arrays for vector field visualization.

    Sets up meshgrid arrays and zero-filled arrays for the epistemic
    uncertainty (epi), the goal-attractor kernel activation (k_max), and the
    velocity (dx, dy) data, sized according to the density parameter.

    Args:
        X: ndarray of shape (n_samples, 2). 2D demonstration data for determining
            axis limits.
        density: float (default 2). Scaling factor for grid resolution. The base
            grid is 30x30.

    Returns:
        tuple: (xx_g, yy_g, epi_grid, k_max_grid, dx_grid, dy_grid) meshes and arrays.
    """
    N_grid = int(density * 30)  # streamplots: the domain is divided into a 30x30 grid. density linearly scales this grid
    x_lim, y_lim = get_limits(X)

    xx_g, yy_g = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], N_grid),
        np.linspace(y_lim[0], y_lim[1], N_grid),
    )

    epi_grid = np.zeros((N_grid, N_grid))
    k_max_grid = np.zeros((N_grid, N_grid))
    dx_grid = np.zeros((N_grid, N_grid))
    dy_grid = np.zeros((N_grid, N_grid))
    return (xx_g, yy_g, epi_grid, k_max_grid, dx_grid, dy_grid)


def fill_grids(
    policy: Any, policy_combination: str, grids: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the policy on a grid to fill velocity and coefficient arrays.

    Iterates over the grid points, querying the CAPOF policy at each location
    to compute the mean velocity, the epistemic uncertainty, and the k_max
    (goal-attractor) coefficient.

    Args:
        policy: The policy to evaluate on the grid.
        policy_combination: Policy combination method.
        grids: (xx_g, yy_g, epi_grid, k_max_grid, dx_grid, dy_grid) from create_grids.

    Returns:
        tuple: (xx_g, yy_g, epi_grid, k_max_grid, dx_grid, dy_grid) with filled values.
    """
    xx_g, yy_g, epi_grid, k_max_grid, dx_grid, dy_grid = grids
    N_grid = epi_grid.shape[0]

    for i in range(0, N_grid):
        for j in range(0, N_grid):
            query_point = np.array([xx_g[i, j], yy_g[i, j]])

            # Compute MoE
            mu, mu_add, (mu_sp, mu_gap, epi, k_max) = policy.compute_moe(query_point.reshape(1, -1), policy_combination)

            mu, epi_grid[i, j], k_max_grid[i, j] = mu[0], epi[0], k_max[0]

            # Save velocities
            dx_grid[i, j] = mu[0]
            dy_grid[i, j] = mu[1]

    return (xx_g, yy_g, epi_grid, k_max_grid, dx_grid, dy_grid)


def plot_vector_fields(
    fig: Any,
    axis: Any,
    grids: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    policy_combination: str,
    density: float = 2,
    colorbar: bool = True,
    cmap: str = "Reds",
) -> None:
    """Plot vector field streamlines with coefficient contours.

    Visualizes the learned vector field as streamlines, with background contours
    showing either the stabilizing policy coefficient (pi_sp) or goal attractor
    coefficient (pi_gp) depending on the policy combination mode.

    Args:
        fig: matplotlib.figure.Figure. Figure object for colorbar placement.
        axis: matplotlib.axes.Axes. Axes object to plot on.
        grids: (xx_g, yy_g, epi_grid, k_max_grid, dx_grid, dy_grid) filled grid data.
        policy_combination: Determines which coefficient to visualize and the
            colormap used.
        density: float (default 2). Streamline density parameter.
        colorbar: bool (default True). Whether to add colorbars for coefficient
            contours.
        cmap: str (default "Reds"). Colormap for stabilizing policy contours.
    """
    xx_g, yy_g, epi_grid, k_max_grid, dx_grid, dy_grid = grids
    if policy_combination in ("LfD + Stabilizing", "Stabilizing"):
        contour_epi = axis.contourf(xx_g, yy_g, epi_grid * (1 - k_max_grid), alpha=0.3, cmap=cmap, zorder=-1, levels=10)
        # axis.contour(xx_g, yy_g, epi_grid, [0.5], colors="r", linestyles="dashed", linewidths=2, zorder=-1)
        if colorbar:
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(contour_epi, orientation="vertical", cax=cax)
            cbar.set_label(r"Stabilizing policy coefficient $\pi_{sp}$")
    elif policy_combination in ("LfD + Goal", "Goal"):
        contour_k_max = axis.contourf(xx_g, yy_g, k_max_grid, alpha=0.4, cmap="Blues", zorder=-1, levels=10)
        # axis.contour(xx_g, yy_g, k_max_grid, [0.5], colors="b", linestyles="dashed", linewidths=2, zorder=-1)
        if colorbar:
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(contour_k_max, orientation="vertical", cax=cax)
            cbar.set_label(r"Goal attractor policy coefficient $\pi_{gp}$")
    elif policy_combination == "MoE":
        contour_epi = axis.contourf(xx_g, yy_g, epi_grid * (1 - k_max_grid), alpha=0.5, cmap=cmap, zorder=-1, levels=10)
        contour_k_max = axis.contourf(xx_g, yy_g, k_max_grid, alpha=0.4, cmap="Blues", zorder=-1, levels=10)
        # axis.contour(xx_g, yy_g, epi_grid, [0.5], colors="r", linestyles="dashed", linewidths=2, zorder=-1)
        # axis.contour(xx_g, yy_g, k_max_grid, [0.5], colors="b", linestyles="dashed", linewidths=2, zorder=-1)
        if colorbar:
            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(contour_epi, orientation="vertical", cax=cax)
            cbar.set_label(r"Stabilizing policy coefficient $\pi_{sp}$")
            cax = divider.append_axes("right", size="5%", pad=0.8)
            cbar = fig.colorbar(contour_k_max, orientation="vertical", cax=cax)
            cbar.set_label(r"Goal attractor policy coefficient $\pi_{gp}$")

    stream = axis.streamplot(xx_g, yy_g, dx_grid, dy_grid, density=density, linewidth=3, color="k", zorder=-1)
    stream.lines.set_alpha(0.25)
    for art in axis.get_children():
        if not isinstance(art, matplotlib.patches.FancyArrowPatch):
            continue
        art.set_alpha(0.25)
