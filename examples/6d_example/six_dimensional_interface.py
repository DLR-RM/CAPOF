"""Six-Dimensional outer control loop interface for CAPOF trajectory reproduction.

This module provides the :class:`SixDimensionalOuterControlLoop` class, which extends
:class:`OuterControlLoop` to handle 6D pose representations (position + orientation)
as homogeneous transformation matrices. It integrates predicted velocity twists to
reproduce demonstrated motions.

See Also:
    OuterControlLoop: Base class providing the control loop structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from capof.outer_control_loop import OuterControlLoop


class SixDimensionalOuterControlLoop(OuterControlLoop):
    """Six-Dimensional outer control loop for CAPOF trajectory reproduction.

    Extends :class:`OuterControlLoop` to work with 6D pose representations using
    homogeneous transformation matrices (4x4). This class integrates predicted
    6D velocity twists to iteratively reproduce demonstrated motions from pose
    estimates.

    Args:
        X: ndarray of shape (n_samples, 6). Demonstrated poses as [position,
            rotvec] concatenations.
        capof: The CAPOF policy instance for computing velocity commands.
        offset: ndarray of shape (6,). Absolute offset added to the success
            threshold for each dimension.
        control_rate: float (default 20). Control loop frequency in Hz
            (loop_time = 1 / control_rate).
        loop_limit: int (default 500). Maximum number of control loop iterations.
        threshold: float (default 0.01). Convergence threshold for the control loop.
        policy_combination: str (default "MoE"). Policy combination method passed
            to the parent class.

    Attributes:
        loop_time: Time step between control iterations.
        visited_states: History of visited TCP frames during the control loop.

    See Also:
        OuterControlLoop: Base class providing the generic control loop structure.
    """

    def __init__(
        self,
        X: np.ndarray,
        capof: Any,
        offset: np.ndarray,
        control_rate: float = 20,
        loop_limit: int = 500,
        threshold: float = 0.01,
        policy_combination: str = "MoE",
    ) -> None:
        super().__init__(X, capof, offset, loop_limit=loop_limit, threshold=threshold, policy_combination=policy_combination)

        self.loop_time = 1 / control_rate

    def run_loop(self, initial_position: np.ndarray) -> None:
        """Run the control loop from ``initial_position`` (a 4x4 TCP frame).

        The loop integrates the CAPOF velocity commands into the TCP frame
        (simulated motion) until the goal is reached or ``loop_limit`` is hit.
        """
        self.visited_states: list[np.ndarray] = []

        # Initialize process parameters
        self._parameter_initialization()

        self.tcp_frame = initial_position

        self._state_based_execution_loop()

    def _read_current_tcp_frame(self) -> None:
        """Read the current TCP frame (no-op for simulation).

        In simulation mode, the TCP frame is set externally via ``initial_position``
        and updated by the control loop, so this method does nothing.
        """

    def _preprocess_data(self) -> None:
        """Preprocess data before control computation (no-op).

        No additional preprocessing is required for the 6D simulation case.
        """

    def _postprocess_data(self) -> None:
        """Postprocess data after control computation (no-op).

        No additional postprocessing is required for the 6D simulation case.
        """

    def _check_for_limits(self) -> None:
        """Check if operational limits are exceeded (no-op).

        In simulation mode, no physical limits are enforced.
        """

    def _publish_vel_info(self) -> None:
        """Store the current TCP frame in the visited states history.

        Appends a copy of the current TCP frame to :attr:`visited_states`
        for later trajectory visualization and analysis.
        """
        self.visited_states.append(self.tcp_frame.copy())

    def _send_command_to_robot(self) -> None:
        """Integrate the predicted twist into the TCP frame (simulated motion)."""
        self.tcp_frame[: self.capof.x_dim, 3] += self.vel_twist[: self.capof.x_dim] * self.loop_time
        self.tcp_frame[:3, :3] = R.from_rotvec(self.vel_twist[3:] * self.loop_time).as_matrix() @ self.tcp_frame[:3, :3]

    def _safe_shutdown(self) -> None:
        """Perform safe shutdown procedures (no-op).

        In simulation mode, no shutdown procedures are required.
        """

    def _check_for_external_stop(self) -> None:
        """Check for external stop signals (no-op).

        In simulation mode, there is no external stop mechanism.
        """
