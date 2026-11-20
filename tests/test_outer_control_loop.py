"""Tests for capof.outer_control_loop.OuterControlLoop (loop mechanics)."""

from typing import Callable

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from sklearn.gaussian_process.kernels import RBF

from capof.capof import ContextAdaptivePolicyFramework
from capof.gp_scikit import GP
from capof.outer_control_loop import OuterControlLoop


def _make_2d_capof() -> ContextAdaptivePolicyFramework:
    """A small 2D CAPOF on a curved path from (0,0) to (1,0)."""
    t = np.linspace(0, 1, 40)
    X = np.column_stack([t, 0.05 * np.sin(4 * np.pi * t)])
    Y = np.gradient(X, 1 / (len(t) - 1), axis=0)  # velocities w.r.t. t
    model = GP(kernel=RBF(length_scale=np.array([0.2, 0.1])), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    return ContextAdaptivePolicyFramework(
        model,
        K_sp=[0.5],
        K_gap=[0.5],
        length_scale=model.l,
        c_dim=0,
        x_dim=2,
        phi_dim=0,
        x_dot_dim=2,
        phi_dot_dim=0,
        mu_add_dim=0,
        goal_poses=X[-1:],
    )


def _make_7d_capof() -> ContextAdaptivePolicyFramework:
    """A small 7D CAPOF (c_dim=1, x_dim=3, phi_dim=3) on a helical path."""
    t = np.linspace(0, 1, 30)
    c = np.zeros((len(t), 1))
    x = np.column_stack([t, 0.1 * np.sin(2 * np.pi * t), 0.05 * np.cos(2 * np.pi * t)])
    phi = 0.3 * t[:, None] * np.array([0.0, 0.0, 1.0])
    X = np.concatenate([c, x, phi], axis=1)
    Y = np.gradient(X[:, 1:], 1 / (len(t) - 1), axis=0)
    model = GP(kernel=RBF(length_scale=np.full(7, 0.2)), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    return ContextAdaptivePolicyFramework(
        model,
        K_sp=[0.5, 0.5],
        K_gap=[0.5, 0.5],
        length_scale=model.l,
        c_dim=1,
        x_dim=3,
        phi_dim=3,
        x_dot_dim=3,
        phi_dot_dim=3,
        mu_add_dim=0,
        goal_poses=X[-1:],
    )


class FakeLoop(OuterControlLoop):
    """Minimal concrete loop: a point mass driven by the policy in 2D."""

    def __init__(
        self, capof: ContextAdaptivePolicyFramework, start: np.ndarray, loop_limit: int = 300, offset: np.ndarray | None = None
    ) -> None:
        # X for threshold normalization: the demonstration data
        super().__init__(
            capof.model.X_train_, capof, np.zeros(2) if offset is None else offset, loop_limit=loop_limit, threshold=0.01
        )
        self.loop_time = 0.05  # 20 Hz
        self.pos = start.copy()
        self.visited: list[np.ndarray] = []
        self.shutdowns = 0

    def _read_current_tcp_frame(self) -> None:
        self.tcp_frame[:2, 3] = self.pos

    def _read_current_task_parameters(self, blocking: bool) -> None:
        self.task_parameters = np.array([])

    def _read_current_hand_pose(self, blocking: bool) -> None:
        pass

    def _preprocess_data(self) -> None:
        pass

    def _postprocess_data(self) -> None:
        pass

    def _check_for_limits(self) -> None:
        pass

    def _publish_vel_info(self) -> None:
        self.visited.append(self.pos.copy())

    def _send_command_to_robot(self) -> None:
        self.pos = self.pos + self.vel_twist[:2] * self.loop_time

    def _safe_shutdown(self) -> None:
        self.shutdowns += 1

    def _check_for_external_stop(self) -> None:
        pass


def test_parameter_initialization() -> None:
    capof = _make_2d_capof()
    sim = FakeLoop(capof, np.array([0.0, 0.0]))
    sim._parameter_initialization()
    assert sim.stop_activated is False
    assert sim.success is False
    assert sim.success_counter == 0
    assert sim.task_parameters.shape == (0,)


def test_loop_reaches_goal() -> None:
    capof = _make_2d_capof()
    # The y-dimension of the demo has a small span, so add an offset to the
    # success threshold (as done for real tasks with tight tolerances)
    sim = FakeLoop(capof, np.array([0.0, 0.0]), loop_limit=300, offset=np.array([0.0, 0.01]))
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)
    sim._state_based_execution_loop()

    assert sim.success is True
    assert sim.shutdowns >= 1  # safe shutdown is always called after the loop
    # The final position should be close to the goal
    goal = capof.goal_poses[0]
    assert np.linalg.norm(sim.pos - goal) < 0.1


def test_loop_stops_at_loop_limit() -> None:
    capof = _make_2d_capof()
    # Start at the goal but with a policy that cannot keep it there: use a
    # very small loop limit so the success counter (10 frames) never completes
    sim = FakeLoop(capof, capof.goal_poses[0].copy(), loop_limit=5)
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)
    sim._state_based_execution_loop()

    assert sim.success is False
    assert len(sim.visited) == 5


def test_success_requires_consecutive_frames() -> None:
    capof = _make_2d_capof()
    sim = FakeLoop(capof, np.array([0.0, 0.0]))
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)

    # Place the state exactly at the goal
    sim.lfd_input = capof.goal_poses[0].copy()

    for _ in range(9):
        sim._check_for_success()
    assert sim.success is False
    assert sim.success_counter == 9

    sim._check_for_success()
    assert sim.success is True


def test_success_counter_resets_when_off_goal() -> None:
    capof = _make_2d_capof()
    sim = FakeLoop(capof, np.array([0.0, 0.0]))
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)

    sim.lfd_input = capof.goal_poses[0].copy()
    for _ in range(5):
        sim._check_for_success()
    assert sim.success_counter == 5

    # Move away from the goal: the counter must reset
    sim.lfd_input = np.array([0.0, 0.0])
    sim._check_for_success()
    assert sim.success_counter == 0
    assert sim.success is False


def test_value_safety_check_rejects_nan() -> None:
    capof = _make_2d_capof()
    sim = FakeLoop(capof, np.array([0.0, 0.0]))
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)

    sim.vel_twist = np.array([np.nan, 0.0])
    with pytest.raises(TypeError, match="velocity twist"):
        sim._value_safety_check()
    assert sim.shutdowns >= 1  # safe shutdown triggered


def test_value_safety_check_rejects_wrong_shape() -> None:
    capof = _make_2d_capof()
    sim = FakeLoop(capof, np.array([0.0, 0.0]))
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)

    sim.vel_twist = np.array([0.0, 0.0, 0.0])  # wrong shape
    with pytest.raises(TypeError, match="velocity twist"):
        sim._value_safety_check()


def test_value_safety_check_accepts_valid_twist() -> None:
    capof = _make_2d_capof()
    sim = FakeLoop(capof, np.array([0.0, 0.0]))
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)

    sim.vel_twist = np.array([0.1, -0.2])
    sim._value_safety_check()  # should not raise


def test_loop_deterministic_same_start_same_steps() -> None:
    # Running the loop from the same start point must take exactly the same
    # number of steps and follow the same trajectory (deterministic policy)
    capof = _make_2d_capof()
    trajectories = []
    for _ in range(2):
        sim = FakeLoop(capof, np.array([0.0, 0.0]), loop_limit=300, offset=np.array([0.0, 0.01]))
        sim._parameter_initialization()
        sim.tcp_frame = np.eye(4)
        sim._state_based_execution_loop()
        assert sim.success is True
        trajectories.append(np.array(sim.visited))

    assert trajectories[0].shape == trajectories[1].shape  # same number of steps
    assert np.array_equal(trajectories[0], trajectories[1])  # identical trajectory


def _make_context_capof() -> ContextAdaptivePolicyFramework:
    """A CAPOF with c_dim=1 and mu_add_dim=1 (task parameter + gripper target)."""
    t = np.linspace(0, 1, 40)
    c = np.full((len(t), 1), 0.5)
    x = np.column_stack([t, 0.05 * np.sin(4 * np.pi * t)])
    X = np.concatenate([c, x], axis=1)
    Y = np.column_stack([np.gradient(x, 1 / (len(t) - 1), axis=0), np.ones((len(t), 1))])
    model = GP(kernel=RBF(length_scale=np.array([0.2, 0.2, 0.1])), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    return ContextAdaptivePolicyFramework(
        model,
        K_sp=[0.5],
        K_gap=[0.5],
        length_scale=model.l,
        c_dim=1,
        x_dim=2,
        phi_dim=0,
        x_dot_dim=2,
        phi_dot_dim=0,
        mu_add_dim=1,
        goal_poses=X[-1:],
    )


class FakeContextLoop(OuterControlLoop):
    """Point-mass loop with task parameters (c_dim=1) and a hand pose."""

    def __init__(
        self,
        capof: ContextAdaptivePolicyFramework,
        start: np.ndarray,
        loop_limit: int = 300,
        offset: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            capof.model.X_train_, capof, np.zeros(2) if offset is None else offset, loop_limit=loop_limit, threshold=0.01
        )
        self.loop_time = 0.05
        self.pos = start.copy()
        self.task_parameter_reads: list[bool] = []
        self.hand_pose_reads: list[bool] = []

    def _read_current_tcp_frame(self) -> None:
        self.tcp_frame[:2, 3] = self.pos

    def _read_current_task_parameters(self, blocking: bool) -> None:
        self.task_parameter_reads.append(blocking)
        self.task_parameters = np.array([0.5])

    def _read_current_hand_pose(self, blocking: bool) -> None:
        self.hand_pose_reads.append(blocking)
        self.hand_pose = np.array([0.0])

    def _preprocess_data(self) -> None:
        pass

    def _postprocess_data(self) -> None:
        pass

    def _check_for_limits(self) -> None:
        pass

    def _publish_vel_info(self) -> None:
        pass

    def _send_command_to_robot(self) -> None:
        self.pos = self.pos + self.vel_twist[:2] * self.loop_time

    def _safe_shutdown(self) -> None:
        pass

    def _check_for_external_stop(self) -> None:
        pass


def test_context_loop_reads_task_parameters_and_hand_pose() -> None:
    # With c_dim > 0 and mu_add_dim > 0 the loop reads the task parameters and
    # the hand pose: blocking during initialization, non-blocking per frame
    capof = _make_context_capof()
    sim = FakeContextLoop(capof, np.array([0.0, 0.0]), loop_limit=20, offset=np.array([0.0, 0.01]))
    sim._parameter_initialization()
    assert sim.task_parameter_reads == [True]
    assert sim.hand_pose_reads == [True]

    sim.tcp_frame = np.eye(4)
    sim._state_based_execution_loop()

    n_frames = len(sim.task_parameter_reads) - 1
    assert n_frames > 0
    assert all(b is False for b in sim.task_parameter_reads[1:])
    assert all(b is False for b in sim.hand_pose_reads[1:])
    assert len(sim.task_parameter_reads) == len(sim.hand_pose_reads)


def test_send_command_failure_triggers_shutdown_and_reraise() -> None:
    # A failure while sending the command must trigger the safe shutdown and
    # propagate the original exception
    capof = _make_2d_capof()

    class FailingLoop(FakeLoop):
        def _send_command_to_robot(self) -> None:
            raise RuntimeError("robot link lost")

    sim = FailingLoop(capof, np.array([0.0, 0.0]))
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)

    with pytest.raises(RuntimeError, match="robot link lost"):
        sim._state_based_execution_loop()
    assert sim.shutdowns == 1  # safe shutdown triggered by the failed send


def test_abstract_methods_raise_not_implemented() -> None:
    # The base class is a template: every robot-facing method must raise
    # NotImplementedError until a subclass implements it
    capof = _make_2d_capof()
    loop = OuterControlLoop(capof.model.X_train_, capof, np.zeros(2))

    unimplemented: list[tuple[Callable, tuple]] = [
        (loop._read_current_tcp_frame, ()),
        (loop._read_current_task_parameters, (True,)),
        (loop._read_current_hand_pose, (True,)),
        (loop._read_current_image_frame, (True,)),
        (loop._preprocess_data, ()),
        (loop._postprocess_data, ()),
        (loop._check_for_limits, ()),
        (loop._publish_vel_info, ()),
        (loop._send_command_to_robot, ()),
        (loop._safe_shutdown, ()),
        (loop._check_for_external_stop, ()),
    ]
    for method, args in unimplemented:
        with pytest.raises(NotImplementedError):
            method(*args)


def test_external_stop_terminates_loop() -> None:
    # An external stop signal terminates the loop early: no success is
    # reported and the safe shutdown is executed
    capof = _make_2d_capof()

    class StopLoop(FakeLoop):
        def __init__(self, capof: ContextAdaptivePolicyFramework, start: np.ndarray, stop_after: int) -> None:
            super().__init__(capof, start)
            self.stop_after = stop_after
            self.frames = 0

        def _check_for_external_stop(self) -> None:
            self.frames += 1
            if self.frames >= self.stop_after:
                self.stop_activated = True

    sim = StopLoop(capof, np.array([0.0, 0.0]), stop_after=3)
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)
    sim._state_based_execution_loop()

    assert len(sim.visited) == 3  # stopped after exactly 3 frames
    assert sim.success is False
    assert sim.stop_activated is True
    assert sim.shutdowns == 1


def test_create_lfd_input_combines_task_parameters_and_rotvec() -> None:
    # The LfD input is [c, x, rotvec(phi)]: the position comes from the TCP
    # frame, the orientation is converted from the rotation matrix to a
    # rotation vector
    capof = _make_7d_capof()
    loop = OuterControlLoop(capof.model.X_train_, capof, np.zeros(3))

    theta = 0.7
    tcp_frame = np.eye(4)
    tcp_frame[:3, :3] = R.from_euler("z", theta).as_matrix()
    tcp_frame[:3, 3] = [0.4, 0.1, -0.2]
    loop.tcp_frame = tcp_frame
    loop.task_parameters = np.array([0.25])

    loop._create_LfD_input()

    expected = np.concatenate([[0.25], [0.4, 0.1, -0.2], R.from_euler("z", theta).as_rotvec()])
    assert loop.lfd_input.shape == (7,)
    assert np.allclose(loop.lfd_input, expected, atol=1e-12)


def test_success_threshold_scales_with_data_span_and_offset() -> None:
    # The success distance is threshold * data span + offset, per dimension
    capof = _make_2d_capof()
    sim = FakeLoop(capof, np.array([0.0, 0.0]))
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)

    goal = capof.goal_poses[0]
    span = np.max(capof.model.X_train_, axis=0) - np.min(capof.model.X_train_, axis=0)

    # Inside the x-threshold: the counter increments
    sim.lfd_input = goal - np.array([0.5 * sim.threshold * span[0], 0.0])
    sim.success_counter = 0
    sim._check_for_success()
    assert sim.success_counter == 1

    # Outside the x-threshold: the counter must not increment
    sim.lfd_input = goal - np.array([2.0 * sim.threshold * span[0], 0.0])
    sim.success_counter = 0
    sim._check_for_success()
    assert sim.success_counter == 0

    # The offset enlarges the threshold: the same point now counts
    sim.offset = np.array([2.0 * sim.threshold * span[0], 0.0])
    sim._check_for_success()
    assert sim.success_counter == 1


def test_value_safety_check_rejects_inf_wrong_dtype_and_non_array() -> None:
    capof = _make_2d_capof()
    sim = FakeLoop(capof, np.array([0.0, 0.0]))
    sim._parameter_initialization()
    sim.tcp_frame = np.eye(4)

    for vel_twist in (np.array([0.1, np.inf]), np.array([0.1, 0.2], dtype=np.float32), [0.1, 0.2]):
        sim.vel_twist = vel_twist  # type: ignore[assignment]  # intentionally invalid
        with pytest.raises(TypeError, match="velocity twist"):
            sim._value_safety_check()
