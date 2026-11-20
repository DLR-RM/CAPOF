"""Tests for capof.capof.ContextAdaptivePolicyFramework (policy combination)."""

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF

from capof.capof import ContextAdaptivePolicyFramework
from capof.gp_scikit import GP
from capof.training import train_capof

ALL_COMBINATIONS = ["LfD", "Stabilizing", "Goal", "LfD + Stabilizing", "LfD + Goal", "MoE"]


def _make_2d_capof() -> ContextAdaptivePolicyFramework:
    """Build a small 2D CAPOF (c_dim=0, x_dim=2, phi_dim=0) on a curved path."""
    t = np.linspace(0, 1, 40)
    X = np.column_stack([t, 0.05 * np.sin(4 * np.pi * t)])
    Y = np.gradient(X, 1 / (len(t) - 1), axis=0)  # velocities w.r.t. t
    goal_poses = X[-1:]
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
        goal_poses=goal_poses,
    )


def _make_7d_capof() -> ContextAdaptivePolicyFramework:
    """Build a 7D CAPOF (c_dim=1, x_dim=3, phi_dim=3) on a helical path."""
    t = np.linspace(0, 1, 60)
    c = np.zeros((len(t), 1))
    x = np.column_stack([t, 0.1 * np.sin(2 * np.pi * t), 0.05 * np.cos(2 * np.pi * t)])
    phi = 0.3 * t[:, None] * np.array([0.0, 0.0, 1.0])
    X = np.concatenate([c, x, phi], axis=1)
    Y = np.gradient(X[:, 1:], 1 / (len(t) - 1), axis=0)  # velocities of (x, phi) w.r.t. t
    goal_poses = X[-1:]
    return train_capof(
        X,
        Y,
        goal_poses,
        K_sp=[0.5, 0.5],
        K_gap=[0.5, 0.5],
        alpha=0.01,
        kernel_length_vec=np.full(7, 0.2),
        c_dim=1,
        x_dim=3,
        phi_dim=3,
        x_dot_dim=3,
        phi_dot_dim=3,
        mu_add_dim=0,
        n_regression_points=60,
    )


def test_compute_moe_shapes_2d() -> None:
    capof = _make_2d_capof()
    X = np.array([[0.3, 0.02], [0.7, -0.03]])
    for comb in ALL_COMBINATIONS:
        mu, mu_add, (mu_sp, mu_gap, epi, k_max) = capof.compute_moe(X, comb)
        assert mu.shape == (2, 2), comb
        assert mu_add.shape == (2, 0), comb
        assert mu_sp.shape == (2, 2), comb
        assert mu_gap.shape == (2, 2), comb
        assert epi.shape == (2,), comb
        assert k_max.shape == (2,), comb
        assert np.isfinite(mu).all(), comb


def test_compute_moe_shapes_7d() -> None:
    capof = _make_7d_capof()
    X = capof.model.X_train_[[5, 30]]
    for comb in ALL_COMBINATIONS:
        mu, mu_add, (mu_sp, mu_gap, epi, k_max) = capof.compute_moe(X, comb)
        assert mu.shape == (2, 6), comb
        assert epi.shape == (2,), comb
        assert k_max.shape == (2,), comb
        assert np.isfinite(mu).all(), comb


def test_epistemic_reduction_models_created_for_phi() -> None:
    capof = _make_7d_capof()
    # With phi_dim > 0 and epistemic reduction, the auxiliary models exist
    assert hasattr(capof, "model_c_x")
    assert hasattr(capof, "model_c_phi")


def test_k_max_in_unit_interval() -> None:
    capof = _make_2d_capof()
    X = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    _, _, (_, _, epi, k_max) = capof.compute_moe(X, "MoE")
    assert np.all(k_max >= 0.0) and np.all(k_max <= 1.0)
    assert np.all(epi >= -1e-12)


def test_goal_attractor_points_towards_goal() -> None:
    capof = _make_2d_capof()
    # Query far from the goal: the goal-attractor mean should point towards it
    X = np.array([[0.0, 0.0]])
    _, _, (mu_sp, mu_gap, epi, k_max) = capof.compute_moe(X, "Goal")
    goal = capof.goal_poses[0]
    direction = goal - X[0]
    assert np.dot(mu_gap[0], direction) > 0


def test_invalid_policy_combination_raises() -> None:
    capof = _make_2d_capof()
    with pytest.raises(ValueError, match="policy_comb"):
        capof.compute_moe(np.array([[0.5, 0.0]]), "Bogus")


def test_invalid_goal_policy_method_raises() -> None:
    t = np.linspace(0, 1, 20)
    X = np.column_stack([t, np.zeros_like(t)])
    Y = np.ones_like(X)
    model = GP(kernel=RBF(length_scale=np.array([0.2, 0.2])), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    with pytest.raises(ValueError, match="goal_policy_method"):
        ContextAdaptivePolicyFramework(
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
            goal_policy_method="bogus",
        )


def test_invalid_vel_reduction_raises() -> None:
    t = np.linspace(0, 1, 20)
    X = np.column_stack([t, np.zeros_like(t)])
    Y = np.ones_like(X)
    model = GP(kernel=RBF(length_scale=np.array([0.2, 0.2])), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    with pytest.raises(ValueError, match="stabilizing_policy_vel_reduction"):
        ContextAdaptivePolicyFramework(
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
            stabilizing_policy_vel_reduction="bogus",
        )


def test_goal_policy_controller_method() -> None:
    # The 'controller' goal policy is a linear state feedback: it should be
    # finite and point towards the goal
    t = np.linspace(0, 1, 40)
    X = np.column_stack([t, 0.05 * np.sin(4 * np.pi * t)])
    Y = np.gradient(X, 1 / (len(t) - 1), axis=0)  # velocities w.r.t. t
    model = GP(kernel=RBF(length_scale=np.array([0.2, 0.1])), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    capof = ContextAdaptivePolicyFramework(
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
        goal_policy_method="controller",
    )
    _, _, (mu_sp, mu_gap, epi, k_max) = capof.compute_moe(np.array([[0.0, 0.0]]), "Goal")
    assert np.isfinite(mu_gap).all()
    direction = capof.goal_poses[0] - np.array([0.0, 0.0])
    assert np.dot(mu_gap[0], direction) > 0


def test_mu_add_passthrough() -> None:
    # With mu_add_dim > 0, the additional LfD output is passed through
    t = np.linspace(0, 1, 40)
    X = np.column_stack([t, np.zeros_like(t)])
    Y = np.column_stack([np.ones_like(X), 2.0 * np.ones_like(X[:, :1])])  # (x_dot, mu_add)
    model = GP(kernel=RBF(length_scale=np.array([0.2, 0.2])), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    capof = ContextAdaptivePolicyFramework(
        model,
        K_sp=[0.5],
        K_gap=[0.5],
        length_scale=model.l,
        c_dim=0,
        x_dim=2,
        phi_dim=0,
        x_dot_dim=2,
        phi_dot_dim=0,
        mu_add_dim=1,
        goal_poses=X[-1:],
    )
    mu, mu_add, _ = capof.compute_moe(np.array([[0.5, 0.0]]), "LfD")
    assert mu_add.shape == (1, 1)
    # The additional target was constant 2.0, so the LfD mean should be near it
    assert mu_add[0, 0] == pytest.approx(2.0, abs=0.3)


def _mixed_query_points(capof: ContextAdaptivePolicyFramework) -> np.ndarray:
    """Query points on the demonstration data and far away from it."""
    X_train = capof.model.X_train_
    n = len(X_train)
    on_data = X_train[[n // 8, n // 2, 3 * n // 4]]
    far = X_train[[n // 8, n // 2]].copy()
    far[:, capof.c_dim :] += 1.0  # shift the pose part far away from the data
    return np.vstack([on_data, far])


def test_epistemic_zero_on_data_and_one_far() -> None:
    # The epistemic uncertainty is (near) zero on the demonstration data and
    # saturates at the prior variance (1.0 for the unit RBF) far away from it
    capof = _make_2d_capof()
    X_on = capof.model.X_train_[[5, 10, 20]]
    X_far = np.array([[2.0, 0.0], [-1.0, 0.5], [0.5, 2.0]])

    _, _, (_, _, epi_on, _) = capof.compute_moe(X_on, "MoE")
    _, _, (_, _, epi_far, _) = capof.compute_moe(X_far, "MoE")

    assert np.all(epi_on < 0.05)
    assert np.all(epi_far > 0.9)
    assert np.all(epi_far > epi_on)


def test_epistemic_in_unit_interval() -> None:
    # Like k_max, the epistemic uncertainty is a coefficient in the unit
    # interval for the unit-variance RBF (bounded by the prior variance)
    capof = _make_2d_capof()
    X = _mixed_query_points(capof)
    _, _, (_, _, epi, _) = capof.compute_moe(X, "MoE")
    assert np.all(epi >= -1e-12) and np.all(epi <= 1.0 + 1e-12)


def test_moe_activations_sum_to_one_and_in_unit_interval() -> None:
    # The MoE weights (1-epi)(1-k_max), epi(1-k_max) and k_max must form a
    # probability simplex: every activation in [0, 1], summing to 1
    for capof in [_make_2d_capof(), _make_7d_capof()]:
        X = _mixed_query_points(capof)
        _, _, (_, _, epi, k_max) = capof.compute_moe(X, "MoE")

        w_lfd = (1.0 - epi) * (1.0 - k_max)
        w_sp = epi * (1.0 - k_max)
        w_gap = k_max

        for w in (epi, k_max, w_lfd, w_sp, w_gap):
            assert np.all(w >= -1e-12) and np.all(w <= 1.0 + 1e-12)
        assert np.allclose(w_lfd + w_sp + w_gap, 1.0, atol=1e-12)


def test_lfd_fully_activated_on_data() -> None:
    # On the demonstration data (away from the goal) the LfD expert is fully
    # activated while the stabilizing and goal attractor experts are not
    capof = _make_2d_capof()
    X = capof.model.X_train_[[5, 10]]
    _, _, (_, _, epi, k_max) = capof.compute_moe(X, "MoE")

    w_lfd = (1.0 - epi) * (1.0 - k_max)
    w_sp = epi * (1.0 - k_max)
    w_gap = k_max

    assert np.all(w_lfd > 0.99)
    assert np.all(w_sp < 0.01)
    assert np.all(w_gap < 0.01)


def test_stabilizing_fully_activated_far_from_data() -> None:
    # Far from the data (and the goal) the epistemic uncertainty saturates,
    # so the stabilizing expert takes over the mixture
    capof = _make_2d_capof()
    X = np.array([[0.5, 2.0]])
    _, _, (_, _, epi, k_max) = capof.compute_moe(X, "MoE")

    w_lfd = (1.0 - epi) * (1.0 - k_max)
    w_sp = epi * (1.0 - k_max)
    w_gap = k_max

    assert np.all(w_sp > 0.99)
    assert np.all(w_lfd < 0.01)
    assert np.all(w_gap < 0.01)


def test_stabilizing_policy_points_to_data() -> None:
    # The stabilizing policy drives the state towards well-demonstrated
    # regions: at a far query point it must point towards the nearest
    # training point (analogous to the goal attractor pointing at the goal)
    capof = _make_2d_capof()
    X = np.array([[0.5, 2.0]])
    _, _, (mu_sp, _, _, _) = capof.compute_moe(X, "MoE")

    X_train = capof.model.X_train_
    nearest = X_train[np.argmin(np.linalg.norm(X_train - X[0], axis=1))]
    assert np.dot(mu_sp[0], nearest - X[0]) > 0


def test_kernel_vel_reduction_points_to_data_and_vanishes_on_data() -> None:
    # With the 'kernel' velocity reduction (the phi_dim > 0 branch) the
    # stabilizing policy still points towards the data and vanishes on it,
    # because the reduction factor 1 - k(x, nearest training point) is 0 there
    t = np.linspace(0, 1, 60)
    c = np.zeros((len(t), 1))
    x = np.column_stack([t, 0.1 * np.sin(2 * np.pi * t), 0.05 * np.cos(2 * np.pi * t)])
    phi = 0.3 * t[:, None] * np.array([0.0, 0.0, 1.0])
    X = np.concatenate([c, x, phi], axis=1)
    Y = np.gradient(X[:, 1:], 1 / (len(t) - 1), axis=0)
    model = GP(kernel=RBF(length_scale=np.full(7, 0.2)), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    capof = ContextAdaptivePolicyFramework(
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
        stabilizing_policy_vel_reduction="kernel",
    )

    # On the data the kernel reduction factor is zero -> no stabilizing velocity
    X_on = X[[5, 30]]
    _, _, (mu_sp, _, _, _) = capof.compute_moe(X_on, "MoE")
    assert np.allclose(mu_sp, 0.0, atol=1e-12)

    # Far from the data the stabilizing policy points towards the nearest
    # training point (pose part)
    X_far = np.array([[0.0, 0.5, 2.0, -0.5, 0.0, 0.0, 0.0]])
    _, _, (mu_sp_far, _, _, _) = capof.compute_moe(X_far, "MoE")
    X_train = capof.model.X_train_
    nearest = X_train[np.argmin(np.linalg.norm(X_train - X_far[0], axis=1))]
    assert np.dot(mu_sp_far[0], nearest[capof.c_dim :] - X_far[0][capof.c_dim :]) > 0


def test_goal_policy_controller_with_phi() -> None:
    # The 'controller' goal policy with phi_dim > 0 is finite and points
    # towards the goal in the position part
    t = np.linspace(0, 1, 60)
    c = np.zeros((len(t), 1))
    x = np.column_stack([t, 0.1 * np.sin(2 * np.pi * t), 0.05 * np.cos(2 * np.pi * t)])
    phi = 0.3 * t[:, None] * np.array([0.0, 0.0, 1.0])
    X = np.concatenate([c, x, phi], axis=1)
    Y = np.gradient(X[:, 1:], 1 / (len(t) - 1), axis=0)
    model = GP(kernel=RBF(length_scale=np.full(7, 0.2)), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    capof = ContextAdaptivePolicyFramework(
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
        goal_policy_method="controller",
    )

    X_q = capof.model.X_train_[[5]]
    _, _, (mu_sp, mu_gap, epi, k_max) = capof.compute_moe(X_q, "Goal")
    assert np.isfinite(mu_gap).all()
    goal = capof.goal_poses[0]
    direction = goal[capof.c_dim : capof.c_dim + capof.x_dim] - X_q[0][capof.c_dim : capof.c_dim + capof.x_dim]
    assert np.dot(mu_gap[0, : capof.x_dot_dim], direction) > 0


def test_goal_policy_controller_clips_large_errors_to_unit_direction() -> None:
    # With large position/orientation errors the controller normalizes each
    # part to unit magnitude before applying the gain (saturation branch)
    t = np.linspace(0, 1, 40)
    c = np.zeros((len(t), 1))
    x = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    phi = 2.0 * t[:, None] * np.array([0.0, 0.0, 1.0])
    X = np.concatenate([c, x, phi], axis=1)
    Y = np.gradient(X[:, 1:], 1 / (len(t) - 1), axis=0)
    model = GP(kernel=RBF(length_scale=np.full(7, 0.2)), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    capof = ContextAdaptivePolicyFramework(
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
        goal_policy_method="controller",
    )

    # Query with a large position error (x = 0 vs. goal x = 1) and a large
    # orientation error (phi = 0 vs. goal phi = 2.0 rad)
    X_q = np.zeros((1, 7))
    _, _, (mu_sp, mu_gap, epi, k_max) = capof.compute_moe(X_q, "Goal")
    assert np.isfinite(mu_gap).all()
    # Both parts are clipped to unit direction times the respective gain
    assert np.linalg.norm(mu_gap[0, : capof.x_dot_dim]) == pytest.approx(capof.K_gap[0], abs=1e-12)
    assert np.linalg.norm(mu_gap[0, capof.x_dot_dim :]) == pytest.approx(capof.K_gap[1], abs=1e-12)


def test_combine_policies_unknown_raises() -> None:
    # Calling the private combiner with an invalid method raises (the public
    # compute_moe validates policy_comb before dispatching)
    capof = _make_2d_capof()
    X = np.array([[0.5, 0.0]])
    mu_lfd, mu_add = capof._compute_lfd_policy(X)
    mu_sp, epi = capof._compute_stabilizing_policy(X)
    mu_gap, k_max = capof._compute_goal_attractor_policy(X)
    with pytest.raises(ValueError, match="Unknown policy combination"):
        capof._combine_policies(mu_lfd, mu_sp, mu_gap, mu_add, epi, k_max, "bogus")


def test_goal_attractor_selects_nearest_goal() -> None:
    # With multiple goal poses the attractor is the one with the highest
    # kernel activation: k_max equals the max over the goals and the
    # velocity direction flips with the selected goal
    capof = _make_2d_capof()
    capof.goal_poses = np.array([[1.0, 0.0], [0.0, 0.0]])
    k = RBF(length_scale=capof.length_scale)

    for x_q, expected_goal_idx in (([0.25, 0.0], 1), ([0.75, 0.0], 0)):
        X_q = np.array([x_q])
        _, _, (mu_sp, mu_gap, epi, k_max) = capof.compute_moe(X_q, "Goal")

        k_values = np.array([float(k(X_q, g.reshape(1, -1))[0, 0]) for g in capof.goal_poses])
        assert int(np.argmax(k_values)) == expected_goal_idx
        assert k_max[0] == pytest.approx(k_values.max(), abs=1e-12)
        # The velocity points towards the selected goal (along the x-axis here)
        sign = np.sign(capof.goal_poses[expected_goal_idx, 0] - x_q[0])
        assert np.dot(mu_gap[0], sign * np.array([1.0, 0.0])) > 0


def test_goal_attractor_vanishes_at_goal() -> None:
    # At a goal pose the reduction factor (1 - k) is zero, so the goal
    # attractor contributes no velocity and k_max saturates at 1
    capof = _make_2d_capof()
    X = capof.goal_poses
    _, _, (mu_sp, mu_gap, epi, k_max) = capof.compute_moe(X, "Goal")
    assert np.allclose(mu_gap, 0.0, atol=1e-12)
    assert np.allclose(k_max, 1.0, atol=1e-12)


def test_moe_is_weighted_sum_of_experts() -> None:
    # The combined means must equal the convex combination of the individual
    # expert means with the returned activations as weights
    capof = _make_2d_capof()
    X = np.array([[0.3, 0.02], [0.7, -0.03], [0.5, 2.0]])

    mu_lfd, _, _ = capof.compute_moe(X, "LfD")
    mu_sp, _, _ = capof.compute_moe(X, "Stabilizing")
    mu_gap, _, _ = capof.compute_moe(X, "Goal")
    mu_moe, _, (_, _, epi, k_max) = capof.compute_moe(X, "MoE")

    e, k = epi[:, None], k_max[:, None]
    assert np.array_equal(mu_moe, (1 - e) * (1 - k) * mu_lfd + e * (1 - k) * mu_sp + k * mu_gap)

    mu_ls, _, _ = capof.compute_moe(X, "LfD + Stabilizing")
    assert np.array_equal(mu_ls, (1 - e) * mu_lfd + e * mu_sp)

    mu_lg, _, _ = capof.compute_moe(X, "LfD + Goal")
    assert np.array_equal(mu_lg, (1 - k) * mu_lfd + k * mu_gap)


def test_compute_moe_is_deterministic() -> None:
    # Repeated queries at the same points must return bit-identical results
    capof = _make_2d_capof()
    X = _mixed_query_points(capof)
    mu, mu_add, (mu_sp, mu_gap, epi, k_max) = capof.compute_moe(X, "MoE")
    for _ in range(2):
        mu_r, mu_add_r, (mu_sp_r, mu_gap_r, epi_r, k_max_r) = capof.compute_moe(X, "MoE")
        assert np.array_equal(mu, mu_r)
        assert np.array_equal(mu_add, mu_add_r)
        assert np.array_equal(mu_sp, mu_sp_r)
        assert np.array_equal(mu_gap, mu_gap_r)
        assert np.array_equal(epi, epi_r)
        assert np.array_equal(k_max, k_max_r)
