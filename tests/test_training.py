"""Tests for capof.training.train_capof (subsampling and parameter passthrough)."""

import numpy as np

from capof.capof import ContextAdaptivePolicyFramework
from capof.training import train_capof


def _helical_demo(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """A helical 7D demonstration [c, x, phi] with (x_dot, phi_dot) velocities."""
    t = np.linspace(0, 1, n)
    c = np.zeros((n, 1))
    x = np.column_stack([t, 0.1 * np.sin(2 * np.pi * t), 0.05 * np.cos(2 * np.pi * t)])
    phi = 0.3 * t[:, None] * np.array([0.0, 0.0, 1.0])
    X = np.concatenate([c, x, phi], axis=1)
    Y = np.gradient(X[:, 1:], 1 / (n - 1), axis=0)
    return X, Y


def _train(
    X: np.ndarray,
    Y: np.ndarray,
    n_regression_points: int = 500,
    epi_reg: float | None = None,
) -> ContextAdaptivePolicyFramework:
    return train_capof(
        X,
        Y,
        goal_poses=X[-1:],
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
        n_regression_points=n_regression_points,
        epi_reg=epi_reg,
    )


def test_train_capof_subsamples_evenly() -> None:
    # The demonstrations are subsampled to at most n_regression_points
    # evenly spaced points (by index), keeping the first and last frame
    X, Y = _helical_demo(60)
    capof = _train(X, Y, n_regression_points=10)

    expected_idx = np.round(np.linspace(0, len(X) - 1, 10)).astype(int)
    assert capof.model.X_train_.shape == (10, 7)
    assert np.array_equal(capof.model.X_train_, X[expected_idx])
    assert np.array_equal(capof.model.y_train_, Y[expected_idx])
    # The fixed length scales are stored on the model and the goals passed through
    assert np.array_equal(capof.model.l, np.full(7, 0.2))
    assert np.array_equal(capof.goal_poses, X[-1:])


def test_train_capof_keeps_all_points_when_below_limit() -> None:
    X, Y = _helical_demo(20)
    capof = _train(X, Y, n_regression_points=500)
    assert capof.model.X_train_.shape == (20, 7)
    assert np.array_equal(capof.model.X_train_, X)


def test_train_capof_epireg_passthrough() -> None:
    X, Y = _helical_demo(20)
    # An explicit epi_reg is used as-is
    capof = _train(X, Y, n_regression_points=20, epi_reg=0.05)
    assert capof.model.epi_reg == 0.05

    # By default epi_reg falls back to alpha
    capof_default = _train(X, Y, n_regression_points=20)
    assert capof_default.model.epi_reg == 0.01
