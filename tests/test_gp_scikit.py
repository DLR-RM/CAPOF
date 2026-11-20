"""Tests for capof.gp_scikit.GP (fit, predict, epistemic queries)."""

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF

from capof.gp_scikit import GP


@pytest.fixture
def fitted_gp() -> GP:
    """A small fitted GP on 1D data (y = sin)."""
    rng = np.random.default_rng(0)
    X = np.sort(rng.uniform(0, 1, size=30)).reshape(-1, 1)
    Y = np.sin(2 * np.pi * X).ravel()

    model = GP(kernel=RBF(length_scale=0.2), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale
    return model


def test_fit_stores_training_data(fitted_gp: GP) -> None:
    assert fitted_gp.X_train_.shape == (30, 1)
    assert fitted_gp.y_train_.shape == (30,)
    assert fitted_gp.K_ is not None  # epistemic Cholesky factor


def test_predict_shape_and_finite(fitted_gp: GP) -> None:
    X = np.linspace(0, 1, 7).reshape(-1, 1)
    mu = fitted_gp.predict(X)
    assert mu.shape == (7,)
    assert np.isfinite(mu).all()


def test_predict_interpolates_training_data(fitted_gp: GP) -> None:
    # With small noise, the mean should be close to the training targets
    mu = fitted_gp.predict(fitted_gp.X_train_)
    assert np.max(np.abs(mu - fitted_gp.y_train_)) < 0.2


def test_epistemic_uncertainty_shape_and_symmetry(fitted_gp: GP) -> None:
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    epi = fitted_gp.epistemic_uncertainty(X)
    assert epi.shape == (5, 5)
    assert np.allclose(epi, epi.T, atol=1e-10)
    assert np.isfinite(epi).all()


def test_epistemic_uncertainty_bounded_by_prior(fitted_gp: GP) -> None:
    # The epistemic variance at any point is bounded by the prior variance k(x, x)
    X = np.linspace(0, 1, 5).reshape(-1, 1)
    epi = fitted_gp.epistemic_uncertainty(X)
    prior = np.diag(fitted_gp.kernel_(X))
    assert np.all(np.diag(epi) <= prior + 1e-10)
    assert np.all(np.diag(epi) >= -1e-10)


def test_epistemic_gradient_matches_finite_differences(fitted_gp: GP) -> None:
    # d/dx diag(epistemic_uncertainty(x)) should equal the diagonal of the
    # returned gradient tensor
    x0 = np.array([[0.4]])
    grad = fitted_gp.epistemic_gradient(x0)
    assert grad.shape == (1, 1, 1)

    h = 1e-5
    epi_plus = np.diag(fitted_gp.epistemic_uncertainty(x0 + h))[0]
    epi_minus = np.diag(fitted_gp.epistemic_uncertainty(x0 - h))[0]
    fd = (epi_plus - epi_minus) / (2 * h)

    assert grad[0, 0, 0] == pytest.approx(fd, rel=1e-3, abs=1e-5)


def test_epistemic_gradient_matches_finite_differences_multi_query(fitted_gp: GP) -> None:
    # With multiple query points, the diagonal of the gradient tensor must
    # match the finite difference of the pointwise epistemic uncertainties
    x0 = np.array([[0.2], [0.5], [0.8]])
    grad = fitted_gp.epistemic_gradient(x0)
    assert grad.shape == (3, 1, 3)

    h = 1e-5
    for i in range(3):
        xi = x0[[i]]
        epi_plus = np.diag(fitted_gp.epistemic_uncertainty(xi + h))[0]
        epi_minus = np.diag(fitted_gp.epistemic_uncertainty(xi - h))[0]
        fd = (epi_plus - epi_minus) / (2 * h)
        assert grad[i, 0, i] == pytest.approx(fd, rel=1e-3, abs=1e-5)


def test_epistemic_reg_defaults_to_alpha() -> None:
    model = GP(kernel=RBF(length_scale=0.2), alpha=0.05, optimizer=None)
    assert model.epi_reg == 0.05

    model2 = GP(kernel=RBF(length_scale=0.2), alpha=0.05, epi_reg=0.1, optimizer=None)
    assert model2.epi_reg == 0.1


def test_epistemic_uncertainty_multi_target_shape() -> None:
    # With multi-target data the joint epistemic uncertainty has one slice
    # per target: (n_queries, n_queries, n_targets)
    X = np.linspace(0, 1, 15).reshape(-1, 1)
    Y = np.column_stack([np.sin(2 * np.pi * X).ravel(), np.cos(2 * np.pi * X).ravel()])
    model = GP(kernel=RBF(length_scale=0.2), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale

    epi = model.epistemic_uncertainty(X[:4])
    assert epi.shape == (4, 4, 2)
    assert np.isfinite(epi).all()


def test_epistemic_uncertainty_increases_with_epireg() -> None:
    # Stronger epistemic regularization (K + sigma_epi^2 I) trusts the data
    # less, so the epistemic uncertainty grows at every query point
    X = np.sort(np.linspace(0, 1, 25)).reshape(-1, 1)
    Y = np.sin(2 * np.pi * X).ravel()
    query = np.array([[0.13], [0.37], [0.62], [0.87]])

    epi = []
    for epi_reg in (0.01, 1.0):
        model = GP(kernel=RBF(length_scale=0.2), alpha=0.01, epi_reg=epi_reg, optimizer=None)
        model.fit(X, Y)
        model.l = model.kernel_.length_scale
        epi.append(np.diag(model.epistemic_uncertainty(query)))

    assert np.all(epi[1] > epi[0])


def test_non_vector_input_kernel_branch() -> None:
    # A kernel with requires_vector_input=False takes the non-numeric
    # validation path in the epistemic queries
    class _ScalarKernel(RBF):
        requires_vector_input = False

    X = np.linspace(0, 1, 20).reshape(-1, 1)
    Y = np.sin(2 * np.pi * X).ravel()
    model = GP(kernel=_ScalarKernel(length_scale=0.2), alpha=0.01, optimizer=None)
    model.fit(X, Y)
    model.l = model.kernel_.length_scale

    epi = model.epistemic_uncertainty(X[:5])
    assert epi.shape == (5, 5)
    assert np.isfinite(epi).all()

    grad = model.epistemic_gradient(X[:5])
    assert grad.shape == (5, 1, 5)
    assert np.isfinite(grad).all()
