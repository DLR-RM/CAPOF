from typing import Optional

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from sklearn.gaussian_process import GaussianProcessRegressor

GPR_CHOLESKY_LOWER = True


class GP(GaussianProcessRegressor):
    """Gaussian process regression (GPR) based on scikit implementation.

    Extends :class:`sklearn.gaussian_process.GaussianProcessRegressor` with
    epistemic-uncertainty queries for the joint predictive distribution:

    - :meth:`epistemic_uncertainty`: epistemic uncertainty at query points.
    - :meth:`epistemic_gradient`: its gradient (used by the stabilizing policy).

    Note:
        After :meth:`fit`, callers typically store the (fixed) kernel length
        scales on the model as ``model.l`` (see :mod:`capof.training`). The
        gradient methods read this attribute; if it is a scalar it is broadcast
        to all feature dimensions.

    Args:
        epi_reg: Regularization for the epistemic uncertainty (default None). The
            regularization term σ_epi^2 is used for the epistemic uncertainty
            computation E_epi(ξ(s*)) = k* (K + σ_epi^2I)^-1 μ, without changing
            the covariance prediction of the joint predictive distribution. If
            None is passed, alpha is used as default; in that case the epistemic
            uncertainty is equal to the covariance.
        kernel: Kernel instance (default None). The kernel specifying the
            covariance function of the GP. If None is passed, the kernel
            ``ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0,
            length_scale_bounds="fixed")`` is used as default. Note that the
            kernel hyperparameters are optimized during fitting unless the
            bounds are marked as "fixed".
        alpha: float or ndarray of shape (n_samples,) (default 1e-10). Value added
            to the diagonal of the kernel matrix during fitting. This can prevent
            a potential numerical issue during fitting, by ensuring that the
            calculated values form a positive definite matrix. It can also be
            interpreted as the variance of additional Gaussian measurement noise
            on the training observations. Note that this is different from using a
            `WhiteKernel`. If an array is passed, it must have the same number of
            entries as the data used for fitting and is used as datapoint-dependent
            noise level. Allowing to specify the noise level directly as a
            parameter is mainly for convenience and for consistency with
            :class:`~sklearn.linear_model.Ridge`.
        optimizer: "fmin_l_bfgs_b", callable or None (default "fmin_l_bfgs_b").
            Can either be one of the internally supported optimizers for
            optimizing the kernel's parameters, specified by a string, or an
            externally defined optimizer passed as a callable. If a callable is
            passed, it must have the signature::

                def optimizer(obj_func, initial_theta, bounds):
                    # * 'obj_func': the objective function to be minimized, which
                    #   takes the hyperparameters theta as a parameter and an
                    #   optional flag eval_gradient, which determines if the
                    #   gradient is returned additionally to the function value
                    # * 'initial_theta': the initial value for theta, which can be
                    #   used by local optimizers
                    # * 'bounds': the bounds on the values of theta
                    ....
                    # Returned are the best found hyperparameters theta and
                    # the corresponding value of the target function.
                    return theta_opt, func_min

            Per default, the L-BFGS-B algorithm from `scipy.optimize.minimize`
            is used. If None is passed, the kernel's parameters are kept fixed.
            Available internal optimizers are: `{'fmin_l_bfgs_b'}`.
        n_restarts_optimizer: int (default 0). The number of restarts of the
            optimizer for finding the kernel's parameters which maximize the
            log-marginal likelihood. The first run of the optimizer is performed
            from the kernel's initial parameters, the remaining ones (if any)
            from thetas sampled log-uniform randomly from the space of allowed
            theta-values. If greater than 0, all bounds must be finite. Note
            that `n_restarts_optimizer == 0` implies that one run is performed.
        normalize_y: bool (default False). Whether or not to normalize the target
            values `y` by removing the mean and scaling to unit-variance. This is
            recommended for cases where zero-mean, unit-variance priors are used.
            Note that, in this implementation, the normalisation is reversed
            before the GP predictions are reported.
        copy_X_train: bool (default True). If True, a persistent copy of the
            training data is stored in the object. Otherwise, just a reference to
            the training data is stored, which might cause predictions to change
            if the data is modified externally.
        random_state: int, RandomState instance or None (default None). Determines
            random number generation used to initialize the centers. Pass an int
            for reproducible results across multiple function calls. See
            :term:`Glossary <random_state>`.
    """

    def __init__(self, epi_reg: Optional[float] = None, **kwargs) -> None:
        """Create a GP regressor.

        All :class:`sklearn.gaussian_process.GaussianProcessRegressor`
        parameters are accepted (``kernel``, ``alpha``, ``optimizer``, ...).

        Args:
            epi_reg: Regularization added to the covariance diagonal for the
                epistemic-uncertainty queries. Defaults to ``alpha`` (the
                noise variance), in which case the epistemic uncertainty equals
                the predictive covariance.
            **kwargs: Forwarded to the sklearn parent class.
        """
        super().__init__(**kwargs)

        # Set regularization by default to alpha (epistemic uncertainty = covariance)
        if epi_reg is None:
            self.epi_reg = self.alpha
        else:
            self.epi_reg = epi_reg

    def fit(self, X, y) -> None:
        """Fit Gaussian process regression model.

        Args:
            X: array-like of shape (n_samples, n_features) or list of object.
                Feature vectors or other representations of training data.
            y: array-like of shape (n_samples,) or (n_samples, n_targets). Target values.
        """

        # Use parent class fit function
        super().fit(X, y)

        # Similar to Alg. 2.1, page 19, line 2 -> K = cholesky(K + σ_epi^2 I)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.epi_reg
        self.K_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)

    def epistemic_gradient(self, X) -> np.ndarray:
        """Compute gradient of epistemic uncertainty of joint predictive distribution.

        Args:
            X: array-like of shape (n_samples, n_features) or list of object.
                Feature vectors or other representations of training data.

        Returns:
            epistemic_grad: ndarray of shape (n_samples, n_features, n_samples).
                Gradient of epistemic uncertainty of the joint predictive
                distribution. Only the RBF kernel is implemented for now.
        """

        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False

        X = self._validate_data(X, ensure_2d=ensure_2d, dtype=dtype, reset=False)

        # Similar to Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . beta
        K_trans = self.kernel_(X, self.X_train_)

        # Similar to Alg 2.1, page 19, line 3 -> beta = L^T \ (L \ K(X_test, X_train))
        beta = cho_solve(
            (self.K_, GPR_CHOLESKY_LOWER),
            K_trans.T,
            check_finite=False,
        )

        # Gradient of covariance (RBF kernel!) see https://arxiv.org/pdf/2410.24035
        weighted_diff = K_trans[:, :, None] * (X[:, None, :] - self.X_train_[None, :, :])
        length_scale = np.atleast_1d(self.l)
        if length_scale.shape[0] == 1:
            length_scale = np.full(X.shape[1], length_scale[0])
        K_grad = -np.einsum("ij,njt->nit", np.diag(length_scale ** (-2)), weighted_diff.transpose(0, 2, 1))
        epistemic_grad = -2 * K_grad @ beta

        return epistemic_grad

    def epistemic_uncertainty(self, X) -> np.ndarray:
        """Compute epistemic uncertainty of joint predictive distribution.

        Args:
            X: array-like of shape (n_samples, n_features) or list of object.
                Query points where the GP is evaluated.

        Returns:
            y_epi: ndarray of shape (n_samples, n_samples) or
                (n_samples, n_samples, n_targets). Epistemic uncertainty of the
                joint predictive distribution at query points.
        """

        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False

        X = self._validate_data(X, ensure_2d=ensure_2d, dtype=dtype, reset=False)

        # Similar to Alg 2.1, page 19, line 5 -> v = K \ K(X_test, X_train)^T
        K_trans = self.kernel_(X, self.X_train_)
        V = solve_triangular(self.K_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False)

        # Alg 2.1, page 19, line 6 -> K(X_test, X_test) - v^T. v
        y_epi = self.kernel_(X) - V.T @ V

        # Undo normalisation
        y_epi = np.outer(y_epi, self._y_train_std**2).reshape(*y_epi.shape, -1)

        # if y_epi has shape (n_samples, n_samples, 1), reshape to
        # (n_samples, n_samples)
        if y_epi.shape[2] == 1:
            y_epi = np.squeeze(y_epi, axis=2)

        return y_epi
