from __future__ import annotations

import sys
from typing import Sequence, Tuple, Union

import numpy as np
from sklearn.gaussian_process.kernels import RBF

from capof.gp_scikit import GP

#: A numeric vector: numpy array or plain sequence of floats.
Vector = Union[np.ndarray, Sequence[float]]


class ContextAdaptivePolicyFramework:
    """Context-adaptive policy framework (CAPOF).

    Combines three expert policies into a mixture of experts:

    - **LfD policy**: the Gaussian-process regression of the demonstrated
      velocities.
    - **Stabilizing policy**: drives the state along the gradient of the
      epistemic uncertainty towards well-demonstrated regions.
    - **Goal attractor policy**: drives the state towards the nearest goal
      pose, weighted by the kernel activation at the goal poses.

    See :meth:`compute_moe` for the combination methods.
    """

    def __init__(
        self,
        model: GP,
        K_sp: Vector,
        K_gap: Vector,
        length_scale: Vector,
        c_dim: int,
        x_dim: int,
        phi_dim: int,
        x_dot_dim: int,
        phi_dot_dim: int,
        mu_add_dim: int,
        goal_poses: np.ndarray,
        goal_policy_method: str = "kernel",
        stabilizing_policy_vel_reduction: str = "epistemic",
    ) -> None:
        """A class with basic functionalities of context-adaptive policy framework.

        Args:
            model: The fitted :class:`~capof.gp_scikit.GP` (LfD policy) instance.
            K_sp: array-like of shape (2,). Hyperparameter Ksp = {Ksp,x, Ksp,ϕ} that
                maps the normalized stabilizing velocities to the domain.
            K_gap: array-like of shape (2,). Hyperparameter Kgap = {Kgap,x, Kgap,ϕ}
                that maps the resulting goal attractor velocities to the domain.
            length_scale: array-like of shape (n_features,). The length scale of the kernel.
            c_dim: The task-parameter dimension.
            x_dim: The x (position) dimension.
            phi_dim: The ϕ (orientation) dimension.
            x_dot_dim: The dx/dt dimension.
            phi_dot_dim: The dϕ/dt dimension.
            mu_add_dim: The additional target dimension.
            goal_poses: array-like of shape (n_poses, n_features). Goal poses of the
                demonstrated trajectories.
            goal_policy_method: Computation method for the mean of the goal attractor
                policy. One of 'kernel' or 'controller' (a ValueError is raised
                otherwise). Default "kernel".
            stabilizing_policy_vel_reduction: Computation method for the velocity
                reduction factor within the stabilizing policy. One of 'epistemic' or
                'kernel' (a ValueError is raised otherwise). Default "epistemic".
        """
        self.model = model
        self.K_sp = K_sp
        self.K_gap = K_gap
        self.length_scale = length_scale
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.phi_dim = phi_dim
        self.x_dot_dim = x_dot_dim
        self.phi_dot_dim = phi_dot_dim
        self.mu_add_dim = mu_add_dim
        self.goal_poses = goal_poses
        self.goal_policy_method = goal_policy_method
        self.stabilizing_policy_vel_reduction = stabilizing_policy_vel_reduction

        if goal_policy_method not in ["kernel", "controller"]:
            raise ValueError(
                "The 'goal_policy_method' parameter must be assigned one of these values: 'kernel', 'controller'!"
            )

        if stabilizing_policy_vel_reduction not in ["epistemic", "kernel"]:
            raise ValueError(
                "The 'stabilizing_policy_vel_reduction' parameter must be assigned one of these values: 'epistemic', 'kernel'!"
            )

        # Extract feature and target dimensions
        self.n_features = self.c_dim + self.x_dim + self.phi_dim
        self.n_targets = self.x_dot_dim + self.phi_dot_dim

        if stabilizing_policy_vel_reduction == "epistemic" and self.phi_dim != 0:
            # Adapt X dimensions to c and x for reduction factors
            X_c_x = model.X_train_[:, : self.c_dim + self.x_dim]

            # Adapt X dimensions to c and ϕ for reduction factors
            X_c_phi = np.concatenate([model.X_train_[:, : self.c_dim], model.X_train_[:, self.c_dim + self.x_dim :]], axis=1)

            # Adapt length scales for reduction factors
            length_scale_c_x = self.length_scale[: self.c_dim + self.x_dim]
            length_scale_c_phi = np.concatenate(
                [self.length_scale[: self.c_dim], self.length_scale[self.c_dim + self.x_dim :]]
            )

            # Define kernels for reduction factors
            kernel_c_x = RBF(length_scale=length_scale_c_x)
            kernel_c_phi = RBF(length_scale=length_scale_c_phi)

            # Train c and x dependend GP
            self.model_c_x = GP(kernel=kernel_c_x, alpha=model.alpha, epi_reg=model.epi_reg, optimizer=None)
            self.model_c_x.fit(X_c_x, model.y_train_[:, 0])

            # Train c and ϕ dependend GP
            self.model_c_phi = GP(kernel=kernel_c_phi, alpha=model.alpha, epi_reg=model.epi_reg, optimizer=None)
            self.model_c_phi.fit(X_c_phi, model.y_train_[:, 0])

    def compute_moe(
        self, X: np.ndarray, policy_comb: str = "MoE"
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Compute mean of the mixture of experts according to the policy combination method.

        Args:
            X: array-like of shape (n_samples, n_features). Query points where the
                MoE is computed.
            policy_comb: Policy combination method. One of 'LfD', 'Stabilizing',
                'Goal', 'LfD + Stabilizing', 'LfD + Goal', 'MoE' (a ValueError is
                raised otherwise). Default "MoE".

        Returns:
            mu_moe: array-like of shape (n_samples, n_targets). Mean of the mixture
                of experts according to the policy combination method.
            mu_add: array-like of shape (n_samples, add_targets). Additional mean from
                the LfD policy distribution (passed through).
            (mu_sp, mu_gap, epi, k_max): tuple of
                - mu_sp: array-like of shape (n_samples, n_targets) — stabilizing policy mean.
                - mu_gap: array-like of shape (n_samples, n_targets) — goal attractor policy mean.
                - epi: array-like of shape (n_samples,) — epistemic uncertainty.
                - k_max: array-like of shape (n_samples,) — kernel activation of the goal attractor.
        """
        if policy_comb not in ["LfD", "Stabilizing", "Goal", "LfD + Stabilizing", "LfD + Goal", "MoE"]:
            raise ValueError(
                "The 'policy_comb' parameter must be assigned one of these values: "
                "'LfD', 'Stabilizing', 'Goal', 'LfD + Stabilizing', 'LfD + Goal', 'MoE'!"
            )

        # Compute individual expert policies
        mu_lfd, mu_add = self._compute_lfd_policy(X)
        mu_sp, epi = self._compute_stabilizing_policy(X)
        mu_gap, k_max = self._compute_goal_attractor_policy(X)

        # Combine individual expert policies
        return self._combine_policies(mu_lfd, mu_sp, mu_gap, mu_add, epi, k_max, policy_comb)

    def _combine_policies(
        self,
        mu_lfd: np.ndarray,
        mu_sp: np.ndarray,
        mu_gap: np.ndarray,
        mu_add: np.ndarray,
        epi: np.ndarray,
        k_max: np.ndarray,
        policy_comb: str,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Combine individual expert policies according to the policy combination method.

        Args:
            mu_lfd: array-like of shape (n_samples, n_targets). Mean of LfD policy distribution.
            mu_sp: array-like of shape (n_samples, n_targets). Mean of stabilizing policy distribution.
            mu_gap: array-like of shape (n_samples, n_targets). Mean of goal attractor policy distribution.
            mu_add: array-like of shape (n_samples, add_targets). Additional mean from LfD
                policy distribution (e.g. hand control command).
            epi: array-like of shape (n_samples, 1). Epistemic uncertainty based on respective model.
            k_max: array-like of shape (n_samples, 1). Kernel activation goal attractor.
            policy_comb: Policy combination method.

        Returns:
            mu_moe: array-like of shape (n_samples, n_targets). Mean of the mixture of
                experts according to the policy combination method.
            mu_add: array-like of shape (n_samples, add_targets). Additional mean from LfD
                policy distribution (passed through).
            (mu_sp, mu_gap, epi, k_max): tuple. The stabilizing policy mean, goal attractor
                policy mean, epistemic uncertainty, and goal-attractor kernel activation
                (all passed through).
        """
        if policy_comb == "LfD":
            return mu_lfd, mu_add, (mu_sp, mu_gap, epi.flatten(), k_max.flatten())
        elif policy_comb == "Stabilizing":
            return mu_sp, mu_add, (mu_sp, mu_gap, epi.flatten(), k_max.flatten())
        elif policy_comb == "Goal":
            return mu_gap, mu_add, (mu_sp, mu_gap, epi.flatten(), k_max.flatten())
        elif policy_comb == "LfD + Stabilizing":
            mu_moe = (1 - epi) * mu_lfd + epi * mu_sp
            return mu_moe, mu_add, (mu_sp, mu_gap, epi.flatten(), k_max.flatten())
        elif policy_comb == "LfD + Goal":
            mu_moe = (1 - k_max) * mu_lfd + k_max * mu_gap
            return mu_moe, mu_add, (mu_sp, mu_gap, epi.flatten(), k_max.flatten())
        elif policy_comb == "MoE":
            mu_moe = (1 - epi) * (1 - k_max) * mu_lfd + epi * (1 - k_max) * mu_sp + k_max * mu_gap
            return mu_moe, mu_add, (mu_sp, mu_gap, epi.flatten(), k_max.flatten())

        # Unreachable when called via compute_moe (which validates policy_comb)
        raise ValueError(f"Unknown policy combination: {policy_comb}")

    def _compute_lfd_policy(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using the Gaussian process regression model.

        Args:
            X: array-like of shape (n_samples, n_features). Query points where the
                MoE is computed.

        Returns:
            mu_lfd: array-like of shape (n_samples, n_targets). Mean of the LfD policy
                distribution (robot velocity targets).
            mu_add: array-like of shape (n_samples, add_targets). Additional mean from the
                LfD policy distribution (e.g. hand control command).
        """

        # Predict using the Gaussian process regression model
        lfd_mu = self.model.predict(X)

        # Split in robot and additional mean
        return lfd_mu[:, : self.n_targets], lfd_mu[:, self.n_targets :]

    def _compute_stabilizing_policy(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean of the stabilizing policy.

        Args:
            X: array-like of shape (n_samples, n_features). Query points where the
                MoE is computed.

        Returns:
            mu_sp: array-like of shape (n_samples, n_targets). Mean of stabilizing
                policy distribution.
            epi: array-like of shape (n_samples, 1). Epistemic uncertainty based on
                respective model.
        """

        # Compute the epistemic uncertainty and its derivative
        epi = np.diag(self.model.epistemic_uncertainty(X)[:, :, 0]).copy()
        epi_derivative = np.diagonal(self.model.epistemic_gradient(X), axis1=0, axis2=2).T.copy()

        if self.phi_dim != 0:
            # Adapt X for reduction factors
            X_c_x = X[:, : self.c_dim + self.x_dim]
            X_c_phi = np.concatenate([X[:, : self.c_dim], X[:, self.c_dim + self.x_dim :]], axis=1)

            if self.stabilizing_policy_vel_reduction == "epistemic":
                # Compute the epistemic uncertainty for X_c_x and X_c_phi
                vel_reduction_x = np.diag(self.model_c_x.epistemic_uncertainty(X_c_x))
                vel_reduction_phi = np.diag(self.model_c_phi.epistemic_uncertainty(X_c_phi))

            else:
                # Find highest activated training data point
                k = RBF(length_scale=self.length_scale)
                k_X_train = k(X, self.model.X_train_)
                attr_point = self.model.X_train_[np.argmax(k_X_train, axis=1)]

                # Adapt attractor points for reduction factors
                attr_point_c_x = attr_point[:, : self.c_dim + self.x_dim]
                attr_point_c_phi = np.concatenate(
                    [attr_point[:, : self.c_dim], attr_point[:, self.c_dim + self.x_dim :]], axis=1
                )

                # Adapt length scales for reduction factors
                length_scale_c_x = self.length_scale[: self.c_dim + self.x_dim]
                length_scale_c_phi = np.concatenate(
                    [self.length_scale[: self.c_dim], self.length_scale[self.c_dim + self.x_dim :]]
                )

                # Create kernels for reduction factors
                k_c_x = RBF(length_scale=length_scale_c_x)
                k_c_phi = RBF(length_scale=length_scale_c_phi)

                # Compute kernel-based velocity reductions
                vel_reduction_x = 1 - np.diag(k_c_x(X_c_x, attr_point_c_x))
                vel_reduction_phi = 1 - np.diag(k_c_phi(X_c_phi, attr_point_c_phi))
        else:
            vel_reduction_x = np.ones(X.shape[0])

        # Derive mu_sp_x by normalizing and applying gain and reduction factor
        mu_sp_x = epi_derivative[:, self.c_dim : self.c_dim + self.x_dim]
        mu_sp_x /= np.maximum(np.linalg.norm(mu_sp_x, axis=1), sys.float_info.min).reshape(-1, 1)
        mu_sp_x *= -self.K_sp[0] * vel_reduction_x.reshape(-1, 1)

        if self.phi_dim != 0:
            # Derive mu_sp_phi by normalizing and applying gain and reduction factor
            mu_sp_phi = epi_derivative[:, self.c_dim + self.x_dim :]
            mu_sp_phi /= np.maximum(np.linalg.norm(mu_sp_phi, axis=1), sys.float_info.min).reshape(-1, 1)
            mu_sp_phi *= -self.K_sp[1] * vel_reduction_phi.reshape(-1, 1)
        else:
            mu_sp_phi = np.array([[]] * X.shape[0])

        return np.concatenate([mu_sp_x[:, : self.x_dot_dim], mu_sp_phi[:, : self.phi_dot_dim]], axis=1), epi.reshape(-1, 1)

    def _compute_goal_attractor_policy(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean of the goal attractor policy.

        Args:
            X: array-like of shape (n_samples, n_features). Query points where the
                MoE is computed.

        Returns:
            mu_gap: array-like of shape (n_samples, n_targets). Mean of goal attractor
                policy distribution.
            k_max: array-like of shape (n_samples, 1). Kernel activation goal attractor
                (passed through).
        """
        # Compute the kernel activation for each end point with respect to the current state (c included)
        k = RBF(length_scale=self.length_scale)
        k_goal_poses = k(X, self.goal_poses)

        # Goal attractor corresponds to the maximum kernel activation
        goal_attractors = self.goal_poses[np.argmax(k_goal_poses, axis=1)]

        if self.goal_policy_method == "kernel":
            # Compute kernel values between each X[i] and goal_attractors[i]
            k_diag = np.diag(k(X, goal_attractors))

            # Compute derivative of the maximum kernel
            length_scale = np.asarray(self.length_scale, dtype=float)
            K_grad = -(k_diag[:, None] * ((X - goal_attractors) @ np.diag(length_scale ** (-2))))

            # Compute position reduction factor for each X value
            k_x = RBF(length_scale=self.length_scale[self.c_dim : self.c_dim + self.x_dim])
            k_x_goal_poses = np.diag(
                k_x(X[:, self.c_dim : self.c_dim + self.x_dim], goal_attractors[:, self.c_dim : self.c_dim + self.x_dim])
            )

            # Derive mu_gap_x by normalizing and applying gain and reduction factor
            mu_gap_x = K_grad[:, self.c_dim : self.c_dim + self.x_dim]
            mu_gap_x /= np.maximum(np.linalg.norm(mu_gap_x, axis=1), sys.float_info.min).reshape(-1, 1)
            mu_gap_x *= self.K_gap[0] * (1 - k_x_goal_poses).reshape(-1, 1)

            if self.phi_dim != 0:
                # Compute orientation reduction factor for each X value
                k_phi = RBF(length_scale=self.length_scale[self.c_dim + self.x_dim :])
                k_phi_goal_poses = np.diag(
                    k_phi(X[:, self.c_dim + self.x_dim :], goal_attractors[:, self.c_dim + self.x_dim :])
                )

                # Derive mu_gap_phi by normalizing and applying gain and reduction factor
                mu_gap_phi = K_grad[:, self.c_dim + self.x_dim :]
                mu_gap_phi /= np.maximum(np.linalg.norm(mu_gap_phi, axis=1), sys.float_info.min).reshape(-1, 1)
                mu_gap_phi *= self.K_gap[1] * (1 - k_phi_goal_poses).reshape(-1, 1)
            else:
                mu_gap_phi = np.array([[]] * X.shape[0])

        else:
            # Linear state feedback controller
            mu_gap = goal_attractors[:, self.c_dim :] - X[:, self.c_dim :]

            # Normalize velocity kernel activation depending (here mu_gap=1 at rbf=0.1)
            mu_gap[:, : self.x_dim] /= np.sqrt(-2 * np.log(0.1)) * self.length_scale[self.c_dim : self.c_dim + self.x_dim]

            # Limit value to 1 in order to regulate magnitude
            if np.linalg.norm(mu_gap[:, : self.x_dim]) > 1:
                mu_gap[:, : self.x_dim] /= np.linalg.norm(mu_gap[:, : self.x_dim])

            # Apply gain to translation
            mu_gap_x = self.K_gap[0] * mu_gap[:, : self.x_dim]

            if self.phi_dim != 0:
                # Normalize velocity kernel activation depending (here mu_gap=1 at rbf=0.1)
                mu_gap[:, self.x_dim :] /= np.sqrt(-2 * np.log(0.1)) * self.length_scale[self.c_dim + self.x_dim :]

                # Limit value to 1 in order to regulate magnitude
                if np.linalg.norm(mu_gap[:, self.x_dim :]) > 1:
                    mu_gap[:, self.x_dim :] /= np.linalg.norm(mu_gap[:, self.x_dim :])

                # Apply gain to orientation
                mu_gap_phi = self.K_gap[1] * mu_gap[:, self.x_dim :]
            else:
                mu_gap_phi = np.array([[]] * X.shape[0])

        return np.concatenate([mu_gap_x[:, : self.x_dot_dim], mu_gap_phi[:, : self.phi_dot_dim]], axis=1), np.max(
            k_goal_poses, axis=1
        ).reshape(-1, 1)
