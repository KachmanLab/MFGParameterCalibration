from typing import Optional

import jax.numpy as jnp


from src.mean_field_games.BaseMeanFieldGame import BaseMeanFieldGame


class LinearQuadratic(BaseMeanFieldGame):
    """
    Linear-Quadratic Mean Field Game.
    f_gamma(x, a, mu) = gamma * sum_{y!=x} (a_{xy} - 2)^2 + mu(x)
    g_gamma(x, mu) = 0
    Action space A = [1, 3]
    """

    def __init__(
        self,
        terminal_cost_scale=5.0,
        congestion_weight=0.1,
        congestion_power=1,
        control_weight=1.0,
        terminal_congestion_weight=1.0,
        destination_congestion_weight=0.0,
        base_transition_rate=2.0,
    ):
        self.terminal_cost_scale = terminal_cost_scale
        self.congestion_weight = congestion_weight
        self.congestion_power = congestion_power
        self.control_weight = control_weight
        self.terminal_congestion_weight = terminal_congestion_weight
        self.destination_congestion_weight = destination_congestion_weight
        self.base_transition_rate = base_transition_rate

    def Q(self, gamma: jnp.ndarray, u: jnp.ndarray, mu: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Computes optimal transition rates Q_gamma[x, y] = a_{xy}^* given value function u and parameter gamma.
        u is of shape (d,), mu_curr is of shape (d,)
        Returns a matrix Q of shape (d, d) with zero diagonal.
        """
        d = u.shape[0]
        A = self.optimal_control(mu, u, gamma)
        # Diagonal should be 0 (no self transitions here, handled in ODE)
        Q_off_diag = A * (1.0 - jnp.eye(d))
        return Q_off_diag

    def f(
        self, gamma: jnp.ndarray, u_next: jnp.ndarray, mu_curr: jnp.ndarray, alpha: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Vectorized running cost for all states.
        Returns vector of shape (d,)
        """
        d = u_next.shape[0]
        Q_off_diag = self.Q(gamma, u_next, mu_curr)

        rho = 1.0 + self.destination_congestion_weight * mu_curr

        # f(x, mu) = control_weight * gamma * sum_{y!=x} (a_{xy} - base)^2 * rho(y) + CONGESTION_WEIGHT * mu(x)^p
        cost = self.control_weight * gamma * jnp.sum(
            (Q_off_diag - self.base_transition_rate) ** 2 * (1.0 - jnp.eye(d)) * rho[None, :], axis=1
        ) + self.congestion_weight * (mu_curr**self.congestion_power)
        return cost

    def mfc_extra_term(self, gamma, dg_dmu, u_next, mu_curr):
        r"""
        Computes the extra term \sum_y \mu(y) [\partial f / \partial \mu(x) + \partial f / \partial \gamma * \partial \gamma / \partial \mu(x)]
        For the LQ model:
        \partial f / \partial \mu(x) = c * p * \mu(x)^{p-1}
        \partial f / \partial \gamma = \sum_{z \neq y} (a_{yz} - 2)^2
        """
        d = u_next.shape[0]
        Q_off_diag = self.Q(gamma, u_next, mu_curr)

        rho = 1.0 + self.destination_congestion_weight * mu_curr

        # \partial f / \partial \gamma = control_weight * \sum_{z \neq y} (a_{yz} - base)^2 * rho(z)
        cost_gamma_deriv = self.control_weight * jnp.sum(
            (Q_off_diag - self.base_transition_rate) ** 2 * (1.0 - jnp.eye(d)) * rho[None, :], axis=1
        )
        scalar_term = jnp.sum(mu_curr * cost_gamma_deriv)

        # \partial f / \partial \mu(x) = c * p * \mu(x)^{p-1}
        # So \sum_y \mu(y) \partial f / \partial \mu(x) = \mu(x) * c * p * \mu(x)^{p-1} = c * p * \mu(x)^p
        congestion_extra = self.congestion_weight * self.congestion_power * (mu_curr**self.congestion_power)

        # Incoming traffic penalty for destination congestion
        # kappa * gamma * W * \sum_x \mu(x) (a_{xz} - base)^2
        incoming_penalty = (
            self.destination_congestion_weight
            * gamma
            * self.control_weight
            * jnp.sum(mu_curr[:, None] * (Q_off_diag - self.base_transition_rate) ** 2 * (1.0 - jnp.eye(d)), axis=0)
        )

        # Total extra term
        return congestion_extra + incoming_penalty + scalar_term * dg_dmu

    def g(self, gamma, mu_terminal):
        """
        Terminal cost g_gamma(x, mu).
        We use a state-dependent cost: lower-indexed states are more expensive.
        For example, with d=3, the profile is [2.0, 1.0, 0.0].
        """
        d = mu_terminal.shape[0]
        cost_profile = jnp.arange(d - 1, -1, -1, dtype=jnp.float32)
        base_cost = self.terminal_cost_scale * cost_profile

        congestion_cost = self.terminal_congestion_weight * (mu_terminal**self.congestion_power)
        return base_cost + congestion_cost

    def mfc_extra_term_terminal(self, mu_terminal):
        r"""
        Computes the extra adjoint penalty at T: \sum_y \mu_T(y) \partial g / \partial \mu_T(x)
        """
        return self.terminal_congestion_weight * self.congestion_power * (mu_terminal**self.congestion_power)

    def optimal_control(self, mu: jnp.ndarray, u: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
        """Compute alpha*(x) for all states x. For the linear quadratic mean field game, the optimal control has
        a closed-form solution."""
        # Delta_u[x, y] = u[y] - u[x]
        Delta_u = u[None, :] - u[:, None]

        rho = 1.0 + self.destination_congestion_weight * mu

        # a_{xy}^* = base - p_y / (2 * gamma * control_weight * rho[y]) where p_y = u[y] - u[x]
        A = self.base_transition_rate - Delta_u / (2.0 * gamma * self.control_weight * rho[None, :])
        # Allow rates to go down to 0.0 (or very close to it)
        A = jnp.clip(A, 0.0, 100.0)
        return A
