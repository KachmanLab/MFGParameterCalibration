from typing import Optional

import jax
import jax.numpy as jnp

from src.mean_field_games.BaseMeanFieldGame import BaseMeanFieldGame


class SIR(BaseMeanFieldGame):
    def __init__(self, c_alpha: float, c_infected: float, learn_cost: bool = False):
        self.c_alpha = float(c_alpha)
        self.c_infected = float(c_infected)
        self.learn_cost = learn_cost

    def Q(self, gamma: jnp.ndarray, u: jnp.ndarray, mu: Optional[jnp.ndarray]) -> jnp.ndarray:
        alpha_star = self.optimal_control(mu, u, gamma)  # (d,)
        return jax.vmap(lambda alpha_row, state_i: get_lambda_t_continuousAlpha(gamma, mu, alpha_row)[state_i])(
            alpha_star, jnp.arange(len(mu))
        )  # (d, d)

    def f(self, gamma: jnp.ndarray, u: jnp.ndarray, mu: jnp.ndarray, alpha_t: jnp.ndarray) -> jnp.ndarray:
        """
        Running cost vector for all states. The representative agent faces a cost for being sick and for having a low
        contact factor. The agents feel very lonely if they meet no-one.
        """
        d = mu.shape[0]
        return jax.vmap(lambda x, a: self._f(x, a, gamma))(
            jnp.arange(0, d, dtype=int),
            alpha_t,
        )

    def _f(self, x: int, alpha_t_x: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
        return jnp.select(
            condlist=[x == 0, x == 1, x == 2],
            choicelist=[
                ((self.c_alpha / 2) * (1.0 - alpha_t_x) ** 2) * (1 - self.learn_cost) + self.learn_cost * gamma[-2],
                self.c_infected * (1 - self.learn_cost) + self.learn_cost * gamma[-1],
                0,
            ],
            default=0,
        )

    @staticmethod
    def g(gamma: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        """
        Terminal cost vector for all states.
        """
        return jnp.zeros_like(mu)

    def Hamiltonian(self, x, mu, delta_u_x, alpha, gamma):
        """
        Compute the Hamiltonian for a given state x, mean field mu, costate p, and control alpha.

        Args:
            x (int): the agent's current state index.
            mu (jnp.ndarray): the mean field distribution at time t (shape: (d,)).
            delta_u_x (jnp.ndarray): The difference in value function between states x from the perspective of y (shape: (d,)).
            alpha (jnp.ndarray): the control at time t (scalar).
            gamma (jnp.ndarray): the parameters of the cybersecurity game (shape: (10,)).

        Returns:
            jnp.ndarray: the scalar Hamiltonian value.
        """
        f = self._f(x, alpha, gamma)
        Q_row = get_lambda_t_continuousAlpha(gamma, mu, alpha)[x]  # (d,)
        return f + jnp.dot(Q_row, delta_u_x)  # both (d,)

    def optimal_control(self, mu: jnp.ndarray, u: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
        """Compute alpha*(x) for all states x."""
        d = mu.shape[0]
        Delta_u = u[None, :] - u[:, None]  # (d, d), Delta_u[x, y] = u[y] - u[x]
        alphas = jnp.linspace(0.0, 1.0, 10)

        def optimal_control_state(x, delta_u_x):
            values = jax.vmap(lambda a: self.Hamiltonian(x, mu, delta_u_x, a, gamma))(alphas)
            return alphas[jnp.argmin(values)]

        return jax.vmap(optimal_control_state)(jnp.arange(d), Delta_u)  # (d,)


def get_lambda_t_continuousAlpha(gamma: jnp.ndarray, mu_t: jnp.ndarray, alpha_t: jnp.ndarray) -> jnp.ndarray:
    r"""
    Build the matrix describing the rate of transition of an agent in the SIR model in the cybersecurity network as
    provided in the paper.


    Args:
        gamma (jnp.ndarray): the parameters of the SIR model (shape: (n, )).
        mu_t (jnp.ndarray): The mean field distribution at time t (shape: (d, )).
        alpha_t (jnp.ndarray): A row of the player's control, corresponding to a single state (shape: (d, )).

    Returns:
        jnp.ndarray: the transition matrix \hat\lambda of shape (d,d)

    """
    beta, gamma = gamma[0], gamma[1]

    mu_t_I = mu_t[1]

    lambda_matrix = jnp.array(
        [
            [-beta * alpha_t * mu_t_I, beta * alpha_t * mu_t_I, 0.0],
            [0.0, -gamma, gamma],
            [0.0, 0.0, 0.0],
            # [sigma, 0.0, -sigma],
        ]
    )
    return lambda_matrix
