from typing import Optional
from enum import Enum

import chex
import jax
import jax.numpy as jnp


from src.mean_field_games.BaseMeanFieldGame import BaseMeanFieldGame


class CyberState(Enum):
    DI = 0
    DS = 1
    UI = 2
    US = 3

    @classmethod
    def configure(cls, costs: dict) -> None:
        """
        Configure the class costs.

        Args:
            costs (dict): a dictionary containing the keys ``k_D`` and ``k_I``,
                representing the costs for being defended and infected.

        Returns:
            None
        """
        assert "k_D" in costs.keys() and "k_I" in costs.keys(), (
            f'The provided costs should include "k_D" and "k_I", received: {costs.keys()}'
        )
        cls._costs = costs

    def costs(self) -> jnp.ndarray:
        """
        Calculate the running costs based on the state at time t.

        Returns:
            float: the running costs.
        """
        if self.__class__._costs is None:
            raise RuntimeError("CyberStates has not been configured. Call CyberState.configure() first.")
        costs = 0
        if self is CyberState.DI or self is CyberState.DS:
            costs += self.__class__._costs["k_D"]
        if self is CyberState.DI or self is CyberState.UI:
            costs += self.__class__._costs["k_I"]
        return jnp.array(costs)


class Cybersecurity(BaseMeanFieldGame):
    def __init__(self, costs: dict):
        super().__init__()
        CyberState.configure(costs)
        self.costs_vector = jnp.array([CyberState(x).costs() for x in range(4)])

    def Q(self, gamma: jnp.ndarray, u: jnp.ndarray, mu: Optional[jnp.ndarray]) -> jnp.ndarray:
        alpha_star = self.optimal_control(mu, u, gamma)  # (d,)
        return jax.vmap(lambda alpha_row, state_i: get_lambda_t_continuousAlpha(gamma, mu, alpha_row)[state_i])(
            alpha_star, jnp.arange(len(mu))
        )  # (d, d)

    def f(self, gamma: jnp.ndarray, u: jnp.ndarray, mu: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
        costs_vector = jnp.array([CyberState(x).costs() for x in range(4)])
        d = u.shape[0]
        return jax.vmap(lambda x: costs_vector[x])(
            jnp.arange(
                0,
                d,
            )
        )

    @staticmethod
    def g(gamma: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(mu)

    def Hamiltonian(
        self, x: jnp.ndarray, mu: jnp.ndarray, delta_u_x: jnp.ndarray, alpha: jnp.ndarray, gamma: jnp.ndarray
    ) -> jnp.ndarray:
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
        f = self.costs_vector[x]
        Q_row = get_lambda_t_continuousAlpha(gamma, mu, alpha)[x]  # (d,)
        return f + jnp.dot(Q_row, delta_u_x)  # both (d,)

    def optimal_control(self, mu: jnp.ndarray, u: jnp.ndarray, gamma: jnp.ndarray):
        """Compute alpha*(x) for all states x."""
        Delta_u = u[None, :] - u[:, None]  # (d, d), Delta_u[x, y] = u[y] - u[x]
        alphas = jnp.array([0.0, 1.0])

        def optimal_control_state(x: jnp.ndarray, delta_u_x: jnp.ndarray):
            values = jax.vmap(lambda a: self.Hamiltonian(x, mu, delta_u_x, a, gamma))(alphas)
            return alphas[jnp.argmin(values)]

        return jax.vmap(optimal_control_state)(jnp.arange(len(mu)), Delta_u)  # (d,)


def get_lambda_t_continuousAlpha(gamma: jnp.ndarray, mu_t: jnp.ndarray, alpha_t: jnp.ndarray) -> jnp.ndarray:
    r"""
    Build the matrix describing the rate of transition of a computer in the cybersecurity network as provided in page
    656 of Carmona and Delarue [1].

    Based on the implementation retrieved from [2].

    Args:
        gamma (jnp.ndarray): the parameters of the cybersecurity game (shape: (n, )).
        mu_t (jnp.ndarray): The mean field distribution at time t (shape: (d, )).
        alpha_t (jnp.ndarray): A row of the player's control, corresponding to a single state (shape: (d, )).

    Returns:
        jnp.ndarray: the transition matrix \hat\lambda of shape (d,d)

    References:
        [1] René Carmona and François Delarue. Probabilistic theory of mean field games with applications
        I. Probability Theory and Stochastic Modelling. Springer Cham, 2018. ISBN 978-3-319-58920-302
        6. doi:10.1007/978-3-319-58920-6.303
        [2] https://github.com/mlauriere/ShandongSummerSchool2025/blob/main/MFG_ODL23Vanguard_DDPG_MFC_cybersecurity_shared.ipynb
    """
    lambda_speed = 0.8
    v_H = 0.6
    chex.assert_shape(gamma, (8,))
    [q_rec_D, q_rec_U, q_inf_D, q_inf_U, beta_DD, beta_DU, beta_UU, beta_UD] = gamma

    (d,) = mu_t.shape
    lambda_matrix = jnp.zeros((d, d))

    lambda_matrix = lambda_matrix.at[CyberState.DI.value, CyberState.DS.value].set(q_rec_D)
    lambda_matrix = lambda_matrix.at[CyberState.DS.value, CyberState.DI.value].set(
        v_H * q_inf_D + beta_DD * mu_t[CyberState.DI.value] + beta_UD * mu_t[CyberState.UI.value]
    )
    lambda_matrix = lambda_matrix.at[CyberState.UI.value, CyberState.US.value].set(q_rec_U)
    lambda_matrix = lambda_matrix.at[CyberState.US.value, CyberState.UI.value].set(
        v_H * q_inf_U + beta_UU * mu_t[CyberState.UI.value] + beta_DU * mu_t[CyberState.DI.value]
    )

    lambda_matrix = lambda_matrix.at[CyberState.DI.value, CyberState.UI.value].set(alpha_t * lambda_speed)
    lambda_matrix = lambda_matrix.at[CyberState.DS.value, CyberState.US.value].set(alpha_t * lambda_speed)
    lambda_matrix = lambda_matrix.at[CyberState.UI.value, CyberState.DI.value].set(alpha_t * lambda_speed)
    lambda_matrix = lambda_matrix.at[CyberState.US.value, CyberState.DS.value].set(alpha_t * lambda_speed)

    # Set each row's diagonal to the negative row sum
    row_sums = jnp.sum(lambda_matrix, axis=1)
    lambda_matrix = lambda_matrix.at[jnp.arange(d), jnp.arange(d)].set(-row_sums)
    return lambda_matrix
