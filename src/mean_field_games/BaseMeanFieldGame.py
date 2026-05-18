from typing import Optional

import jax.numpy as jnp


class BaseMeanFieldGame:
    """
    Base class representing a Parametric Mean Field Game M_gamma = (d, A, T, Q_gamma, f_gamma, g_gamma).
    Methods are static to ensure pure JAX compatibility for JIT compilation and custom_vjp.
    """

    def Q(self, gamma: jnp.ndarray, u: jnp.ndarray, mu: Optional[jnp.ndarray]) -> jnp.ndarray:
        """
        Computes the transition rate matrix Q_gamma given the value function u.

        Args:
            gamma (jnp.ndarray): gamma (jnp.ndarray): the parameters of the MFG (shape: (d,)).
            u (jnp.ndarray): the value function of the MFG (shape: (d,)).
            mu (jnp.ndarray): mu (jnp.ndarray): the mean field distribution at time t (shape: (d,)).

        Returns:
            Q (jnp.ndarray): transition rate matrix with zero row sums (shape: (d, d)).
        """
        raise NotImplementedError

    def f(self, gamma: jnp.ndarray, u: jnp.ndarray, mu: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized running cost for all states.

        Args:
            gamma (jnp.ndarray): the parameters of the MFG (shape: (d,)).
            u (jnp.ndarray): the value function of the MFG (shape: (d,)).
            mu (jnp.ndarray): the mean field distribution at time t (shape: (d,)).
            alpha (jnp.ndarray): the control at time t (shape (d,d))

        Returns:
            jnp.ndarray: Running costs of shape (d,)
        """
        raise NotImplementedError

    @staticmethod
    def g(gamma: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        """
        Terminal cost vector for all states.

        Args:
            gamma (jnp.ndarray): the parameters of the MFG (shape: (d,)).
            mu (jnp.ndarray): the mean field distribution at time t (shape: (d,)).

        Returns:
            jnp.ndarray: Terminal costs of shape (d,)
        """
        raise NotImplementedError

    def Hamiltonian(
        self, x: int, mu: jnp.ndarray, delta_u_x: jnp.ndarray, alpha: jnp.ndarray, gamma: jnp.ndarray
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
        raise NotImplementedError

    def optimal_control(self, mu: jnp.ndarray, u: jnp.ndarray, gamma: jnp.ndarray):
        """Compute alpha*(x) for all states x."""
        raise NotImplementedError
