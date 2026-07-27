from typing import Optional

import jax
import jax.numpy as jnp

from src.mean_field_games.BaseMeanFieldGame import BaseMeanFieldGame

# TODO: Run emissions game & ablate terminal costs for the linear quadratic game. 

class EmissionsGame(BaseMeanFieldGame):
    """
    Finite-state emissions-regulation Mean Field Game, adapted from
    Carmona, Delarue & Lachapelle (CDL), "Control of McKean-Vlasov
    Dynamics versus Mean Field Games" (arXiv:1210.5771), Section 5.3.

    A firm's discretized cumulative emissions live on a ladder
    {0, ..., d-1}. Uncontrolled dynamics push the state up by 1 at rate
    `base_transition_rate` (a Poisson-type emissions process). The
    abatement control alpha(x) in [0, base_transition_rate] reduces the
    up-rate at state x to (base_transition_rate - alpha(x)), at private
    running cost alpha(x)^2 / 2. State d-1 is absorbing, to appropriately
    play the game, you need to pick d well above Lambda +
    base_transition_rate * T so it rarely binds.

    Terminal cost for a firm ending at state x, given the population's
    mean terminal state mean_T = sum_x x * mu_T(x):

        'deterrent' (literal version of CDL): penalty applies iff mean_T >  Lambda
        'bandwagon' (flipped):     penalty applies iff mean_T <= Lambda

    g(x, mu_T) = lam * max(x - Lambda, 0) * penalty_active(mean_T)

    The 'deterrent' rule is a crowd-averse coupling (expecting
    the cap to be breached makes an individual firm abate MORE, which
    can only shrink the population mean relative to doing nothing) and
    seems in this finite-state setting, incapable of having more than
    one equilibrium -- it can have a unique equilibrium, or
    (for some initial conditions) none at all, matching CDL's own
    "critical case" existence-failure finding. The 'bandwagon' rule flips
    the sign of the coupling (crowd-seeking: expecting enforcement to
    already have collapsed removes any private incentive to abate, which
     helps *cause* the collapse) and admits multiple equilibria
     for a range of initial conditions.

    Learnable parameters
    ---------------------
    gamma = jnp.array([Lambda, lam])
        Lambda : per-firm emissions cap.
        lam    : penalty price per unit of excess emissions.

    Fixed hyperparameters (constructor arguments, not learned)
    ------------------------------------------------------------
    d                    : number of discrete states {0, ..., d-1}.
    base_transition_rate : uncontrolled up-transition rate.
    variant              : 'deterrent' or 'bandwagon'.
    switch_sharpness     : the true CDL terminal cost switches on a hard
        indicator 1{mean_T >< Lambda}, which has zero gradient almost
        everywhere in gamma, which makes it difficult to apply
        gradient-based fitting. We replace it with sigmoid(switch_sharpness
         * (mean_T - Lambda)) (or its mirror image for 'bandwagon'),
        which recovers the exact CDL rule as switch_sharpness -> infinity.
        Higher values track the true model more closely but give smaller/
        spikier gradients near the threshold; lower values are easier to
        fit but bias the game away from the literal CDL rule. 15-30 is a
        reasonable range for d, Lambda on the order of 5-20.
    """

    def __init__(
        self,
        d: int = 20,
        base_transition_rate: float = 1.0,
        variant: str = "bandwagon",
        switch_sharpness: float = 20.0,
    ):
        if variant not in ("deterrent", "bandwagon"):
            raise ValueError("variant must be 'deterrent' or 'bandwagon'")
        self.d = d
        self.base_transition_rate = base_transition_rate
        self.variant = variant
        self.switch_sharpness = switch_sharpness
        self._states = jnp.arange(d, dtype=jnp.float32)


    def optimal_control(self, mu: Optional[jnp.ndarray], u: jnp.ndarray, gamma: jnp.ndarray) -> jnp.ndarray:
        """
        Closed-form optimal abatement feedback, embedded as a full
        (d, d) transition-rate matrix with A[x, x+1] = base_transition_rate
        - alpha*(x) and zero elsewhere (this ladder model only has one
        controllable transition per state). alpha*(x) does not depend on
        mu or gamma directly here (the abatement running cost is private
        and gamma-free) -- both are accepted for interface compatibility
        and because subclasses/variants may want them.

        alpha*(x) = clip(u[x+1] - u[x], 0, base_transition_rate)
        """
        b = self.base_transition_rate
        delta = jnp.concatenate([u[1:] - u[:-1], jnp.zeros(1, dtype=u.dtype)])  # delta[d-1] := 0
        alpha_star = jnp.clip(delta, 0.0, b)
        up_rate = b - alpha_star
        up_rate = up_rate.at[self.d - 1].set(0.0)  # absorbing top state, no outflow
        A = jnp.zeros((self.d, self.d), dtype=u.dtype)
        A = A.at[jnp.arange(self.d - 1), jnp.arange(1, self.d)].set(up_rate[:-1])
        return A

    def Q(self, gamma: jnp.ndarray, u: jnp.ndarray, mu: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Q_gamma[x, y] = a_{xy}^*. Zero diagonal (row-sum closure, if
        needed by your integrator, is left to the ODE solver, matching
        the LinearQuadratic convention)."""
        d = u.shape[0]
        A = self.optimal_control(mu, u, gamma)
        return A * (1.0 - jnp.eye(d, dtype=u.dtype))

    def f(
        self, gamma: jnp.ndarray, u_next: jnp.ndarray, mu_curr: jnp.ndarray, alpha: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Vectorized running cost for all states, evaluated at the optimal
        feedback implied by u_next (private abatement cost only -- no
        direct dependence on mu_curr in this model).
        Returns vector of shape (d,).
        """
        Q_off_diag = self.Q(gamma, u_next, mu_curr)
        up_rate = jnp.sum(Q_off_diag, axis=1)                  # single nonzero entry per row
        alpha_star = self.base_transition_rate - up_rate
        return 0.5 * alpha_star**2

    def _mean(self, mu: jnp.ndarray) -> jnp.ndarray:
        d = mu.shape[0]
        return jnp.sum(self._states[:d] * mu)

    def _penalty_active(self, mean_T: jnp.ndarray, Lambda: jnp.ndarray) -> jnp.ndarray:
        """Smooth (sigmoid) relaxation of the CDL aggregate switch; see
        `switch_sharpness` in the class docstring."""
        z = self.switch_sharpness * (mean_T - Lambda)
        if self.variant == "deterrent":
            return jax.nn.sigmoid(z)     # -> 1 when mean_T > Lambda
        else:
            return jax.nn.sigmoid(-z)    # -> 1 when mean_T <= Lambda

    def g(self, gamma: jnp.ndarray, mu_terminal: jnp.ndarray) -> jnp.ndarray:
        """
        Terminal cost vector for all states, shape (d,).
        gamma = [Lambda, lam].
        """
        Lambda, lam = gamma[0], gamma[1]
        d = mu_terminal.shape[0]
        states = self._states[:d]
        mean_T = self._mean(mu_terminal)
        excess = jnp.clip(states - Lambda, min=0.0)
        penalty_active = self._penalty_active(mean_T, Lambda)
        return lam * excess * penalty_active

    def Hamiltonian(
        self, x: int, mu: jnp.ndarray, delta_u_x: jnp.ndarray, alpha: jnp.ndarray, gamma: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Un-minimized Hamiltonian at state x for an arbitrary (not
        necessarily optimal) scalar abatement control `alpha`. Useful for
        first-order-condition checks / best-response validation; the
        actual optimum is given in closed form by `optimal_control`
        (LinearQuadratic similarly never needs to call this -- it is
        provided here for completeness / interface parity).

        delta_u_x is taken to mean u[y] - u[x] for the CURRENT state x,
        i.e. row x of (u[None,:]-u[:,None]); the only entry that matters
        for this ladder model is y = x+1.
        """
        b = self.base_transition_rate
        d = delta_u_x.shape[0]
        up_rate = jnp.where(x < self.d - 1, b - alpha, 0.0)
        y_next = jnp.minimum(x + 1, d - 1)
        return up_rate * delta_u_x[y_next] + 0.5 * alpha**2

    def mfc_extra_term(self, gamma: jnp.ndarray, dg_dmu: jnp.ndarray, u_next: jnp.ndarray, mu_curr: jnp.ndarray):
        """Running cost f has no direct mu_curr-dependence in this model
        (abatement cost is purely private), so there is no extra running
        term. Returned for interface parity with LinearQuadratic."""
        return jnp.zeros_like(mu_curr)

    def mfc_extra_term_terminal(self, gamma: jnp.ndarray, mu_terminal: jnp.ndarray):
        r"""
        d/d mu_T(y) [ sum_x mu_T(x) g(x, mu_T) ] via autodiff, since the
        terminal coupling here is genuinely nonlinear in mu_T (through
        the sigmoid switch) rather than a simple polynomial as in
        LinearQuadratic -- easier and less error-prone to let JAX
        differentiate `g` directly than to hand-derive the chain rule.
        """
        def total_terminal_cost(m):
            return jnp.sum(m * self.g(gamma, m))
        return jax.grad(total_terminal_cost)(mu_terminal)