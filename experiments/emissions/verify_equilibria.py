"""
Generic forward-backward Picard solver, built ONLY from a game's public
Q/f/g methods, used here purely to verify EmissionsGame is wired up
correctly and still reproduces the multiplicity finding.

This mirrors what a generic BaseMeanFieldGame solver almost certainly
does internally: backward HJB via Hamiltonian = drift(Q) + running cost
(f), forward Kolmogorov via the same Q, Picard-iterated on the terminal
(or full) mean-field flow to convergence.
"""

import jax
import jax.numpy as jnp

from core.PicardSolver import PicardSolver
from src.mean_field_games.EmissionsGame import EmissionsGame

jax.config.update("jax_enable_x64", True)


def aggregate_emission(mu: jnp.ndarray, d: int):
    """
    Calculate the aggregate emissions of all data points based on the terminal population state `mu` and the `d`
    emission bins.
    """
    return float(jnp.sum(jnp.arange(d) * mu))


if __name__ == "__main__":
    d, b, T, x0 = 20, 1.0, 6.0, 3
    Lambda, lam = 6.0, 5.0
    gamma = jnp.array([Lambda, lam])
    mu0 = jnp.zeros(d).at[x0].set(1.0)
    n_steps = 150

    def run(variant, mu_term_init, gamma_true):
        mfg = EmissionsGame(d=d, base_transition_rate=b, variant=variant, switch_sharpness=2)
        g = jnp.repeat(gamma_true[jnp.newaxis, :], n_steps, axis=0)
        mu_term_init = jnp.repeat(mu_term_init[None, :], n_steps, axis=0)

        def cond(val):
            return ((val[0] < 1000) & (jnp.max(jnp.abs(val[2] - val[1])) > 1e-5)) | (val[0] < 2)

        def body(val):
            nxt = PicardSolver.picard_operator_static(val[2], g, None, mu0, mfg, T / n_steps, False)
            return val[0] + 1, val[2], nxt

        _, _, bar_mu = jax.lax.while_loop(cond, body, (0, mu_term_init, mu_term_init))

        return bar_mu

    low_guess = jnp.zeros(d).at[x0].set(1.0)
    high_guess = jnp.zeros(d).at[min(d - 1, x0 + int(b * T))].set(1.0)

    print("=== bandwagon variant: Picard iteration from two different initial guesses ===")
    mu_t = run("bandwagon", low_guess, gamma)
    print(f"seed=low-mean guess  -> converged mean_T = {aggregate_emission(mu_t[-1], d):.4f}")
    mu_t2 = run("bandwagon", high_guess, gamma)
    print(f"seed=high-mean guess -> converged mean_T = {aggregate_emission(mu_t2[-1], d):.4f}")

    print("\n=== deterrent variant: same two seeds ===")
    mu_t3 = run("deterrent", low_guess, gamma)
    print(f"seed=low-mean guess  -> converged mean_T = {aggregate_emission(mu_t3[-1], d):.4f}")
    mu_t4 = run("deterrent", high_guess, gamma)
    print(f"seed=high-mean guess -> converged mean_T = {aggregate_emission(mu_t4[-1], d):.4f}")
