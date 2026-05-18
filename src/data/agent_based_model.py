"""
JAX implementation of the agent-based-model (ABM) by Gaskin et al. Incorporates an additional fleeing feature.

Reference:
    Thomas Gaskin, Grigorios A. Pavliotis, and Mark Girolami. Neural parameter calibration207
    for large-scale multiagent models. Proceedings of the National Academy of Sciences, 120(7):208
    e2216415120, 2023. doi:10.1073/pnas.2216415120.209

"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import NamedTuple
import numpy as np


S, I, R = 0, 1, 2  # state constants (int)


def sample_ABM_init(n: int) -> dict:
    n_infected = np.random.randint(0, n // 3)
    n_recovered = np.random.randint(0, n // 3)
    return {"N_infected": n_infected, "N_recovered": n_recovered}


class SIRState(NamedTuple):
    """Immutable snapshot of the full simulation state."""

    positions: jnp.ndarray  # (N, 2)  float32
    states: jnp.ndarray  # (N,)    int32  — 0=S, 1=I, 2=R
    time_since_infection: jnp.ndarray  # (N,)    int32


@partial(jit, static_argnames=("periodic",))
def pairwise_distances(positions: jnp.ndarray, space: jnp.ndarray, periodic: bool) -> jnp.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 2)
    if periodic:
        # Explicitly reshape space to (1, 1, 2) to make broadcasting unambiguous
        space_r = space.reshape(1, 1, 2)
        diff = diff - jnp.round(diff / space_r) * space_r
    return jnp.sqrt(jnp.sum(diff**2, axis=-1))


@partial(jit, static_argnames=("periodic", "flee"))
def move_agents(
    positions: jnp.ndarray,
    states: jnp.ndarray,
    sigma_s: float,
    sigma_i: float,
    sigma_r: float,
    space: jnp.ndarray,
    r_infectious: float,
    key: jax.random.PRNGKey,
    periodic: bool,
    flee: bool = True,
) -> jnp.ndarray:
    key_rand, _ = jax.random.split(key)

    sigmas = jnp.where(states == S, sigma_s, jnp.where(states == I, sigma_i, sigma_r))

    raw = jax.random.normal(key_rand, positions.shape)
    norm = jnp.linalg.norm(raw, axis=-1, keepdims=True) + 1e-8
    random_dir = raw / norm * sigmas[:, None]

    # Always compute flee_vec — mask it to zero if flee=False or agent isn't susceptible
    dists = pairwise_distances(positions, space, periodic)

    infected_mask = (states == I).astype(jnp.float32)
    in_radius = (dists < r_infectious) & (dists > 0)

    diff = positions[:, None, :] - positions[None, :, :]
    dist_safe = jnp.where(dists > 0, dists, 1.0)[:, :, None]
    unit_away = diff / dist_safe

    weights = (
        jnp.where(
            in_radius,
            (r_infectious - dists) / r_infectious,
            0.0,
        )
        * infected_mask[None, :]
    )

    flee_vec = jnp.sum(unit_away * weights[:, :, None], axis=1)
    flee_norm = jnp.linalg.norm(flee_vec, axis=-1, keepdims=True) + 1e-8
    flee_dir = flee_vec / flee_norm * sigmas[:, None]

    # Zero out flee for non-susceptible agents, and entirely if flee=False
    is_susceptible = (states == S).astype(jnp.float32)[:, None]
    flee_dir = flee_dir * is_susceptible * float(flee)  # float(flee) is 0.0 or 1.0 at trace time

    direction = random_dir + flee_dir

    new_positions = positions + direction
    if periodic:
        new_positions = new_positions % space
    else:
        new_positions = jnp.where(new_positions < 0, -new_positions, new_positions)
        new_positions = jnp.where(new_positions > space, 2 * space - new_positions, new_positions)

    return new_positions


@partial(jit, static_argnames=("periodic",))
def infection_step(
    sir: SIRState,
    p_infect: float,
    t_infectious: int,
    r_infectious: float,
    space: jnp.ndarray,
    key: jax.random.PRNGKey,
    periodic: bool,
) -> SIRState:
    dists = pairwise_distances(sir.positions, space, periodic)  # (N, N)

    # Number of infected neighbours within radius for each susceptible agent
    infected_mask = (sir.states == I).astype(jnp.float32)  # (N,)
    in_radius = ((dists < r_infectious) & (dists > 0)).astype(jnp.float32)
    n_contacts = in_radius @ infected_mask  # (N,)

    # Infect with probability 1 - (1 - p_infect)^n_contacts
    p_get_infected = 1.0 - jnp.pow(1.0 - p_infect, n_contacts)
    newly_infected = (sir.states == S) & (jax.random.uniform(key, sir.states.shape) < p_get_infected)

    # Recover agents that have been infectious for >= t_infectious steps
    newly_recovered = (sir.states == I) & (sir.time_since_infection >= t_infectious)

    new_states = jnp.where(newly_infected, I, jnp.where(newly_recovered, R, sir.states))

    new_times = jnp.where(
        new_states == I,
        jnp.where(newly_infected, 0, sir.time_since_infection + 1),
        0,
    )

    return SIRState(sir.positions, new_states, new_times)


@partial(jit, static_argnames=("periodic", "flee"))
def step(
    sir: SIRState,
    p_infect: float,
    t_infectious: int,
    r_infectious: float,
    sigma_s: float,
    sigma_i: float,
    sigma_r: float,
    space: jnp.ndarray,
    key: jax.random.PRNGKey,
    periodic: bool,
    flee: bool = True,
) -> SIRState:
    key_move, key_infect = jax.random.split(key)

    sir = infection_step(sir, p_infect, t_infectious, r_infectious, space, key_infect, periodic)

    new_positions = move_agents(
        sir.positions,
        sir.states,
        sigma_s,
        sigma_i,
        sigma_r,
        space,
        r_infectious,
        key_move,
        periodic,
        flee,
    )

    return SIRState(new_positions, sir.states, sir.time_since_infection)


class SIR_ABM_JAX:
    def __init__(
        self,
        *,
        N: int,
        space: tuple,
        sigma_s: float,
        sigma_i: float,
        sigma_r: float,
        r_infectious: float,
        p_infect: float,
        t_infectious: int,
        is_periodic: bool,
        N_infected: int = 1,
        N_recovered: int = 0,
        flee: bool = True,
        seed: int = 0,
        **__,
    ):
        self.params = dict(
            p_infect=p_infect,
            t_infectious=t_infectious,
            r_infectious=r_infectious,
            sigma_s=sigma_s,
            sigma_i=sigma_i,
            sigma_r=sigma_r,
        )
        self.space = jnp.array(space, dtype=jnp.float32)
        self.is_periodic = is_periodic
        self.flee = flee
        self.N = N
        self.key = jax.random.PRNGKey(seed)

        # Initial positions
        key, k1, k2 = jax.random.split(self.key, 3)
        positions = jax.random.uniform(k1, (N, 2)) * self.space

        # Initial states
        states_np = np.array(
            [I] * N_infected + [S] * (N - N_infected - N_recovered) + [R] * N_recovered,
            dtype=np.int32,
        )
        states = jnp.array(states_np)
        times = jnp.zeros(N, dtype=jnp.int32)

        self.state = SIRState(positions, states, times)
        self.key = k2

    def run_single(self, parameters=None):
        p = self.params["p_infect"] if parameters is None else float(parameters[0])
        t_inf = self.params["t_infectious"] if parameters is None else int(parameters[1])

        self.key, subkey = jax.random.split(self.key)
        self.state = step(
            self.state,
            p_infect=p,
            t_infectious=t_inf,
            r_infectious=self.params["r_infectious"],
            sigma_s=self.params["sigma_s"],
            sigma_i=self.params["sigma_i"],
            sigma_r=self.params["sigma_r"],
            space=self.space,
            key=subkey,
            periodic=self.is_periodic,
            flee=self.flee,
        )
        return self.state

    def counts(self):
        """Returns (S, I, R) counts as plain Python ints."""
        s = int(jnp.sum(self.state.states == S))
        i = int(jnp.sum(self.state.states == I))
        r = int(jnp.sum(self.state.states == R))
        return jnp.array([s, i, r])

    def reset(self, seed: int = 0):
        self.__init__(
            seed=seed,
            **self.params,
            N=self.N,
            space=tuple(self.space.tolist()),
            is_periodic=self.is_periodic,
            flee=self.flee,
        )
