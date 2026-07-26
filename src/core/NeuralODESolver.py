import jax
import diffrax
import jax.numpy as jnp

from src.core.PicardSolver import PicardSolver


class NeuralODESolver(PicardSolver):
    """
    Picard solver whose forward FPK half-step is replaced by a diffrax
    `diffeqsolve` using the closed-form optimal control for the LQ MFG.

    This implementation keeps the original fixed-point iteration and
    implicit-differentiation backward pass from the PicardSolver, but
    overrides the solve_fpk function.
    """
    def __init__(
        self,
        mfg,
        ts: jnp.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-5,
        is_mfc: bool = False,
        damping: float = 0.0,
        ode_solver: diffrax.AbstractSolver = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        adjoint: diffrax.AbstractAdjoint = None,
        max_steps: int = 4096,
    ):
        dt = ts[1] - ts[0]
        super().__init__(mfg, dt, max_iter=max_iter, tol=tol, is_mfc=is_mfc, damping=damping)
        self.ts = ts
        self.ode_solver = ode_solver if ode_solver is not None else diffrax.Tsit5()
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint if adjoint is not None else diffrax.RecursiveCheckpointAdjoint()
        self.max_steps = max_steps

    def solve_hjb(self, mu_traj, gamma_net, dg_dmu_traj):
        """
        Solve the Hamilton Jacobi Bellman equation using the current
        estimate for \gamma.
        """
        gamma_traj = jax.vmap(gamma_net)(self.ts, mu_traj)
        return super().solve_hjb(mu_traj, gamma_traj, dg_dmu_traj)

    def solve_fpk(self, u_traj, gamma_net, mu_0):
        """
        Override the solve_fpk from the original PicardSolver. Instead of using forward Euler,
        this function integrates the FPK equation continuously in time via diffrax, using u_traj
        (and gamma_net for predicting gamma_traj).
        """
        u_interp = diffrax.LinearInterpolation(self.ts, u_traj)

        def vector_field(t, mu, args):
            u_t = u_interp.evaluate(t)
            gamma_t = gamma_net(t, mu)
            Q = self.mfg.Q(gamma_t, u_t, mu)
            generator = Q - jnp.diag(jnp.sum(Q, axis=1))
            return mu @ generator

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            self.ode_solver,
            t0=self.ts[0],
            t1=self.ts[-1],
            dt0=self.dt,
            y0=mu_0,
            saveat=diffrax.SaveAt(ts=self.ts),
            stepsize_controller=diffrax.ConstantStepSize(),
            adjoint=self.adjoint,
            max_steps=self.max_steps,
        )
        return sol.ys