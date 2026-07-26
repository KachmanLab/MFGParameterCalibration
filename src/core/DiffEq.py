import diffrax
import jax.numpy as jnp


def make_neural_ode_forward(t_grid: jnp.ndarray):
    dt0 = t_grid[1] - t_grid[0]

    def forward(model, dg_dmu, init_mu, mu0):
        """
        Forward function of the forward-learning neural MFG. dg_dmu and init_mu are not used, but added
        for compatibility reasons with solver_fn.
        """

        def vector_field(t, mu, args):
            return model(t, mu)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            t0=t_grid[0],
            t1=t_grid[-1],
            dt0=dt0,
            y0=mu0,
            saveat=diffrax.SaveAt(ts=t_grid),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        )
        return sol.ys

    return forward
