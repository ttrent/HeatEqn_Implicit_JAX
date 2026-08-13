"""Grid-independent residual norms and stopping criteria for iterative solvers."""

import jax
import jax.numpy as jnp

from io_utils import SolverParams


def rms_norm(residual: jax.Array) -> jax.Array:
    """Root-mean-square norm, so tolerances are independent of grid size.

    Parameters
    ----------
    residual : jax.Array
        Residual vector, shape ``(nx,)``, dtype float64.

    Returns
    -------
    jax.Array
        Scalar ``sqrt(mean(residual**2))``, dtype float64.
    """
    return jnp.sqrt(jnp.mean(residual**2))


def convergence_threshold(
    initial_residual: jax.Array,
    solver_params: SolverParams,
) -> jax.Array:
    """Mixed absolute/relative stopping threshold on the RMS residual norm.

    Parameters
    ----------
    initial_residual : jax.Array
        Residual at the initial guess, shape ``(nx,)``, dtype float64.
    solver_params : SolverParams
        Solver controls; reads ``absolute_tolerance`` and
        ``relative_tolerance``.

    Returns
    -------
    jax.Array
        Scalar ``atol + rtol * rms_norm(initial_residual)``, dtype float64.
    """
    return (
        solver_params.absolute_tolerance
        + solver_params.relative_tolerance * rms_norm(initial_residual)
    )
