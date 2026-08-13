"""Spatial Laplacian operators for the 1D heat equation."""

import jax
import jax.numpy as jnp

from io_utils import SimParams
from state import State


def laplacian_1d(state: State, params: SimParams) -> jax.Array:
    """Second-order central-difference Laplacian with periodic BCs.

    Uses ``jnp.roll`` for the periodic wraparound, forming the compact
    three-point stencil ``(u[i-1] - 2 * u[i] + u[i+1]) / dx**2``.

    Parameters
    ----------
    state : State
        Simulation state; ``state.u`` has shape ``(nx,)`` and dtype float64.
        ``state.t`` is unused by the spatial operator.
    params : SimParams
        Simulation parameters; the grid spacing ``params.grid.dx`` is used.

    Returns
    -------
    jax.Array
        Laplacian of ``state.u``, same shape and dtype as ``state.u``.
    """
    u = state.u
    dx = params.grid.dx
    return (jnp.roll(u, -1) - 2.0 * u + jnp.roll(u, 1)) / dx**2
