"""Aggregated right-hand side ``du/dt`` for the heat equation."""

import jax

from io_utils import SimParams
from rhs.laplacian import laplacian_1d
from state import State


def rhs(state: State, params: SimParams) -> jax.Array:
    """Total ``du/dt`` from every spatial operator and source term.

    Sums all physical contributions to the time derivative so integrators
    depend only on this aggregate, never on the individual physics. New
    terms (advection, sources, ...) are added here alone.

    Parameters
    ----------
    state : State
        Simulation state; ``state.u`` has shape ``(nx,)`` and dtype float64.
    params : SimParams
        Simulation parameters passed through to each operator.

    Returns
    -------
    jax.Array
        Time derivative ``du/dt`` of ``state.u``, same shape and dtype.

    Notes
    -----
    Currently the pure heat equation ``du/dt = laplacian(u)``.
    """
    return laplacian_1d(state, params)
