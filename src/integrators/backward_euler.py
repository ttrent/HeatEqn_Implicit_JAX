"""Backward Euler implicit time integrator."""

import jax

from io_utils import SimParams
from rhs import rhs
from state import State


def backward_euler_residual(
    state: State,
    u_new: jax.Array,
    params: SimParams,
) -> jax.Array:
    """Residual of the backward Euler update for a proposed new field.

    Backward Euler solves ``u_new = u + dt * rhs(u_new, t + dt)`` implicitly.
    This returns ``R(u_new) = u_new - u - dt * rhs(u_new, t + dt)``, whose
    root a solver drives to zero to obtain the updated field.

    Parameters
    ----------
    state : State
        Current state at time ``t``; ``state.u`` has shape ``(nx,)``,
        dtype float64.
    u_new : jax.Array
        Proposed field at ``t + dt``, shape ``(nx,)``, dtype float64.
    params : SimParams
        Simulation parameters; ``params.time.step_size`` is ``dt``.

    Returns
    -------
    jax.Array
        Residual, same shape and dtype as ``u_new``.
    """
    dt = params.time.step_size
    new_state = State(u=u_new, t=state.t + dt)
    return u_new - state.u - dt * rhs(new_state, params)
