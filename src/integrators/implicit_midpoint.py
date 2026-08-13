"""Implicit midpoint (second-order, A-stable) time integrator."""

import jax

from io_utils import SimParams
from rhs import rhs
from state import State


def implicit_midpoint_residual(
    state: State,
    u_new: jax.Array,
    params: SimParams,
) -> jax.Array:
    """Residual of the implicit midpoint update for a proposed new field.

    The rule solves ``u_new = u + dt * rhs((u + u_new) / 2, t + dt / 2)``
    implicitly. This returns ``R(u_new) = u_new - u - dt * rhs(u_mid, ...)``
    with ``u_mid = (u + u_new) / 2``, whose root a solver drives to zero.

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

    Notes
    -----
    Second-order accurate and A-stable; ``rhs`` is evaluated at the midpoint
    in both field and time ``t + dt / 2``.
    """
    dt = params.time.step_size
    u_mid = 0.5 * (state.u + u_new)
    mid_state = State(u=u_mid, t=state.t + 0.5 * dt)
    return u_new - state.u - dt * rhs(mid_state, params)
