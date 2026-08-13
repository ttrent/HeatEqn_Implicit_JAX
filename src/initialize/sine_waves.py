"""Initial condition built as a superposition of sine modes on the grid."""

import jax
import jax.numpy as jnp

from io_utils import SimParams, SineMode
from state import State


def _sine_mode(x: jax.Array, mode: SineMode, x0: float, xn: float) -> jax.Array:
    """Single sine mode ``A * sin(2*pi*k*(x - x0)/xn)`` sampled on the grid.

    Parameters
    ----------
    x : jax.Array
        Grid coordinates, shape ``(nx,)``, dtype float64.
    mode : SineMode
        Mode with wavenumber ``k`` and amplitude ``A``.
    x0 : float
        Left boundary, used as the phase origin.
    xn : float
        Right boundary, used as the spatial period.

    Returns
    -------
    jax.Array
        Mode contribution, same shape and dtype as ``x``.
    """
    return mode.amplitude * jnp.sin(2.0 * jnp.pi * mode.wavenumber * (x - x0) / xn)


def sine_waves(params: SimParams) -> State:
    """Build the initial state as a sum of sine modes on the periodic grid.

    Builds ``u(x) = offset + sum_m A_m * sin(2*pi*k_m*(x - x0)/xn)`` over the
    modes in ``params.initial_state``, evaluated on the periodic grid
    ``x = x0 + dx * arange(nx)``.

    Parameters
    ----------
    params : SimParams
        Simulation parameters; supplies the grid (``size``, ``x0``, ``xn``,
        ``dx``), the initial-state ``offset`` and ``modes``, and the start
        time ``params.time.start``.

    Returns
    -------
    State
        Initial state with field ``u`` of shape ``(nx,)``, dtype float64, at
        time ``params.time.start``.
    """
    grid = params.grid
    x = grid.x0 + grid.dx * jnp.arange(grid.size, dtype=jnp.float64)

    waves = sum(
        (_sine_mode(x, mode, grid.x0, grid.xn) for mode in params.initial_state.modes),
        start=jnp.zeros_like(x),
    )
    u = params.initial_state.offset + waves

    return State(u=u, t=params.time.start)
