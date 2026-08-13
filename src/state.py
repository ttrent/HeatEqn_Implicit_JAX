"""Evolving simulation state carried through time integration."""

from typing import NamedTuple

import jax


class State(NamedTuple):
    """Solution field and current time, carried through ``lax.scan``.

    Attributes
    ----------
    u : jax.Array
        Solution field on the periodic grid, shape ``(nx,)``, dtype float64.
    t : float
        Current simulation time.
    """

    u: jax.Array
    t: float
