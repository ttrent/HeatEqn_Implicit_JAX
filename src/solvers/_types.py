"""Shared solver contracts: residual, solver, and diagnostics types."""

from collections.abc import Callable
from typing import NamedTuple

import jax

from io_utils import SimParams
from state import State

ResidualFn = Callable[[State, jax.Array, SimParams], jax.Array]


class SolverDiagnostics(NamedTuple):
    """Solver telemetry returned alongside the converged field.

    Attributes
    ----------
    iterations : jax.Array
        Number of outer solver iterations taken, scalar int.
    converged : jax.Array
        Whether the residual met the stopping tolerance, scalar bool.
    residual_norm : jax.Array
        Final RMS residual norm, scalar float64.
    """

    iterations: jax.Array
    converged: jax.Array
    residual_norm: jax.Array


Solver = Callable[[ResidualFn, State, SimParams], tuple[jax.Array, SolverDiagnostics]]
