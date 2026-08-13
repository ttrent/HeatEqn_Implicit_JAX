"""Nested-scan driver that advances state and streams solver diagnostics."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from io_utils import SimParams
from solvers import ResidualFn, Solver, SolverDiagnostics
from state import State

# One save interval yields the snapshot state, its final-step diagnostics, and
# whether every implicit step within the interval converged.
IntervalDiagnostics = tuple[State, SolverDiagnostics, jax.Array]


def make_buffer_runner(
    residual_fn: ResidualFn, solver: Solver, params: SimParams
) -> jax.stages.Wrapped:
    """Build a jitted runner that advances save-intervals through nested scans.

    The returned ``run_buffer(state, length)`` advances ``length`` saved
    snapshots. An inner ``lax.scan`` (``step``) takes ``save_rate.steps``
    implicit steps per snapshot, and an outer ``lax.scan`` (``save_interval``)
    stacks the snapshots so a whole buffer never leaves the device until it is
    streamed to disk. ``length`` is a static argument, so each distinct value
    compiles a separate XLA executable.

    Parameters
    ----------
    residual_fn : ResidualFn
        Implicit-scheme residual whose root advances the field one time step.
    solver : Solver
        Nonlinear solve that drives ``residual_fn`` to zero each step.
    params : SimParams
        Simulation configuration; supplies ``time.step_size`` and
        ``time.save_rate.steps``.

    Returns
    -------
    jax.stages.Wrapped
        Jitted ``run_buffer(state, length)`` returning the advanced state and
        the stacked ``(states, last_diagnostics, interval_converged)`` for each
        saved snapshot.
    """
    dt = params.time.step_size
    save_steps = params.time.save_rate.steps

    def step(state: State, _: None) -> tuple[State, SolverDiagnostics]:
        u_new, diagnostics = solver(residual_fn, state, params)
        return State(u=u_new, t=state.t + dt), diagnostics

    def save_interval(state: State, _: None) -> tuple[State, IntervalDiagnostics]:
        state, diagnostics = lax.scan(step, state, xs=None, length=save_steps)
        last = SolverDiagnostics(
            iterations=diagnostics.iterations[-1],
            converged=diagnostics.converged[-1],
            residual_norm=diagnostics.residual_norm[-1],
        )
        return state, (state, last, jnp.all(diagnostics.converged))

    @partial(jax.jit, static_argnames="length")
    def run_buffer(state: State, length: int) -> tuple[State, IntervalDiagnostics]:
        return lax.scan(save_interval, state, xs=None, length=length)

    return run_buffer
