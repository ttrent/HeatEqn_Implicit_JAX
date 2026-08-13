"""Jacobian-free Newton–Krylov solver for implicit time-step equations."""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import gmres

from io_utils import SimParams, SolverParams
from solvers._types import ResidualFn, SolverDiagnostics
from solvers.convergence import convergence_threshold, rms_norm
from state import State

_EW_GAMMA = 0.9  # Eisenstat–Walker safeguard factor on the forcing-term update


class _NewtonIterate(NamedTuple):
    """State carried through the ``lax.while_loop`` of one Newton solve.

    Attributes
    ----------
    u : jax.Array
        Current field iterate, shape ``(nx,)``, dtype float64.
    residual : jax.Array
        Residual ``R(u)`` at ``u``, same shape and dtype.
    iteration : jax.Array
        Outer Newton-iteration counter, scalar int.
    linear_rtol : jax.Array
        Eisenstat–Walker forcing term for the next inner GMRES solve, scalar
        float64.
    """

    u: jax.Array
    residual: jax.Array
    iteration: jax.Array
    linear_rtol: jax.Array


def _eisenstat_walker_rtol(
    previous_norm: jax.Array,
    next_norm: jax.Array,
    solver_params: SolverParams,
) -> jax.Array:
    """Eisenstat–Walker forcing term for the next inner GMRES solve.

    Scales the inner relative tolerance by the squared residual reduction
    ``gamma * (next_norm / previous_norm) ** 2`` and clamps it to
    ``[linear_rtol_min, linear_rtol_max]``: a large residual drop tightens the
    next solve, a small drop loosens it.

    Parameters
    ----------
    previous_norm : jax.Array
        RMS residual norm before the step, scalar float64.
    next_norm : jax.Array
        RMS residual norm after the step, scalar float64.
    solver_params : SolverParams
        Newton–Krylov controls; reads ``linear_rtol_min`` and
        ``linear_rtol_max``.

    Returns
    -------
    jax.Array
        Clamped forcing term, scalar float64.
    """
    safe_previous = jnp.where(previous_norm > 0.0, previous_norm, 1.0)
    progress_ratio = jnp.where(
        previous_norm > 0.0,
        next_norm / safe_previous,
        0.0,
    )
    return jnp.clip(
        _EW_GAMMA * progress_ratio**2,
        solver_params.linear_rtol_min,
        solver_params.linear_rtol_max,
    )


def _gmres_solve(
    operator: Callable[[jax.Array], jax.Array],
    rhs_vector: jax.Array,
    solver_params: SolverParams,
    rtol: jax.Array | float | None = None,
) -> jax.Array:
    """Matrix-free GMRES solve of ``operator(x) = rhs_vector`` for ``x``.

    Parameters
    ----------
    operator : Callable[[jax.Array], jax.Array]
        Linear map applying the never-assembled Jacobian to a vector.
    rhs_vector : jax.Array
        Right-hand side, shape ``(nx,)``, dtype float64.
    solver_params : SolverParams
        Newton–Krylov controls; reads ``linear_rtol_min``, ``linear_atol``,
        ``gmres_restart`` and ``gmres_maxiter``.
    rtol : jax.Array | float | None, optional
        Relative convergence tolerance for this solve. When ``None`` (the
        default), falls back to ``solver_params.linear_rtol_min``, the tightest
        configured forcing term used by the tangent solve.

    Returns
    -------
    jax.Array
        Approximate solution ``x``, shape ``(nx,)``, dtype float64.
    """
    if rtol is None:
        rtol = solver_params.linear_rtol_min
    solution, _ = gmres(
        operator,
        rhs_vector,
        tol=rtol,  # type: ignore[arg-type]
        atol=solver_params.linear_atol,
        restart=solver_params.gmres_restart,
        maxiter=solver_params.gmres_maxiter,
    )
    return solution


def _newton_iteration(
    iterate: _NewtonIterate,
    residual_fn: Callable[[jax.Array], jax.Array],
    solver_params: SolverParams,
) -> _NewtonIterate:
    """Take one inexact Newton step with a matrix-free GMRES correction.

    Linearizes ``residual_fn`` at the current field, solves
    ``J @ correction = -R`` with GMRES at the current forcing term, applies the
    correction, and refreshes the Eisenstat–Walker forcing term from the
    achieved residual reduction.

    Parameters
    ----------
    iterate : _NewtonIterate
        Current Newton iterate.
    residual_fn : Callable[[jax.Array], jax.Array]
        Residual as a function of the field, ``u -> R(u)`` of shape ``(nx,)``.
    solver_params : SolverParams
        Newton–Krylov controls.

    Returns
    -------
    _NewtonIterate
        Next iterate, with incremented counter and updated forcing term.
    """
    _, apply_jacobian = jax.linearize(residual_fn, iterate.u)
    correction = _gmres_solve(
        apply_jacobian, -iterate.residual, solver_params, rtol=iterate.linear_rtol
    )
    u_next = iterate.u + correction
    residual_next = residual_fn(u_next)
    linear_rtol_next = _eisenstat_walker_rtol(
        rms_norm(iterate.residual), rms_norm(residual_next), solver_params
    )
    return _NewtonIterate(
        u=u_next,
        residual=residual_next,
        iteration=iterate.iteration + 1,
        linear_rtol=linear_rtol_next,
    )


def newton_krylov_solve(
    residual_fn: Callable[[jax.Array], jax.Array],
    u_init: jax.Array,
    solver_params: SolverParams,
) -> tuple[jax.Array, SolverDiagnostics]:
    """Inexact Jacobian-free Newton–Krylov root find for the field.

    Iterates outer Newton steps in a ``lax.while_loop`` until the RMS residual
    norm drops below the mixed absolute/relative threshold or the iteration cap
    is reached. Each step linearizes ``residual_fn`` at the current field with
    forward-mode ``jax.linearize`` to apply the Jacobian matrix-free, solves the
    Newton system ``J @ correction = -R`` with GMRES, and applies the
    correction. The inner GMRES tolerance follows an Eisenstat–Walker forcing
    term, so early steps are solved loosely and later steps tightly.

    This raw iteration is not differentiable: the ``lax.while_loop`` has a
    data-dependent trip count. Use :func:`differentiable_solve` for an
    autodiff-friendly entry point built on the implicit function theorem.

    Parameters
    ----------
    residual_fn : Callable[[jax.Array], jax.Array]
        Residual as a function of the field, ``u -> R(u)`` of shape ``(nx,)``.
    u_init : jax.Array
        Initial guess, shape ``(nx,)``, dtype float64.
    solver_params : SolverParams
        Newton–Krylov controls.

    Returns
    -------
    u_new : jax.Array
        Converged field, shape ``(nx,)``, dtype float64.
    diagnostics : SolverDiagnostics
        Iteration count, convergence flag, and final RMS residual norm.
    """
    initial_residual = residual_fn(u_init)
    threshold = convergence_threshold(initial_residual, solver_params)

    def not_converged(iterate: _NewtonIterate) -> jax.Array:
        return (rms_norm(iterate.residual) > threshold) & (
            iterate.iteration < solver_params.max_iterations
        )

    def take_step(iterate: _NewtonIterate) -> _NewtonIterate:
        return _newton_iteration(iterate, residual_fn, solver_params)

    initial_iterate = _NewtonIterate(
        u=u_init,
        residual=initial_residual,
        iteration=jnp.asarray(0),
        linear_rtol=jnp.asarray(solver_params.linear_rtol_init),
    )
    final_iterate = lax.while_loop(not_converged, take_step, initial_iterate)
    final_norm = rms_norm(final_iterate.residual)
    diagnostics = SolverDiagnostics(
        iterations=final_iterate.iteration,
        converged=final_norm <= threshold,
        residual_norm=final_norm,
    )
    return final_iterate.u, diagnostics


def differentiable_solve(
    residual_fn: ResidualFn,
    state: State,
    params: SimParams,
) -> tuple[jax.Array, SolverDiagnostics]:
    """Autodiff-friendly wrapper around :func:`newton_krylov_solve`.

    Adapts the implicit-update ``residual_fn`` into a field-only root problem,
    runs the (non-differentiable) Newton–Krylov iteration once to obtain the
    converged field and its diagnostics, then reattaches the derivative of that
    root through the implicit function theorem with ``jax.lax.custom_root``. A
    single matrix-free GMRES call serves as the tangent solve, so gradients flow
    via the implicit function theorem rather than by tracing the
    ``lax.while_loop``. Diagnostics are returned as ``stop_gradient`` telemetry,
    keeping the float residual norm off the differentiated path.

    Parameters
    ----------
    residual_fn : ResidualFn
        Implicit-update residual ``(state, u_new, params) -> R`` of shape
        ``(nx,)``; its root defines the next field.
    state : State
        Current state at time ``t``; ``state.u`` (shape ``(nx,)``, dtype
        float64) seeds the solve.
    params : SimParams
        Simulation parameters; solver controls are read from ``params.solver``.

    Returns
    -------
    u_new : jax.Array
        Converged field at the next step, shape ``(nx,)``, dtype float64.
    diagnostics : SolverDiagnostics
        Iteration count, convergence flag, and final RMS residual norm, detached
        from the gradient path.
    """
    solver_params = params.solver

    def residual_of(u_new: jax.Array) -> jax.Array:
        return residual_fn(state, u_new, params)

    u_star, diagnostics = newton_krylov_solve(residual_of, state.u, solver_params)
    tangent_solve = partial(_gmres_solve, solver_params=solver_params)

    # custom_root's solve arg: u_star is already a root, so return it unchanged and
    # let tangent_solve supply the gradient (implicit function theorem).
    u_new = jax.lax.custom_root(
        f=residual_of,
        initial_guess=jax.lax.stop_gradient(u_star),
        solve=lambda _residual_fn, converged_root: converged_root,
        tangent_solve=tangent_solve,
    )
    return u_new, jax.lax.stop_gradient(diagnostics)
