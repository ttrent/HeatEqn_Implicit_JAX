"""Implicit-solve routines for time integration."""

from solvers._types import ResidualFn, Solver, SolverDiagnostics
from solvers.newton_method import differentiable_solve, newton_krylov_solve

SOLVERS = {
    "newton_method": differentiable_solve,
}

__all__ = [
    "SOLVERS",
    "ResidualFn",
    "Solver",
    "SolverDiagnostics",
    "differentiable_solve",
    "newton_krylov_solve",
]
