"""Time-stepping integrators: implicit-scheme residual functions."""

from integrators.backward_euler import backward_euler_residual
from integrators.implicit_midpoint import implicit_midpoint_residual

INTEGRATORS = {
    "backward_euler": backward_euler_residual,
    "implicit_midpoint": implicit_midpoint_residual,
}

__all__ = [
    "INTEGRATORS",
    "backward_euler_residual",
    "implicit_midpoint_residual",
]
