"""Time-stepping integrators: implicit-scheme residuals and the scan driver."""

from integrators.backward_euler import backward_euler_residual
from integrators.implicit_midpoint import implicit_midpoint_residual
from integrators.stepping import make_buffer_runner

INTEGRATORS = {
    "backward_euler": backward_euler_residual,
    "implicit_midpoint": implicit_midpoint_residual,
}

__all__ = [
    "INTEGRATORS",
    "backward_euler_residual",
    "implicit_midpoint_residual",
    "make_buffer_runner",
]
