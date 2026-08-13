"""Right-hand-side spatial operators for the heat equation."""

from rhs.laplacian import laplacian_1d
from rhs.rhs import rhs

__all__ = [
    "laplacian_1d",
    "rhs",
]
