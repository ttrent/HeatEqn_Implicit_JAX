"""Closed-form analytic solution of the heat equation for output comparison."""

from pathlib import Path
from typing import NamedTuple, cast

import h5py
import numpy as np
import yaml
from numpy.typing import ArrayLike


class AnalyticInputs(NamedTuple):
    """Everything needed to evaluate the analytic solution for one run.

    Attributes
    ----------
    x : np.ndarray
        Grid coordinates, shape ``(nx,)``, dtype float64.
    t : np.ndarray
        Saved simulation times, shape ``(n_saves,)``, dtype float64.
    x0 : float
        Left boundary, the phase origin of every sine mode.
    xn : float
        Right boundary, the spatial period of every sine mode.
    offset : float
        Constant background of the initial condition.
    wavenumbers : np.ndarray
        Mode wavenumbers ``k_m``, shape ``(n_modes,)``, dtype float64.
    amplitudes : np.ndarray
        Mode amplitudes ``A_m``, shape ``(n_modes,)``, aligned with
        ``wavenumbers``, dtype float64.
    t_start : float
        Initial time at which the initial condition is imposed.
    """

    x: np.ndarray
    t: np.ndarray
    x0: float
    xn: float
    offset: float
    wavenumbers: np.ndarray
    amplitudes: np.ndarray
    t_start: float


class AnalyticSolution(NamedTuple):
    """Analytic field sampled on the simulation's grid and saved times.

    Attributes
    ----------
    x : np.ndarray
        Grid coordinates, shape ``(nx,)``, dtype float64.
    t : np.ndarray
        Absolute saved times, shape ``(n_saves,)``, dtype float64.
    u : np.ndarray
        Analytic field, shape ``(n_saves, nx)``, dtype float64; row ``i`` is
        the solution at time ``t[i]``.
    """

    x: np.ndarray
    t: np.ndarray
    u: np.ndarray


def analytic_solution(
    t: ArrayLike,
    x: ArrayLike,
    x0: float,
    xn: float,
    offset: float,
    wavenumbers: ArrayLike,
    amplitudes: ArrayLike,
    alpha: float = 1.0,
) -> np.ndarray:
    """Exact heat-equation solution for a sum of sine modes on a periodic grid.

    Evaluates the closed-form decay of each sine mode under
    ``du/dt = alpha * d2u/dx2``:
    ``u(x, t) = offset + sum_m A_m * sin(q_m*(x - x0)) * exp(-alpha*q_m**2*t)``
    with ``q_m = 2*pi*k_m/xn``. Each mode is an eigenfunction of the continuous
    Laplacian with eigenvalue ``-q_m**2``, so it decays in time while the
    constant ``offset`` is stationary.

    Parameters
    ----------
    t : ArrayLike
        Elapsed time(s) since the initial condition, scalar or shape ``(M,)``.
    x : ArrayLike
        Grid coordinates, shape ``(nx,)``.
    x0 : float
        Left boundary, used as the phase origin of every mode.
    xn : float
        Right boundary, used as the spatial period of every mode.
    offset : float
        Constant background added to every mode.
    wavenumbers : ArrayLike
        Mode wavenumbers ``k_m``, shape ``(n_modes,)``.
    amplitudes : ArrayLike
        Mode amplitudes ``A_m``, shape ``(n_modes,)``, aligned with
        ``wavenumbers``.
    alpha : float, optional
        Thermal diffusivity in ``du/dt = alpha * d2u/dx2`` (default ``1.0``,
        the value fixed by this project's right-hand side).

    Returns
    -------
    np.ndarray
        Analytic field, shape ``(M, nx)``, dtype float64; row ``i`` is the
        solution at time ``t[i]``. A scalar ``t`` yields shape ``(1, nx)``.

    Notes
    -----
    The decay uses the *continuous* eigenvalue ``-q_m**2``; a finite-difference
    solver decays each mode slightly differently, so this reference and a
    discrete simulation agree exactly only at ``t = 0``.
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    x = np.asarray(x, dtype=np.float64)
    wavenumbers = np.asarray(wavenumbers, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)

    q = 2.0 * np.pi * wavenumbers / xn
    spatial = amplitudes[:, None] * np.sin(q[:, None] * (x[None, :] - x0))
    decay = np.exp(-alpha * (q**2)[:, None] * t[None, :])
    return offset + decay.T @ spatial


def read_analytic_inputs(path: str | Path) -> AnalyticInputs:
    """Read an output HDF5 file into the inputs of :func:`analytic_solution`.

    Loads the saved grid and times and parses the embedded ``config_yaml``
    attribute -- the only place the initial-condition offset and sine modes are
    stored -- for the grid bounds, offset, modes, and start time.

    Parameters
    ----------
    path : str | Path
        Path to an HDF5 file created by the simulation's output writer.

    Returns
    -------
    AnalyticInputs
        Grid, saved times, grid bounds, offset, per-mode wavenumbers and
        amplitudes, and the initial time.

    Notes
    -----
    ``config_yaml`` is parsed with :func:`yaml.safe_load`, which never executes
    arbitrary code even though the file is read from disk.
    """
    with h5py.File(path, "r") as f:
        x = cast(h5py.Dataset, f["x"])[:]
        t = cast(h5py.Dataset, f["t"])[:]
        config = yaml.safe_load(cast(str, f.attrs["config_yaml"]))

    grid = config["grid"]
    initial_state = config["initial_state"]
    modes = initial_state["modes"]
    wavenumbers = np.array([mode["wavenumber"] for mode in modes], dtype=np.float64)
    amplitudes = np.array([mode["amplitude"] for mode in modes], dtype=np.float64)

    return AnalyticInputs(
        x=np.asarray(x, dtype=np.float64),
        t=np.asarray(t, dtype=np.float64),
        x0=float(grid["x0"]),
        xn=float(grid["xn"]),
        offset=float(initial_state["offset"]),
        wavenumbers=wavenumbers,
        amplitudes=amplitudes,
        t_start=float(config["time"]["start"]),
    )


def analytic_solution_from_output(path: str | Path) -> AnalyticSolution:
    """Analytic heat-equation solution sampled like a simulation output file.

    Reads the grid, saved times, and initial condition from ``path`` and
    evaluates :func:`analytic_solution` at every saved time, so the result can
    be compared point-for-point with the stored numerical field. This is the
    entry point intended for reuse in other analysis scripts.

    Parameters
    ----------
    path : str | Path
        Path to an HDF5 file created by the simulation's output writer.

    Returns
    -------
    AnalyticSolution
        Grid ``x`` of shape ``(nx,)``, saved times ``t`` of shape
        ``(n_saves,)``, and analytic field ``u`` of shape ``(n_saves, nx)``,
        all dtype float64. ``t`` holds the absolute saved times; the decay is
        evaluated at ``t - t_start``.
    """
    inputs = read_analytic_inputs(path)
    elapsed = inputs.t - inputs.t_start
    u = analytic_solution(
        elapsed,
        inputs.x,
        inputs.x0,
        inputs.xn,
        inputs.offset,
        inputs.wavenumbers,
        inputs.amplitudes,
    )
    return AnalyticSolution(x=inputs.x, t=inputs.t, u=u)
