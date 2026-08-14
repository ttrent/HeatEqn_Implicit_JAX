"""Plot spatial- and temporal-resolution convergence for the heat solver.

Reads the per-run HDF5 files written by ``analysis/convergence_sweep.py`` under
``output/convergence/`` and renders log-log convergence figures. The error is
the RMS over the grid of the absolute fractional error
``|u_num - u_ref| / |u_ref|`` at the final saved time.

Spatial figure: error versus grid spacing ``dx`` for both integrators, compared
to the continuous analytic solution, with the halved-dt runs overlaid so their
overlap confirms dt-independence. Temporal figures (one per integrator): error
versus ``dt`` at a fixed grid, compared to three references -- the continuous
analytic solution (which plateaus at the spatial-error floor), the semi-discrete
exact solution (the discrete-Laplacian eigenmodes, which isolates the
time-integration error), and a smallest-dt self-convergence run. Each
convergence series carries a dashed fitted-order line and a faint ideal-order
guide.
"""

import argparse
from pathlib import Path
from typing import NamedTuple, cast

import h5py
import matplotlib
import numpy as np
import yaml
from analytic_solution import analytic_solution_from_output, read_analytic_inputs

matplotlib.use("Agg")  # headless: render without a GUI

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

MIN_DPI = 200
EXPECTED_SPATIAL_ORDER = 2.0
EXPECTED_TEMPORAL_ORDER = {"backward_euler": 1.0, "implicit_midpoint": 2.0}
INTEGRATOR_COLORS = {"backward_euler": "C0", "implicit_midpoint": "C1"}


class RunData(NamedTuple):
    """One loaded run reduced to what the plots need.

    Attributes
    ----------
    integrator : str
        Time-stepping scheme name.
    nx : int
        Number of grid points.
    dt : float
        Time step.
    dx : float
        Grid spacing.
    x : np.ndarray
        Grid coordinates, shape ``(nx,)``, dtype float64.
    u_final : np.ndarray
        Numeric field at the final saved time, shape ``(nx,)``, dtype float64.
    path : Path
        Source HDF5 file, used to evaluate the analytic references.
    """

    integrator: str
    nx: int
    dt: float
    dx: float
    x: np.ndarray
    u_final: np.ndarray
    path: Path


def load_run(path: Path) -> RunData:
    """Load a run's grid, final field, and parameters from its HDF5 file.

    Parameters
    ----------
    path : Path
        Path to a file written by ``analysis/convergence_sweep.py``.

    Returns
    -------
    RunData
        The grid, final-time field, and the integrator/nx/dt/dx metadata.
    """
    with h5py.File(path, "r") as f:
        x = cast(h5py.Dataset, f["x"])[:]
        u_final = cast(h5py.Dataset, f["u"])[-1]
        nx = int(f.attrs["nx"])
        dx = float(f.attrs["dx"])
        dt = float(f.attrs["dt"])
        config = yaml.safe_load(cast(str, f.attrs["config_yaml"]))
    return RunData(
        integrator=config["methods"]["integrator"],
        nx=nx,
        dt=dt,
        dx=dx,
        x=np.asarray(x, dtype=np.float64),
        u_final=np.asarray(u_final, dtype=np.float64),
        path=path,
    )


def discover_runs(study_dir: Path) -> list[RunData]:
    """Load every run in a study directory.

    Parameters
    ----------
    study_dir : Path
        Directory holding the study's ``*.h5`` files.

    Returns
    -------
    list[RunData]
        Loaded runs, empty if the directory has no HDF5 files.
    """
    return [load_run(path) for path in sorted(study_dir.glob("*.h5"))]


def continuous_final(path: Path) -> np.ndarray:
    """Continuous analytic solution at the final saved time.

    Parameters
    ----------
    path : Path
        Path to a run's HDF5 output file.

    Returns
    -------
    np.ndarray
        Analytic field on the run's grid at the final time, shape ``(nx,)``.
    """
    return analytic_solution_from_output(path).u[-1]


def semidiscrete_final(path: Path) -> np.ndarray:
    """Semi-discrete exact solution at the final saved time.

    Uses the same sine modes as the continuous solution, but decays each with
    the eigenvalue of the periodic three-point Laplacian,
    ``lambda_m = -(2 / dx**2) * (1 - cos(q_m * dx))`` with
    ``q_m = 2 * pi * k_m / xn``. This is the exact solution of the spatially
    discretised ODE system, so comparing to it isolates the time-integration
    error (the spatial truncation error cancels).

    Parameters
    ----------
    path : Path
        Path to a run's HDF5 output file.

    Returns
    -------
    np.ndarray
        Semi-discrete field on the run's grid at the final time, shape ``(nx,)``.
    """
    inputs = read_analytic_inputs(path)
    dx = float(inputs.x[1] - inputs.x[0])
    elapsed = inputs.t[-1] - inputs.t_start
    q = 2.0 * np.pi * inputs.wavenumbers / inputs.xn
    eigenvalues = -(2.0 / dx**2) * (1.0 - np.cos(q * dx))
    spatial = inputs.amplitudes[:, None] * np.sin(
        q[:, None] * (inputs.x[None, :] - inputs.x0)
    )
    decay = np.exp(eigenvalues * elapsed)
    return inputs.offset + decay @ spatial


def rms_fractional_error(u_num: np.ndarray, u_ref: np.ndarray) -> float:
    """RMS over the grid of the absolute fractional error at one time.

    Parameters
    ----------
    u_num : np.ndarray
        Numeric field, shape ``(nx,)``, dtype float64.
    u_ref : np.ndarray
        Reference field, shape ``(nx,)``, dtype float64; must not vanish (the
        offset-10 background keeps it bounded away from zero).

    Returns
    -------
    float
        ``sqrt(mean(((u_num - u_ref) / u_ref) ** 2))``.
    """
    fractional = (u_num - u_ref) / u_ref
    return float(np.sqrt(np.mean(fractional**2)))


def fit_order(x_values: np.ndarray, errors: np.ndarray) -> tuple[float, float]:
    """Fit a power law ``error = c * x**p`` by least squares in log-log space.

    Parameters
    ----------
    x_values : np.ndarray
        Independent variable (``dx`` or ``dt``), shape ``(n,)``, positive.
    errors : np.ndarray
        Measured errors, shape ``(n,)``, positive.

    Returns
    -------
    tuple[float, float]
        The fitted order ``p`` (slope) and the log-intercept ``log10(c)``.
    """
    slope, intercept = np.polyfit(np.log10(x_values), np.log10(errors), 1)
    return float(slope), float(intercept)


def _draw_ideal_guide(
    ax: "plt.Axes",
    x_values: np.ndarray,
    errors: np.ndarray,
    order: float,
) -> None:
    """Overlay a faint reference line of the ideal order through the coarsest point.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on (log-log).
    x_values : np.ndarray
        Independent variable of the fitted series, shape ``(n,)``.
    errors : np.ndarray
        Errors aligned with ``x_values``, shape ``(n,)``.
    order : float
        Ideal convergence order (slope) of the guide line.
    """
    x_sorted = np.unique(x_values)
    anchor_x = float(x_sorted[-1])
    anchor_err = float(errors[np.argmax(x_values)])
    guide = anchor_err * (x_sorted / anchor_x) ** order
    ax.loglog(
        x_sorted, guide, ":", color="gray", alpha=0.6, label=f"ideal: {order:.0f}"
    )


def plot_spatial(runs: list[RunData], out_path: Path, dpi: int) -> None:
    """Render the spatial convergence figure (error versus ``dx``).

    For each integrator the base-dt series is fitted (dashed line plus
    annotation) and the halved-dt series is overlaid; overlapping series confirm
    the errors are dt-independent. A single faint ideal slope-2 guide is drawn.

    Parameters
    ----------
    runs : list[RunData]
        Loaded spatial-study runs (both time steps, both integrators).
    out_path : Path
        Destination PNG path.
    dpi : int
        Figure resolution.
    """
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    base_dx: list[np.ndarray] = []
    base_err: list[np.ndarray] = []

    for integrator in sorted({r.integrator for r in runs}):
        color = INTEGRATOR_COLORS.get(integrator, "C0")
        group = [r for r in runs if r.integrator == integrator]
        base_dt = max(r.dt for r in group)
        for dt in sorted({r.dt for r in group}, reverse=True):
            series = sorted((r for r in group if r.dt == dt), key=lambda r: r.dx)
            dx = np.array([r.dx for r in series])
            err = np.array(
                [
                    rms_fractional_error(r.u_final, continuous_final(r.path))
                    for r in series
                ]
            )
            if dt == base_dt:
                ax.loglog(dx, err, "o-", color=color, label=integrator)
                order, intercept = fit_order(dx, err)
                ax.loglog(
                    dx,
                    10.0**intercept * dx**order,
                    "--",
                    color=color,
                    alpha=0.8,
                    label=f"{integrator} fit: {order:.2f}",
                )
                base_dx.append(dx)
                base_err.append(err)
            else:
                ax.loglog(
                    dx, err, "x:", color=color, alpha=0.4, label=f"{integrator} dt/2"
                )

    if base_dx:
        _draw_ideal_guide(
            ax,
            np.concatenate(base_dx),
            np.concatenate(base_err),
            EXPECTED_SPATIAL_ORDER,
        )

    ax.set_xlabel("dx")
    ax.set_ylabel("RMS fractional error at final time")
    ax.set_title("Spatial convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_temporal(
    runs: list[RunData], integrator: str, out_path: Path, dpi: int
) -> None:
    """Render a temporal convergence figure for one integrator (error versus ``dt``).

    Plots the error against three references: the continuous analytic solution
    (which plateaus at the spatial-error floor), the semi-discrete exact solution
    (which isolates the time-integration error and is fitted), and the
    smallest-dt run (self-convergence). A faint ideal-order guide is overlaid.

    Parameters
    ----------
    runs : list[RunData]
        Loaded temporal-study runs (all integrators; filtered here).
    integrator : str
        Integrator whose runs are plotted.
    out_path : Path
        Destination PNG path.
    dpi : int
        Figure resolution.
    """
    group = [r for r in runs if r.integrator == integrator]
    reference = min(group, key=lambda r: r.dt)  # smallest dt: self-convergence base
    tested = sorted((r for r in group if r is not reference), key=lambda r: r.dt)

    dt = np.array([r.dt for r in tested])
    err_continuous = np.array(
        [rms_fractional_error(r.u_final, continuous_final(r.path)) for r in tested]
    )
    err_semidiscrete = np.array(
        [rms_fractional_error(r.u_final, semidiscrete_final(r.path)) for r in tested]
    )
    err_self = np.array(
        [rms_fractional_error(r.u_final, reference.u_final) for r in tested]
    )

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.loglog(dt, err_continuous, "o-", color="C3", label="vs continuous (plateaus)")
    ax.loglog(dt, err_semidiscrete, "s-", color="C0", label="vs semi-discrete")
    ax.loglog(dt, err_self, "^-", color="C2", label="vs self-convergence")

    order, intercept = fit_order(dt, err_semidiscrete)
    ax.loglog(
        dt,
        10.0**intercept * dt**order,
        "--",
        color="C0",
        alpha=0.8,
        label=f"semi-discrete fit: {order:.2f}",
    )
    _draw_ideal_guide(ax, dt, err_semidiscrete, EXPECTED_TEMPORAL_ORDER[integrator])

    ax.set_xlabel("dt")
    ax.set_ylabel("RMS fractional error at final time")
    ax.set_title(f"Temporal convergence: {integrator}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output/convergence"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/convergence"))
    parser.add_argument("--dpi", type=int, default=MIN_DPI)
    return parser.parse_args()


def main() -> None:
    """Build the spatial figure and one temporal figure per integrator."""
    args = parse_args()
    dpi = max(args.dpi, MIN_DPI)

    spatial = discover_runs(args.input_dir / "spatial")
    if spatial:
        plot_spatial(spatial, args.output_dir / "spatial_convergence.png", dpi)

    temporal = discover_runs(args.input_dir / "temporal")
    for integrator in sorted({r.integrator for r in temporal}):
        plot_temporal(
            temporal,
            integrator,
            args.output_dir / f"temporal_convergence_{integrator}.png",
            dpi,
        )


if __name__ == "__main__":
    main()
