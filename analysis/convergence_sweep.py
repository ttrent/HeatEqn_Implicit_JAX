"""Run spatial- and temporal-resolution convergence studies for the solver.

Generates one ephemeral YAML config per run -- so the derived fields in
``load_config`` (``dx``, ``n_steps``, ``save_rate.steps``) are recomputed
correctly -- then invokes ``src/main.py`` as an isolated subprocess and keeps
each run's HDF5 output under ``output/convergence/``.

The spatial study refines the grid at a fixed time step -- chosen per integrator
so the time-integration error stays below the spatial truncation error over the
grids it covers -- and re-runs each grid at half the step for a dt-independence
check. The temporal study fixes the grid and refines the step, adding a
smallest-dt reference run for self-convergence. Solver tolerances are tightened
for every run so the implicit-solve floor stays below the finest measured error.
Analysis and plotting live in ``analysis/plot_convergence.py``; this script only
produces the runs.
"""

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple, cast

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "not_tracked" / "config.yml"
CONVERGENCE_DIR = REPO_ROOT / "output" / "convergence"

START = 0.0
END = 2.0  # short window: the sine modes are still alive, not yet decayed to offset
INTEGRATORS = ("backward_euler", "implicit_midpoint")

# Spatial study: refine the grid at a fixed step, chosen per integrator so the
# time-integration error stays below the spatial truncation error over the grids
# it covers. implicit_midpoint (order 2) stays clean at dt=1e-3 across the full
# ladder; backward_euler (order 1) needs dt=1e-4 and is only trustworthy up to
# nx=256 (finer grids plateau at its O(dt) temporal floor, at a cost of millions
# of steps to push lower). Each entry is (integrator, base dt, grid sizes); every
# grid is also run at half the step.
SPATIAL_SWEEPS = (
    ("implicit_midpoint", 1.0e-3, (32, 64, 128, 256, 512, 1024, 2048)),
    ("backward_euler", 1.0e-4, (32, 64, 128, 256)),
)

# Temporal study: refine dt at a fixed grid; the smallest dt is the
# self-convergence reference run.
TEMPORAL_NX = 256
TEMPORAL_DTS = (0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125)
TEMPORAL_REF_DT = 0.000390625

# Tightened implicit-solve controls; only these three fields differ from the
# base config so the solver floor sits well below the finest measured error.
SOLVER_OVERRIDES = {
    "absolute_tolerance": 1.0e-12,
    "relative_tolerance": 1.0e-10,
    "max_iterations": 30,
}


class Run(NamedTuple):
    """One simulation to execute in the sweep.

    Attributes
    ----------
    study : str
        Study name, ``"spatial"`` or ``"temporal"``; also the output subdirectory.
    integrator : str
        Time-stepping scheme name.
    nx : int
        Number of grid points.
    dt : float
        Time step.
    """

    study: str
    integrator: str
    nx: int
    dt: float


def output_path(run: Run) -> Path:
    """Destination HDF5 path for a run, encoding its parameters in the name.

    Parameters
    ----------
    run : Run
        The run whose output path is built.

    Returns
    -------
    Path
        ``output/convergence/<study>/<integrator>_nx<nx>_dt<dt>.h5``; the base
        and halved time steps yield distinct names within the same directory.
    """
    name = f"{run.integrator}_nx{run.nx:04d}_dt{run.dt:.3e}.h5"
    return CONVERGENCE_DIR / run.study / name


def build_config(base: dict[str, Any], run: Run) -> dict[str, Any]:
    """Build a raw config dict for a run by overriding the base config.

    Copies ``base`` and overrides the grid size, time window and step, output
    cadence (save only the final state), integrator, solver tolerances, and
    output destination. The result is raw YAML data, so ``load_config``
    recomputes every derived field.

    Parameters
    ----------
    base : dict[str, Any]
        Parsed base configuration (from ``not_tracked/config.yml``).
    run : Run
        Parameters for this run.

    Returns
    -------
    dict[str, Any]
        Raw configuration ready to serialise to YAML.
    """
    config = copy.deepcopy(base)
    config["grid"]["size"] = run.nx
    config["time"]["start"] = START
    config["time"]["end"] = END
    config["time"]["step_size"] = run.dt
    # unit=time with value=END saves exactly one snapshot, landing on t=END.
    config["time"]["save_rate"] = {"unit": "time", "value": END}
    config["methods"]["integrator"] = run.integrator
    config["solver"].update(SOLVER_OVERRIDES)
    out = output_path(run)
    config["output"]["directory"] = str(out.parent)
    config["output"]["filename"] = out.name
    return config


def run_simulation(config: dict[str, Any]) -> None:
    """Run one simulation in an isolated subprocess.

    Serialises ``config`` to a temporary YAML file and invokes ``src/main.py``
    with the current interpreter, giving each run a fresh process and XLA cache.
    ``main.py`` embeds the config text in its HDF5 output, so the temporary file
    is deleted afterwards.

    Parameters
    ----------
    config : dict[str, Any]
        Raw configuration to run.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
        config_path = handle.name
    try:
        subprocess.run(
            [sys.executable, "src/main.py", config_path],
            cwd=REPO_ROOT,
            check=True,
        )
    finally:
        Path(config_path).unlink(missing_ok=True)


def all_converged(path: Path) -> bool:
    """Whether every saved step of a completed run converged.

    Parameters
    ----------
    path : Path
        Path to a run's HDF5 output file.

    Returns
    -------
    bool
        ``True`` if all ``diagnostics/converged`` flags are set.
    """
    with h5py.File(path, "r") as f:
        converged = cast(h5py.Dataset, f["diagnostics/converged"])[:]
    return bool(np.all(converged))


def spatial_runs() -> list[Run]:
    """Build the spatial-study run list (grid refinement at a fixed step).

    Returns
    -------
    list[Run]
        Every ``(integrator, nx)`` in each integrator's ladder, at both its base
        step and half that step.
    """
    return [
        Run("spatial", integrator, nx, dt)
        for integrator, base_dt, nx_values in SPATIAL_SWEEPS
        for nx in nx_values
        for dt in (base_dt, base_dt / 2.0)
    ]


def temporal_runs() -> list[Run]:
    """Build the temporal-study run list (dt refinement at fixed grid).

    Returns
    -------
    list[Run]
        Every ``(integrator, dt)`` including the smallest-dt reference run.
    """
    return [
        Run("temporal", integrator, TEMPORAL_NX, dt)
        for integrator in INTEGRATORS
        for dt in (*TEMPORAL_DTS, TEMPORAL_REF_DT)
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=("spatial", "temporal", "all"),
        default="all",
        help="Which convergence study to run (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the runs without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected convergence study serially, one subprocess per run."""
    args = parse_args()

    runs: list[Run] = []
    if args.study in ("spatial", "all"):
        runs.extend(spatial_runs())
    if args.study in ("temporal", "all"):
        runs.extend(temporal_runs())

    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    failed: list[Path] = []
    nonconverged: list[Path] = []

    for index, run in enumerate(runs, start=1):
        out = output_path(run)
        print(
            f"[{index}/{len(runs)}] {run.study} {run.integrator} "
            f"nx={run.nx} dt={run.dt:.3e} -> {out}"
        )
        if args.dry_run:
            continue
        try:
            run_simulation(build_config(base, run))
        except subprocess.CalledProcessError as error:
            failed.append(out)
            print(f"  ERROR: run failed ({error}); continuing.", file=sys.stderr)
            continue
        if not all_converged(out):
            nonconverged.append(out)
            print(f"  WARNING: non-converged steps in {out}", file=sys.stderr)

    _report(failed, nonconverged)


def _report(failed: list[Path], nonconverged: list[Path]) -> None:
    """Print a closing summary of failed and non-converged runs to stderr.

    Parameters
    ----------
    failed : list[Path]
        Runs whose subprocess exited with an error.
    nonconverged : list[Path]
        Completed runs that contained a non-converged implicit step.
    """
    for label, paths in (("failed", failed), ("non-converged", nonconverged)):
        if paths:
            print(f"\n{len(paths)} run(s) {label}:", file=sys.stderr)
            for path in paths:
                print(f"  {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
