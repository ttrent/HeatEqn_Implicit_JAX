"""Write simulation results and run metadata to a single HDF5 file."""

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
from numpy.typing import ArrayLike

from io_utils.input_config import SimParams
from state import State

if TYPE_CHECKING:
    from solvers._types import SolverDiagnostics

_CHUNK_ROWS = 64  # rows per HDF5 chunk for the resizable, time-indexed datasets


def _git_info() -> dict[str, str | bool]:
    """Collect git revision metadata for the current repository.

    Runs ``git`` in this file's directory to read the commit hash, branch,
    ``git describe`` string, and whether the working tree has uncommitted
    changes. Any failure (``git`` missing or not a repository) falls back to
    ``"unknown"`` / ``False`` so writing output never blocks a run.

    Returns
    -------
    dict[str, str | bool]
        Keys ``commit``, ``branch``, ``describe`` (str) and ``dirty`` (bool).
    """
    repo_dir = Path(__file__).resolve().parent

    def _run(args: list[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    try:
        commit = _run(["rev-parse", "HEAD"])
        branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
        describe = _run(["describe", "--tags", "--always", "--dirty"])
        dirty = bool(_run(["status", "--porcelain"]))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "commit": "unknown",
            "branch": "unknown",
            "describe": "unknown",
            "dirty": False,
        }
    return {
        "commit": commit,
        "branch": branch,
        "describe": describe,
        "dirty": dirty,
    }


def create_output_file(params: SimParams, config_path: str | Path) -> Path:
    """Create the HDF5 output file with run metadata and empty datasets.

    Stores the raw config YAML, a UTC creation timestamp, and git revision
    info as root attributes, saves the spatial grid, and creates the growable,
    time-indexed datasets that :func:`append_snapshot` extends one row per
    save. Call once, right after the config is loaded.

    Parameters
    ----------
    params : SimParams
        Parsed simulation parameters; supplies the output destination, grid
        (``nx``, ``dx``, ``x0``), and time step ``dt``.
    config_path : str | Path
        Path to the YAML config file, stored verbatim in the ``config_yaml``
        attribute for reproducibility.

    Returns
    -------
    Path
        Path to the created HDF5 file.

    Notes
    -----
    The field ``u`` is stored as ``(n_saves, nx)`` float64; the time ``t``,
    ``step`` index, and the ``diagnostics`` group (``iterations``,
    ``converged``, ``residual_norm``) are aligned length-``n_saves`` datasets.
    All grow along the leading (time) axis.
    """
    grid = params.grid
    nx = grid.size
    out_dir = Path(params.output.directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / params.output.filename

    x = grid.x0 + grid.dx * np.arange(nx, dtype=np.float64)
    config_yaml = Path(config_path).read_text(encoding="utf-8")
    git = _git_info()

    with h5py.File(out_path, "w") as f:
        f.attrs["config_yaml"] = config_yaml
        f.attrs["created_utc"] = datetime.now(UTC).isoformat()
        f.attrs["git_commit"] = git["commit"]
        f.attrs["git_branch"] = git["branch"]
        f.attrs["git_describe"] = git["describe"]
        f.attrs["git_dirty"] = git["dirty"]
        f.attrs["nx"] = nx
        f.attrs["dx"] = grid.dx
        f.attrs["dt"] = params.time.step_size

        f.create_dataset("x", data=x)
        f.create_dataset(
            "u",
            shape=(0, nx),
            maxshape=(None, nx),
            chunks=(_CHUNK_ROWS, nx),
            dtype=np.float64,
        )
        f.create_dataset(
            "t",
            shape=(0,),
            maxshape=(None,),
            chunks=(_CHUNK_ROWS,),
            dtype=np.float64,
        )
        f.create_dataset(
            "step",
            shape=(0,),
            maxshape=(None,),
            chunks=(_CHUNK_ROWS,),
            dtype=np.int64,
        )

        diagnostics = f.create_group("diagnostics")
        diagnostics.create_dataset(
            "iterations",
            shape=(0,),
            maxshape=(None,),
            chunks=(_CHUNK_ROWS,),
            dtype=np.int64,
        )
        diagnostics.create_dataset(
            "converged",
            shape=(0,),
            maxshape=(None,),
            chunks=(_CHUNK_ROWS,),
            dtype=np.bool_,
        )
        diagnostics.create_dataset(
            "residual_norm",
            shape=(0,),
            maxshape=(None,),
            chunks=(_CHUNK_ROWS,),
            dtype=np.float64,
        )

    return out_path


def _append_row(group: h5py.Group, name: str, value: ArrayLike) -> None:
    """Extend a resizable dataset by one row and write ``value`` at the end.

    Parameters
    ----------
    group : h5py.Group
        Group (or file) holding the dataset.
    name : str
        Name of the dataset, resizable along its leading axis.
    value : ArrayLike
        Row payload written at the new last index.
    """
    dataset = cast(h5py.Dataset, group[name])
    index = dataset.shape[0]
    dataset.resize(index + 1, axis=0)
    dataset[index] = value


def append_snapshot(
    path: str | Path,
    state: State,
    step: int,
    diagnostics: "SolverDiagnostics | None" = None,
) -> None:
    """Append one time slice, extending every time-indexed dataset by a row.

    Moves the field to host memory and writes the field, time, step index, and
    solver diagnostics as a new aligned row. Reopens the file in append mode
    each call, so the two output functions stay decoupled and every save is
    flushed to disk. Call outside any ``jit`` / ``lax.scan`` region, since it
    reads concrete (host) values.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file created by :func:`create_output_file`.
    state : State
        Snapshot to record; ``state.u`` has shape ``(nx,)``, dtype float64,
        and ``state.t`` is the scalar time.
    step : int
        Global time-step index of this snapshot.
    diagnostics : SolverDiagnostics | None, optional
        Solver telemetry for the step that produced ``state``. ``None`` (e.g.
        the initial condition, which involved no solve) writes a sentinel row
        of ``iterations=0``, ``converged=True``, ``residual_norm=NaN``.
    """
    u = np.asarray(state.u, dtype=np.float64)
    if diagnostics is None:
        iterations, converged, residual_norm = 0, True, float("nan")
    else:
        iterations = int(diagnostics.iterations)
        converged = bool(diagnostics.converged)
        residual_norm = float(diagnostics.residual_norm)

    with h5py.File(path, "a") as f:
        _append_row(f, "u", u)
        _append_row(f, "t", float(state.t))
        _append_row(f, "step", int(step))
        diagnostics_group = cast(h5py.Group, f["diagnostics"])
        _append_row(diagnostics_group, "iterations", iterations)
        _append_row(diagnostics_group, "converged", converged)
        _append_row(diagnostics_group, "residual_norm", residual_norm)
