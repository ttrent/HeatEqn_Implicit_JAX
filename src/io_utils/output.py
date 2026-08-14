"""Write simulation results and run metadata to a single HDF5 file."""

import resource
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
    time-indexed datasets that :func:`append_snapshot` and
    :func:`append_snapshots` extend as the run progresses. Call once, right
    after the config is loaded.

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


def _append_rows(group: h5py.Group, name: str, values: ArrayLike) -> None:
    """Extend a resizable dataset along its leading axis and write ``values``.

    Accepts either a single row or a block of rows. A single row (``values``
    with one axis fewer than the dataset) is promoted to a block of one;
    otherwise the row count is taken from the leading axis of ``values``.

    Parameters
    ----------
    group : h5py.Group
        Group (or file) holding the dataset.
    name : str
        Name of the dataset, resizable along its leading axis.
    values : ArrayLike
        One row, or a block of rows stacked along the leading axis, written at
        the end of the dataset.
    """
    dataset = cast(h5py.Dataset, group[name])
    block = np.asarray(values)
    if block.ndim == dataset.ndim - 1:
        block = block[np.newaxis]
    n_rows = block.shape[0]
    index = dataset.shape[0]
    dataset.resize(index + n_rows, axis=0)
    dataset[index : index + n_rows] = block


def append_snapshots(
    path: str | Path,
    states: State,
    steps: ArrayLike,
    diagnostics: "SolverDiagnostics | None" = None,
) -> None:
    """Append one or more time slices, sizing each dataset from the data.

    Moves the fields to host memory and writes the fields, times, step indices,
    and solver diagnostics as one aligned block. ``states`` may hold a single
    snapshot (``states.u`` shape ``(nx,)``) or a batch (``(n_rows, nx)``);
    :func:`_append_rows` extends every dataset to match. Reopens the file in
    append mode once, so a full write buffer is flushed together. Call outside
    any ``jit`` / ``lax.scan`` region, since it reads concrete (host) values.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file created by :func:`create_output_file`.
    states : State
        One snapshot or a stack of them; ``states.u`` has shape ``(nx,)`` or
        ``(n_rows, nx)`` and ``states.t`` the matching scalar or ``(n_rows,)``.
    steps : ArrayLike
        Global time-step index of each snapshot, scalar or shape ``(n_rows,)``.
    diagnostics : SolverDiagnostics | None, optional
        Per-snapshot solver telemetry, each leaf scalar or shape ``(n_rows,)``.
        ``None`` (e.g. the initial condition, which involved no solve) writes a
        sentinel of ``iterations=0``, ``converged=True``, ``residual_norm=NaN``
        for every row.
    """
    u = np.asarray(states.u, dtype=np.float64)
    t = np.asarray(states.t, dtype=np.float64)
    step_index = np.asarray(steps, dtype=np.int64)
    if diagnostics is None:
        n_rows = u.shape[0] if u.ndim == 2 else 1
        iterations = np.zeros(n_rows, dtype=np.int64)
        converged = np.ones(n_rows, dtype=np.bool_)
        residual_norm = np.full(n_rows, np.nan, dtype=np.float64)
    else:
        iterations = np.asarray(diagnostics.iterations, dtype=np.int64)
        converged = np.asarray(diagnostics.converged, dtype=np.bool_)
        residual_norm = np.asarray(diagnostics.residual_norm, dtype=np.float64)

    with h5py.File(path, "a") as f:
        _append_rows(f, "u", u)
        _append_rows(f, "t", t)
        _append_rows(f, "step", step_index)
        diagnostics_group = cast(h5py.Group, f["diagnostics"])
        _append_rows(diagnostics_group, "iterations", iterations)
        _append_rows(diagnostics_group, "converged", converged)
        _append_rows(diagnostics_group, "residual_norm", residual_norm)


def _format_clock(seconds: float) -> str:
    """Format a duration like GNU ``time``: ``m:ss.ss`` or ``h:mm:ss``.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        ``"m:ss.ss"`` for durations under an hour, otherwise ``"h:mm:ss"``.
    """
    total_minutes, secs = divmod(seconds, 60.0)
    hours, minutes = divmod(int(total_minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{int(secs):02d}"
    return f"{minutes}:{secs:05.2f}"


def _resource_usage() -> dict[str, int | float]:
    """Collect this process's OS resource usage via ``getrusage``.

    Reads ``resource.RUSAGE_SELF``, the same source ``/usr/bin/time -v`` draws
    from, so the values mirror that tool's report for the running process.
    Counters the Linux kernel does not track (swaps, socket messages, ...) are
    returned as ``0`` by ``getrusage`` itself.

    Returns
    -------
    dict[str, int | float]
        User/system CPU seconds, peak resident set size (KiB), page-fault,
        context-switch, block-I/O, swap, and signal counters, and the system
        page size in bytes.

    Notes
    -----
    ``ru_maxrss`` is in kibibytes on Linux (this project's target), so it is
    reported as ``max_rss_kb``.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_time_s": usage.ru_utime,
        "system_time_s": usage.ru_stime,
        "max_rss_kb": usage.ru_maxrss,
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
        "voluntary_ctx_switches": usage.ru_nvcsw,
        "involuntary_ctx_switches": usage.ru_nivcsw,
        "fs_inputs": usage.ru_inblock,
        "fs_outputs": usage.ru_oublock,
        "swaps": usage.ru_nswap,
        "signals": usage.ru_nsignals,
        "page_size_bytes": resource.getpagesize(),
    }


def _format_time_v(
    usage: dict[str, int | float],
    elapsed_time: float,
    compile_time: float,
    run_time: float,
) -> str:
    """Render resource usage as a ``/usr/bin/time -v``-style text report.

    Parameters
    ----------
    usage : dict[str, int | float]
        Resource counters from :func:`_resource_usage`.
    elapsed_time : float
        Total wall-clock time of the run, in seconds.
    compile_time : float
        Wall-clock time spent warming the XLA cache, in seconds.
    run_time : float
        Wall-clock time spent in the timed integration loop, in seconds.

    Returns
    -------
    str
        Multi-line summary mirroring ``/usr/bin/time -v``, prefixed with the
        separate compile and run wall-clock timings.
    """
    cpu_time = usage["user_time_s"] + usage["system_time_s"]
    percent_cpu = round(100 * cpu_time / elapsed_time) if elapsed_time > 0 else 0
    lines = [
        f"Compile wall-clock time (seconds): {compile_time:.3f}",
        f"Run wall-clock time (seconds): {run_time:.3f}",
        f"User time (seconds): {usage['user_time_s']:.2f}",
        f"System time (seconds): {usage['system_time_s']:.2f}",
        f"Percent of CPU this job got: {percent_cpu}%",
        f"Elapsed (wall clock) time (h:mm:ss or m:ss): {_format_clock(elapsed_time)}",
        f"Maximum resident set size (kbytes): {usage['max_rss_kb']}",
        f"Major (requiring I/O) page faults: {usage['major_page_faults']}",
        f"Minor (reclaiming a frame) page faults: {usage['minor_page_faults']}",
        f"Voluntary context switches: {usage['voluntary_ctx_switches']}",
        f"Involuntary context switches: {usage['involuntary_ctx_switches']}",
        f"Swaps: {usage['swaps']}",
        f"File system inputs: {usage['fs_inputs']}",
        f"File system outputs: {usage['fs_outputs']}",
        f"Signals delivered: {usage['signals']}",
        f"Page size (bytes): {usage['page_size_bytes']}",
    ]
    return "\n".join(lines)


def write_run_stats(
    path: str | Path,
    *,
    elapsed_time: float,
    compile_time: float,
    run_time: float,
) -> None:
    """Write wall-clock timings and OS resource usage to the output file.

    Records how the run went as a ``run_stats`` group: structured scalar
    attributes for later analysis plus a ``summary`` text attribute mirroring
    ``/usr/bin/time -v`` for quick reading. Resource counters come from
    :func:`_resource_usage` (``getrusage(RUSAGE_SELF)``); reading them here, at
    the end of the run, captures the true peak resident set size. Call once, as
    the final step, outside any ``jit`` region.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file created by :func:`create_output_file`.
    elapsed_time : float
        Wall-clock time of the run, in seconds.
    compile_time : float
        Wall-clock time spent warming the XLA cache, in seconds.
    run_time : float
        Wall-clock time spent in the timed integration loop, in seconds.

    Notes
    -----
    ``max_rss_kb`` is in kibibytes (Linux ``getrusage`` convention). Counters the
    Linux kernel does not track are stored as ``0``.
    """
    usage = _resource_usage()
    cpu_time = usage["user_time_s"] + usage["system_time_s"]
    summary = _format_time_v(usage, elapsed_time, compile_time, run_time)

    with h5py.File(path, "a") as f:
        stats = f.create_group("run_stats")
        stats.attrs["elapsed_s"] = elapsed_time
        stats.attrs["compile_s"] = compile_time
        stats.attrs["run_s"] = run_time
        stats.attrs["percent_cpu"] = (
            round(100 * cpu_time / elapsed_time) if elapsed_time > 0 else 0
        )
        stats.attrs["finished_utc"] = datetime.now(UTC).isoformat()
        for name, value in usage.items():
            stats.attrs[name] = value
        stats.attrs["summary"] = summary
