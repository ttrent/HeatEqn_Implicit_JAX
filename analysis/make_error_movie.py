"""Render a two-panel state / fractional-error movie from a solver HDF5 file.

Top panel: the numeric state ``u(x)`` versus the spatial coordinate. Bottom panel:
the fractional error ``|u_num - u_analytic| / |u_analytic|`` versus ``x`` on a symlog
axis, comparing the stored field to the closed-form analytic solution. Frames are
restricted to the early evolution (``--t-max``) where the sine modes actually decay,
rendered in parallel and encoded to H.264 mp4 with ffmpeg.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple, cast

import h5py
import matplotlib
import numpy as np
from analytic_solution import analytic_solution_from_output

matplotlib.use("Agg")  # headless: render without a GUI, safe in worker processes

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

MIN_DPI = 200
DEFAULT_FPS = 30
DEFAULT_T_MAX = 25.0


class FrameTask(NamedTuple):
    """Everything one worker needs to render a single two-panel frame.

    Attributes
    ----------
    out_path : str
        Destination PNG path, zero-padded so ffmpeg orders frames correctly.
    x : np.ndarray
        Grid coordinates, shape ``(nx,)``, dtype float64.
    u_row : np.ndarray
        Numeric field values for this frame, shape ``(nx,)``, dtype float64.
    err_row : np.ndarray
        Fractional error ``|u_num - u_analytic| / |u_analytic|`` for this frame,
        shape ``(nx,)``, dtype float64.
    t_value : float
        Absolute simulation time of this frame, used only for the annotation.
    top_ylim : tuple[float, float]
        Global y-limits for the state panel, shared by every frame.
    err_ylim : tuple[float, float]
        Global y-limits for the error panel, shared by every frame.
    linthresh : float
        Symlog linear-region threshold for the error panel, shared by every frame.
    dpi : int
        Frame resolution; the driver clamps this to at least ``MIN_DPI``.
    """

    out_path: str
    x: np.ndarray
    u_row: np.ndarray
    err_row: np.ndarray
    t_value: float
    top_ylim: tuple[float, float]
    err_ylim: tuple[float, float]
    linthresh: float
    dpi: int


def render_frame(task: FrameTask) -> None:
    """Render one two-panel frame PNG. Runs in a worker process; stays picklable.

    Parameters
    ----------
    task : FrameTask
        The per-frame payload (grid, numeric row, error row, time, shared limits,
        symlog threshold, dpi).
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)

    ax_top.plot(task.x, task.u_row, color="C0")
    ax_top.set_ylim(*task.top_ylim)
    ax_top.set_ylabel("u")
    ax_top.set_title(f"t = {task.t_value:.3f}")
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(task.x, task.err_row, color="C3")
    ax_bot.set_yscale("symlog", linthresh=task.linthresh)
    ax_bot.set_ylim(*task.err_ylim)
    ax_bot.set_xlim(float(task.x[0]), float(task.x[-1]))
    ax_bot.set_xlabel("x")
    ax_bot.set_ylabel(r"$|u_\mathrm{num} - u_\mathrm{ana}|\,/\,|u_\mathrm{ana}|$")
    ax_bot.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(task.out_path, dpi=task.dpi)
    plt.close(fig)  # release the figure or memory grows with every frame


def load_solution(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read grid, numeric field, and times from a solver HDF5 file.

    Parameters
    ----------
    path : Path
        Path to a file written by ``src/io_utils/output.py``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``x`` shape ``(nx,)``, ``u`` shape ``(n_rows, nx)``, ``t`` shape
        ``(n_rows,)``; all dtype float64. Row 0 is the initial condition.
    """
    with h5py.File(path, "r") as f:
        x = cast(h5py.Dataset, f["x"])[:]
        u = cast(h5py.Dataset, f["u"])[:]
        t = cast(h5py.Dataset, f["t"])[:]
    return np.asarray(x), np.asarray(u), np.asarray(t)


def fractional_error(u: np.ndarray, u_analytic: np.ndarray) -> np.ndarray:
    """Absolute fractional error of the numeric field against the analytic field.

    Parameters
    ----------
    u : np.ndarray
        Numeric field, shape ``(n_rows, nx)``, dtype float64.
    u_analytic : np.ndarray
        Analytic field sampled at the same grid and times, shape ``(n_rows, nx)``.

    Returns
    -------
    np.ndarray
        ``|u - u_analytic| / |u_analytic|``, shape ``(n_rows, nx)``, dtype float64.
        Non-finite entries occur only where ``u_analytic`` vanishes (zero-crossing
        fields, e.g. a zero offset) and are filtered out of the shared limits.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(u - u_analytic) / np.abs(u_analytic)


def select_indices(
    t: np.ndarray,
    t_min: float,
    t_max: float,
    stride: int,
    max_frames: int | None,
) -> np.ndarray:
    """Choose which saved rows become frames, restricted to a time window.

    Parameters
    ----------
    t : np.ndarray
        Absolute saved times, shape ``(n_rows,)``.
    t_min : float
        Lower bound (inclusive) of the time window to render.
    t_max : float
        Upper bound (inclusive) of the time window to render.
    stride : int
        Keep every ``stride``-th row within the window (``1`` keeps all rows).
    max_frames : int | None
        Optional cap on the number of frames after striding.

    Returns
    -------
    np.ndarray
        Selected row indices into ``u``/``t``, dtype int.
    """
    indices = np.nonzero((t >= t_min) & (t <= t_max))[0]
    indices = indices[:: max(stride, 1)]
    if max_frames is not None and indices.size > max_frames:
        indices = indices[:max_frames]
    return indices


def global_ylim(u: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    """Compute state-panel y-limits spanning the field so frames do not jump.

    Parameters
    ----------
    u : np.ndarray
        Numeric field values for the selected frames, shape ``(n_frames, nx)``.
    margin : float, optional
        Fractional padding added above and below the data range.

    Returns
    -------
    tuple[float, float]
        ``(low, high)`` limits shared by every frame.
    """
    u_min = float(np.min(u))
    u_max = float(np.max(u))
    pad = margin * ((u_max - u_min) or 1.0)
    return (u_min - pad, u_max + pad)


def global_err_limits(
    frac_err: np.ndarray,
    margin: float = 0.1,
) -> tuple[tuple[float, float], float]:
    """Compute shared error-panel y-limits and a symlog linear threshold.

    Parameters
    ----------
    frac_err : np.ndarray
        Fractional error for the selected frames, shape ``(n_frames, nx)``.
    margin : float, optional
        Fractional headroom added above the peak error.

    Returns
    -------
    tuple[tuple[float, float], float]
        ``((0.0, high), linthresh)``: the error is non-negative, so the panel
        starts at zero; ``linthresh`` sizes the symlog linear region around zero.
    """
    finite = frac_err[np.isfinite(frac_err)]
    err_max = float(finite.max()) if finite.size else 1.0
    err_max = err_max if err_max > 0.0 else 1.0
    linthresh = max(err_max * 1.0e-3, 1.0e-8)
    return (0.0, err_max * (1.0 + margin)), linthresh


def encode_movie(frames_dir: str, output_path: Path, fps: int) -> None:
    """Encode the numbered PNG frames into an H.264 mp4 with ffmpeg.

    Parameters
    ----------
    frames_dir : str
        Directory holding ``frame_%06d.png``.
    output_path : Path
        Destination mp4 path.
    fps : int
        Output frame rate.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(frames_dir, "frame_%06d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # yuv420p needs even dimensions
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the movie build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/heat_solution.h5"))
    parser.add_argument("--output", type=Path, default=Path("output/error_movie.mp4"))
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--dpi", type=int, default=MIN_DPI)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=DEFAULT_T_MAX)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build the movie: load data, render frames in parallel, encode, clean up."""
    args = parse_args()
    dpi = max(args.dpi, MIN_DPI)  # enforce the DPI >= 200 invariant

    x, u, t = load_solution(args.input)
    analytic = analytic_solution_from_output(args.input)
    frac_err = fractional_error(u, analytic.u)

    indices = select_indices(t, args.t_min, args.t_max, args.stride, args.max_frames)
    if indices.size == 0:
        raise SystemExit(
            f"no saved snapshots in time window [{args.t_min}, {args.t_max}]"
        )

    top_ylim = global_ylim(u[indices])
    err_ylim, linthresh = global_err_limits(frac_err[indices])

    if args.frames_dir is not None:
        frames_dir = str(args.frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        remove_after = False
    else:
        frames_dir = tempfile.mkdtemp(prefix="heaterrmovie_")
        remove_after = not args.keep_frames

    tasks = [
        FrameTask(
            out_path=os.path.join(frames_dir, f"frame_{frame_i:06d}.png"),
            x=x,
            u_row=u[row],
            err_row=frac_err[row],
            t_value=float(t[row]),
            top_ylim=top_ylim,
            err_ylim=err_ylim,
            linthresh=linthresh,
            dpi=dpi,
        )
        for frame_i, row in enumerate(indices)
    ]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for _ in executor.map(render_frame, tasks):
            pass  # force iteration so worker exceptions propagate here

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_movie(frames_dir, args.output, args.fps)

    if remove_after:
        shutil.rmtree(frames_dir, ignore_errors=True)

    print(f"wrote {args.output} ({len(tasks)} frames at {args.fps} fps, dpi {dpi})")


if __name__ == "__main__":
    main()
