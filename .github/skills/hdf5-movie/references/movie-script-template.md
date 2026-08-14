# Movie script template

A complete, ruff-clean starting point for a movie-making script in `analysis/`. Copy it
into a request-specific file (e.g. `analysis/make_field_movie.py`) and change **only** the
region marked `CUSTOMIZE` to draw what the user asked for. Everything else encodes the
skill invariants — Matplotlib frames, `ProcessPoolExecutor` over all cores, temp-dir
frames, DPI >= 200, 30 fps default, H.264 mp4 via ffmpeg — and should stay as-is.

The per-frame worker (`render_frame`) is a module-level function taking one picklable
`FrameTask`, so `ProcessPoolExecutor` can serialize it. The driver is under an
`if __name__ == "__main__":` guard, which multiprocessing requires.

```python
"""Render an HDF5 solution field to an mp4 movie with parallel Matplotlib frames."""

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

matplotlib.use("Agg")  # headless: render without a GUI, safe in worker processes

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

MIN_DPI = 200
DEFAULT_FPS = 30


class FrameTask(NamedTuple):
    """Everything one worker needs to render a single frame.

    Attributes
    ----------
    out_path : str
        Destination PNG path, zero-padded so ffmpeg orders frames correctly.
    x : np.ndarray
        Grid coordinates, shape ``(nx,)``, dtype float64.
    u_row : np.ndarray
        Field values for this frame, shape ``(nx,)``, dtype float64.
    t_value : float
        Absolute simulation time of this frame, used only for the annotation.
    ylim : tuple[float, float]
        Global y-limits shared by every frame so the axes never jump.
    dpi : int
        Frame resolution; the driver clamps this to at least ``MIN_DPI``.
    """

    out_path: str
    x: np.ndarray
    u_row: np.ndarray
    t_value: float
    ylim: tuple[float, float]
    dpi: int


def render_frame(task: FrameTask) -> None:
    """Render one frame PNG. Runs in a worker process; must stay picklable.

    Parameters
    ----------
    task : FrameTask
        The per-frame payload (grid, field row, time, shared limits, dpi).
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    # --- CUSTOMIZE: draw what the user asked for from the task fields ---------
    ax.plot(task.x, task.u_row, color="C0")
    ax.set_xlim(float(task.x[0]), float(task.x[-1]))
    ax.set_ylim(*task.ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"t = {task.t_value:.4f}")
    # --- end CUSTOMIZE -------------------------------------------------------

    fig.tight_layout()
    fig.savefig(task.out_path, dpi=task.dpi)
    plt.close(fig)  # release the figure or memory grows with every frame


def load_solution(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read grid, field, and times from a solver HDF5 file.

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


def select_indices(n_rows: int, stride: int, max_frames: int | None) -> np.ndarray:
    """Choose which saved rows become frames.

    Parameters
    ----------
    n_rows : int
        Number of saved snapshots in the file.
    stride : int
        Keep every ``stride``-th row (``1`` keeps all rows).
    max_frames : int | None
        Optional cap on the number of frames after striding.

    Returns
    -------
    np.ndarray
        Selected row indices into ``u``/``t``, dtype int.
    """
    indices = np.arange(0, n_rows, max(stride, 1))
    if max_frames is not None and indices.size > max_frames:
        indices = indices[:max_frames]
    return indices


def global_ylim(u: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    """Compute y-limits spanning the whole field so frames do not jump.

    Parameters
    ----------
    u : np.ndarray
        Field values, shape ``(n_rows, nx)``.
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
    parser.add_argument("--output", type=Path, default=Path("output/heat_movie.mp4"))
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--dpi", type=int, default=MIN_DPI)
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
    indices = select_indices(u.shape[0], args.stride, args.max_frames)
    ylim = global_ylim(u)

    if args.frames_dir is not None:
        frames_dir = str(args.frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        remove_after = False
    else:
        frames_dir = tempfile.mkdtemp(prefix="heatmovie_")
        remove_after = not args.keep_frames

    tasks = [
        FrameTask(
            out_path=os.path.join(frames_dir, f"frame_{frame_i:06d}.png"),
            x=x,
            u_row=u[row],
            t_value=float(t[row]),
            ylim=ylim,
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
```

## Adapt per request

- Rewrite only the `CUSTOMIZE` block, and extend `FrameTask` with any extra per-frame data
  the new plot needs (e.g. an analytic row, a second field, diagnostics).
- For an analytic overlay, reuse `analysis/analytic_solution.py`
  (`analytic_solution_from_output`) on the host and pass each frame its row.
- For very large fields, drop the array fields from `FrameTask` and have `render_frame`
  reopen the file by path and read only its row (h5py handles are not picklable).
- Rename `--output` and the script file to match the subject
  (e.g. `analysis/make_error_movie.py`, `output/error_movie.mp4`).
