---
name: hdf5-movie
description: 'Use when asked to make, create, render, or generate a movie, animation, video, or mp4 from this project HDF5 simulation output (output/heat_solution.h5) from the implicit heat-equation solver. Defines the fixed pipeline: render per-frame plots with Matplotlib, parallelized across all CPU cores, into a temporary frame directory, then encode to H.264 mp4 with ffmpeg. Enforces the project invariants (DPI >= 200, 30 fps default, script written into analysis/, runs in the HeatEqnImpJAX conda env). The visualization content is supplied by the user each request; this skill fixes only the steps from data to movie.'
---

# HDF5 Simulation Movie

Turn a solver HDF5 output file into a video. This skill fixes **how** you get from data
to an mp4 — Matplotlib frames, rendered in parallel across every core, staged in a temp
directory, encoded by ffmpeg. It says **nothing** about *what* the movie shows: the user
describes the visualization each time (field vs. x, error vs. analytic, diagnostics, a
heatmap panel, etc.), and you build exactly that in the per-frame render function.

For the data layout (`u`, `t`, `x`, `step`, the row-0 initial condition, `config_yaml`),
use the `hdf5-output` skill and reuse the reader in `analysis/analytic_solution.py`.

## Invariants (always hold unless the user overrides)

1. **Matplotlib** renders every frame (`Agg` backend — no GUI).
2. **Parallel over all cores**: `concurrent.futures.ProcessPoolExecutor` with
   `max_workers=os.cpu_count()`. No new dependencies.
3. **Frames go to a temporary directory** (`tempfile.mkdtemp`) — override only if the user
   names a directory.
4. **ffmpeg** encodes the frames into the movie (`subprocess.run(..., check=True)`).
5. **DPI is at least 200** for every frame.
6. **30 fps** unless the user asks for another rate.
7. The movie-making **script is written into `analysis/`** unless the user says elsewhere.
8. Output is **H.264 mp4** (`libx264`, `-pix_fmt yuv420p`) written into `output/`.
9. **Temp frames are deleted** after a successful encode unless the user says keep them.
10. The script obeys `.github/instructions/jax-style.instructions.md` and runs inside the
    **`HeatEqnImpJAX`** conda env.

## Procedure

1. **Clarify the visualization.** Confirm what each frame shows and the source `.h5` path
   (default `output/heat_solution.h5`). Everything below is fixed regardless of the answer.
2. **Write a fresh script** into `analysis/` (e.g. `analysis/make_<subject>_movie.py`).
   Start from [references/movie-script-template.md](./references/movie-script-template.md)
   and replace only the marked `CUSTOMIZE` render body. Do not create a single reusable
   catch-all script — tailor one per request.
3. **Load data once** on the host with `h5py` (see the `hdf5-output` skill). Row 0 is the
   initial condition; `residual_norm[0]` is a `NaN` sentinel.
4. **Compute global limits once** from the full dataset (y-limits, color scale, etc.) and
   pass them to every frame so the axes do not jump between frames.
5. **Render frames in parallel.** A top-level, picklable worker renders one frame index to
   `frame_%06d.png` at `dpi >= 200`. Map it over all frames with a
   `ProcessPoolExecutor(max_workers=os.cpu_count())`.
6. **Encode with ffmpeg** from the temp directory (command below).
7. **Clean up** the temp directory on success (unless the user asked to keep frames), and
   print the output path.
8. **Lint and run** (see Commands).

## ffmpeg command

```bash
ffmpeg -y -framerate <FPS> -i <FRAMES_DIR>/frame_%06d.png \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart \
  <OUTPUT_DIR>/<name>.mp4
```

- `-framerate` goes **before** `-i` (input rate); default `<FPS>` is 30.
- The `pad` filter forces even width/height — `yuv420p` requires it, and odd dims from
  `figsize * dpi` otherwise make ffmpeg fail.
- Build this as a list and run via `subprocess.run(cmd, check=True)`, not a shell string.

## Critical gotchas

- **Picklable worker.** The per-frame function must be a module-level `def` (no closures,
  no lambdas) or `ProcessPoolExecutor` cannot serialize it.
- **`__main__` guard.** Wrap the driver in `if __name__ == "__main__":` — required for
  multiprocessing safety.
- **`Agg` backend.** `import matplotlib; matplotlib.use("Agg")` at module top, before
  `import matplotlib.pyplot as plt`.
- **Close every figure** with `plt.close(fig)` in the worker, or memory grows per frame.
- **Zero-pad filenames** (`frame_%06d.png`) so ffmpeg orders them correctly.
- **h5py handles are not picklable.** Either pass each worker the small per-frame slice it
  needs, or have the worker reopen the file by path (better for very large fields).
- **Long runs.** The shipped config saves thousands of rows. Support `--stride` /
  `--max-frames` to subsample; do not subsample by default.

## Commands

Run everything in the project conda environment:

```bash
conda run -n HeatEqnImpJAX ruff format analysis/make_<subject>_movie.py
conda run -n HeatEqnImpJAX ruff check --fix analysis/make_<subject>_movie.py
conda run -n HeatEqnImpJAX python analysis/make_<subject>_movie.py
```
