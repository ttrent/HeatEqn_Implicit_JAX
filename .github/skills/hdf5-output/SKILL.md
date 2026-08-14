---
name: hdf5-output
description: 'Use when reading, writing, analyzing, post-processing, or visualizing this project HDF5 simulation output (output/heat_solution.h5) from the implicit heat-equation solver. Provides the complete file schema: root metadata attributes (nx, dx, dt, config_yaml, git_*, created_utc), the time-indexed datasets u/t/x/step, the diagnostics group (iterations, converged, residual_norm), and the run_stats group, with exact shapes, dtypes, physical and numerical meaning, time-axis alignment, the sentinel initial-condition row, and save_rate/buffer_rows streaming. Use for any task touching the .h5 file, src/io_utils/output.py, or analysis scripts that load results.'
---

# HDF5 Simulation Output

Complete map of the HDF5 file written by this implicit 1D heat-equation solver, so any
task that reads, writes, analyzes, or plots results uses the correct field names, shapes,
dtypes, and meanings. For the exhaustive field-by-field reference — every `run_stats`
counter, the governing PDE, and the derivations — read
[references/schema.md](references/schema.md).

## File

- Path: `output/heat_solution.h5` (directory and filename come from the `output` block of
  the config).
- Written by `src/io_utils/output.py`:
  - `create_output_file(params, config_path)` — writes root attributes and the grid, then
    creates the empty, growable datasets. Call once at startup.
  - `append_snapshots(path, states, steps, diagnostics=None)` — appends one row or a block
    of rows; `diagnostics=None` writes the initial-condition sentinel.
  - `write_run_stats(path, ...)` — writes the final `run_stats` group. Call once at the end.
- Canonical reader: `analysis/analytic_solution.py` (`read_analytic_inputs`), which reads
  `x`, `t`, and parses the `config_yaml` attribute.

## Schema

`n_rows` = number of saved snapshots = `1 + time.n_steps // time.save_rate.steps`
(row 0 is the initial condition); `nx` = `grid.size`.

```text
output/heat_solution.h5
├── (root attrs) config_yaml, created_utc, git_commit, git_branch,
│                git_describe, git_dirty, nx, dx, dt
├── x        (nx,)         float64   spatial grid, x0 + dx*arange(nx)
├── u        (n_rows, nx)  float64   solution field per saved snapshot
├── t        (n_rows,)     float64   absolute simulation time per row
├── step     (n_rows,)     int64     global implicit-step index per row
├── diagnostics/
│   ├── iterations     (n_rows,)  int64    Newton iterations (last step of interval)
│   ├── converged      (n_rows,)  bool     converged flag (last step of interval)
│   └── residual_norm  (n_rows,)  float64  final RMS residual (last step of interval)
└── run_stats/ (attrs only) elapsed_s, compile_s, run_s, percent_cpu, finished_utc,
               user_time_s, system_time_s, max_rss_kb, minor_page_faults,
               major_page_faults, voluntary_ctx_switches, involuntary_ctx_switches,
               fs_inputs, fs_outputs, swaps, signals, page_size_bytes, summary
```

## Must-know facts

- **Time alignment.** `u`, `t`, `step`, and every `diagnostics/*` dataset share the same
  leading (time) axis. Row `i` is one snapshot; `t[i] = time.start + step[i] * dt`.
- **Row 0 is the initial condition** (no implicit solve). Its diagnostics are sentinels:
  `iterations = 0`, `converged = True`, `residual_norm = NaN`.
- **Diagnostics are per save interval, last step only.** Each row after row 0 stores the
  diagnostics of the *final* implicit step within its save interval, not an aggregate over
  the `save_rate.steps` steps in that interval. The whole-interval "all steps converged"
  flag is computed at runtime for a stderr warning and is **not** stored in the file.
- **Metadata lives only in `config_yaml`.** The initial condition (`offset`, sine `modes`)
  and grid bounds `x0`/`xn` are not separate datasets; parse the `config_yaml` root
  attribute to recover them (see the reader below).
- **Streaming layout.** Time datasets grow along the leading axis, are chunked at 64 rows,
  and are flushed in blocks of `output.buffer_rows` snapshots — the full history never has
  to live in memory. `x` is fixed-size; `run_stats` values are group attributes.

## Read

```python
import h5py

with h5py.File("output/heat_solution.h5", "r") as f:
    x = f["x"][:]                       # (nx,)
    u = f["u"][:]                       # (n_rows, nx)
    t = f["t"][:]                       # (n_rows,)
    step = f["step"][:]                 # (n_rows,)
    iters = f["diagnostics/iterations"][:]
    converged = f["diagnostics/converged"][:]
    residual = f["diagnostics/residual_norm"][:]
    nx, dx, dt = f.attrs["nx"], f.attrs["dx"], f.attrs["dt"]
    config_yaml = f.attrs["config_yaml"]   # parse with yaml.safe_load for modes/bounds
```

Reuse `analysis/analytic_solution.py:read_analytic_inputs` when you need the grid bounds,
offset, and sine modes reconstructed from `config_yaml`.

## Write / extend

Do not hand-roll the file layout. Use the writers in `src/io_utils/output.py`
(`create_output_file`, `append_snapshots`, `write_run_stats`) so attributes, dtypes,
chunking, and the sentinel initial-condition row stay consistent. Call `append_snapshots`
outside any `jit` / `lax.scan` region — it reads concrete host values. If you add or rename
a dataset/attribute, update both `src/io_utils/output.py` and `references/schema.md`.
