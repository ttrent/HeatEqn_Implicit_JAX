# HDF5 Output Schema — Full Reference

Exhaustive reference for `output/heat_solution.h5`, the single file written by this
implicit 1D heat-equation solver. The authoritative source of truth is
`src/io_utils/output.py`; if that file changes, update this document to match.

Symbols used below:

- `nx` — number of spatial grid points (`grid.size`).
- `n_rows` — number of saved snapshots along the leading (time) axis, including the
  initial condition: `n_rows = 1 + time.n_steps // time.save_rate.steps`.

---

## 1. Governing problem

The solver integrates the 1D heat equation on a periodic domain `[x0, xn]`:

```
du/dt = alpha * d2u/dx2,   alpha = 1
```

- **Spatial discretization** (`src/rhs/laplacian.py`): second-order central difference with
  periodic wraparound via `jnp.roll`, `(u[i-1] - 2*u[i] + u[i+1]) / dx**2`.
- **Right-hand side** (`src/rhs/rhs.py`): currently `du/dt = laplacian(u)` (no source or
  advection term), so the effective diffusivity `alpha` is 1.
- **Time integration** (`src/integrators/`): implicit — `backward_euler` or
  `implicit_midpoint`, selected by `methods.integrator`. Each step drives an
  implicit-scheme residual `R(u_new) = 0` to zero.
- **Nonlinear solve** (`src/solvers/newton_method.py`): Jacobian-free Newton–Krylov,
  selected by `methods.solver`.

The grid is `x = x0 + dx * arange(nx)` with `dx = (xn - x0) / nx`.

---

## 2. Root attributes

Written by `create_output_file`. Stored as attributes on the HDF5 root.

| Attribute | Dtype | Meaning |
|---|---|---|
| `config_yaml` | str | Verbatim contents of the YAML config used for the run. The only place the initial condition (`offset`, `modes`) and grid bounds `x0`/`xn` are stored; parse to reconstruct them. |
| `created_utc` | str | ISO-8601 UTC timestamp when the file was created. |
| `git_commit` | str | Full commit hash of the repo at run time, or `"unknown"`. |
| `git_branch` | str | Current branch name, or `"unknown"`. |
| `git_describe` | str | `git describe --tags --always --dirty`, or `"unknown"`. |
| `git_dirty` | bool | `True` if the working tree had uncommitted changes. |
| `nx` | int64 | Number of spatial grid points (`grid.size`). |
| `dx` | float64 | Grid spacing, `(xn - x0) / nx`. |
| `dt` | float64 | Time-step size (`time.step_size`). |

---

## 3. Datasets (root)

Created empty by `create_output_file` and extended by `append_snapshots`. All time-indexed
datasets grow along the leading axis (`maxshape` leading dim `None`) and are chunked at
`_CHUNK_ROWS = 64` rows.

| Dataset | Shape | Dtype | Chunks | Meaning |
|---|---|---|---|---|
| `x` | `(nx,)` | float64 | — | Spatial grid coordinates, `x0 + dx*arange(nx)`. Fixed-size (not resizable). |
| `u` | `(n_rows, nx)` | float64 | `(64, nx)` | Solution field. Row `i` is the field at time `t[i]`; row 0 is the initial condition. |
| `t` | `(n_rows,)` | float64 | `(64,)` | Absolute simulation time of each row. `t[i] = time.start + step[i] * dt`. |
| `step` | `(n_rows,)` | int64 | `(64,)` | Global implicit-step index of each row. Row 0 is `0`; later rows are multiples of `save_rate.steps`. |

---

## 4. Group: `diagnostics`

Per-snapshot solver telemetry, one row per saved snapshot, aligned with `u`/`t`/`step`
along the leading axis. Leaf fields mirror `SolverDiagnostics` in
`src/solvers/_types.py`.

| Dataset | Shape | Dtype | Chunks | Meaning |
|---|---|---|---|---|
| `diagnostics/iterations` | `(n_rows,)` | int64 | `(64,)` | Outer Newton–Krylov iterations taken by the last implicit step of the save interval. |
| `diagnostics/converged` | `(n_rows,)` | bool | `(64,)` | Whether that last step met the stopping tolerance. |
| `diagnostics/residual_norm` | `(n_rows,)` | float64 | `(64,)` | Final RMS residual norm of that last step. |

### Sentinel row (initial condition)

Row 0 involved no solve, so `append_snapshots(..., diagnostics=None)` writes sentinels:

- `iterations = 0`
- `converged = True`
- `residual_norm = NaN`

### Last-step-of-interval semantics

Within one save interval the inner `lax.scan` (`src/integrators/stepping.py`) takes
`save_rate.steps` implicit steps. Only the **final** step's diagnostics are kept per
interval:

```python
last = SolverDiagnostics(
    iterations=diagnostics.iterations[-1],
    converged=diagnostics.converged[-1],
    residual_norm=diagnostics.residual_norm[-1],
)
```

The whole-interval "did every step converge" value (`jnp.all(diagnostics.converged)`) is
returned separately and used only to emit a stderr warning and a failure count in
`src/main.py`; it is **not** written to the file. So `diagnostics/converged[i] == True`
means the last step of interval `i` converged, not necessarily all steps within it.

---

## 5. Group: `run_stats`

Written once by `write_run_stats` at the end of the run. All values are attributes on the
`run_stats` group. Resource counters come from `getrusage(RUSAGE_SELF)`; counters the Linux
kernel does not track are stored as `0`.

| Attribute | Dtype | Meaning |
|---|---|---|
| `elapsed_s` | float64 | Total wall-clock time of the whole program (compile + run). |
| `compile_s` | float64 | Wall-clock time warming the XLA cache (explicit `.compile()`). |
| `run_s` | float64 | Wall-clock time in the timed integration loop. |
| `percent_cpu` | int | `round(100 * (user_time_s + system_time_s) / elapsed_s)`, or `0` if `elapsed_s <= 0`. |
| `finished_utc` | str | ISO-8601 UTC timestamp when the run finished. |
| `user_time_s` | float64 | User CPU seconds (`ru_utime`). |
| `system_time_s` | float64 | System CPU seconds (`ru_stime`). |
| `max_rss_kb` | int | Peak resident set size, in kibibytes (Linux `ru_maxrss` convention). |
| `minor_page_faults` | int | Page reclaims without I/O (`ru_minflt`). |
| `major_page_faults` | int | Page faults requiring I/O (`ru_majflt`). |
| `voluntary_ctx_switches` | int | Voluntary context switches (`ru_nvcsw`). |
| `involuntary_ctx_switches` | int | Involuntary context switches (`ru_nivcsw`). |
| `fs_inputs` | int | Block-input operations (`ru_inblock`). |
| `fs_outputs` | int | Block-output operations (`ru_oublock`). |
| `swaps` | int | Swaps (`ru_nswap`; typically `0`). |
| `signals` | int | Signals delivered (`ru_nsignals`). |
| `page_size_bytes` | int | System page size, `resource.getpagesize()`. |
| `summary` | str | Multi-line `/usr/bin/time -v`-style text report, prefixed with the compile and run timings. |

---

## 6. Data alignment and relationships

- **Shared leading axis.** `u`, `t`, `step`, and all `diagnostics/*` datasets have the same
  length `n_rows` and are indexed together: row `i` is one snapshot.
- **Step ↔ time.** `t[i] = time.start + step[i] * dt`. Row 0 has `step = 0`; subsequent
  rows increase by `save_rate.steps` per saved snapshot, i.e. `step` runs
  `0, s, 2s, ..., n_steps` for `s = save_rate.steps`.
- **Row count.** `n_rows = 1 + time.n_steps // save_rate.steps`, where
  `time.n_steps = round((end - start) / step_size)`. The leading `1` is the initial
  condition. Example (default config): `n_steps = round(2.0/0.001) = 2000`,
  `save_rate.steps = 10`, so `n_rows = 1 + 200 = 201`.
- **Save cadence.** `time.save_rate` records both `steps` and `time`; `save_rate.steps`
  implicit steps elapse between saved snapshots.
- **Buffered streaming.** `src/main.py` runs the integration in buffers of
  `output.buffer_rows` snapshots (default 100). Each buffer is computed on-device by the
  nested `lax.scan` in `src/integrators/stepping.py`, then streamed to disk with
  `append_snapshots`, so the full history never resides in memory at once. A trailing
  partial buffer (`n_rows - 1` not divisible by `buffer_rows`) is run as a shorter final
  scan.

---

## 7. Initial condition and the analytic reference

The initial field (`src/initialize/sine_waves.py`) is a sum of sine modes on the periodic
grid:

```
u(x, 0) = offset + sum_m A_m * sin(2*pi*k_m*(x - x0)/xn)
```

with per-mode wavenumber `k_m` and amplitude `A_m` from `initial_state.modes`, and a
constant `offset`. The closed-form solution (`analysis/analytic_solution.py`) decays each
mode independently:

```
u(x, t) = offset + sum_m A_m * sin(q_m*(x - x0)) * exp(-alpha * q_m**2 * t),
q_m = 2*pi*k_m/xn,   alpha = 1
```

The decay uses the *continuous* eigenvalue `-q_m**2`; a finite-difference simulation decays
each mode slightly differently, so the reference and the numerical field `u` agree exactly
only at `t = 0`.

---

## 8. Consuming the file

Standard read pattern with `h5py` (open read-only, slice the datasets, parse
`config_yaml` for anything not stored as a dataset):

```python
import h5py
import yaml

with h5py.File("output/heat_solution.h5", "r") as f:
    x = f["x"][:]
    u = f["u"][:]
    t = f["t"][:]
    step = f["step"][:]
    iterations = f["diagnostics/iterations"][:]
    converged = f["diagnostics/converged"][:]
    residual_norm = f["diagnostics/residual_norm"][:]
    config = yaml.safe_load(f.attrs["config_yaml"])   # never executes arbitrary code
    run_stats = dict(f["run_stats"].attrs)
```

`analysis/analytic_solution.py` is the canonical consumer:

- `read_analytic_inputs(path)` — reads `x`, `t`, and parses `config_yaml` for grid bounds,
  `offset`, per-mode wavenumbers/amplitudes, and `t_start`.
- `analytic_solution_from_output(path)` — evaluates the analytic field at every saved time
  (`t - t_start`) so it can be compared point-for-point with the stored `u`.

Use `yaml.safe_load` (not `yaml.load`) when parsing `config_yaml`.

---

## 9. Writing / extending the file

Always go through `src/io_utils/output.py`; do not reconstruct the layout by hand.

- `create_output_file(params, config_path) -> Path` — writes root attributes and `x`, then
  creates the empty growable datasets (`u`, `t`, `step`, and the `diagnostics` group). Call
  once, right after loading the config.
- `append_snapshots(path, states, steps, diagnostics=None)` — appends a single snapshot or
  a block (`states.u` shape `(nx,)` or `(n_block, nx)`); each dataset is resized from the
  data. `diagnostics=None` writes the sentinel row. Call outside any `jit` / `lax.scan`
  region because it reads concrete host values.
- `write_run_stats(path, *, elapsed_time, compile_time, run_time)` — creates the
  `run_stats` group. Call once as the final step.

When adding, renaming, or retyping any dataset or attribute, update `src/io_utils/output.py`
and this reference together, and keep the schema tree in `../SKILL.md` in sync.
