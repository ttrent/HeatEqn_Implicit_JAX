# HeatEqn Implicit JAX

An implicit 1-D heat-equation solver built on [JAX](https://github.com/google/jax). It
advances the diffusion equation with A-stable implicit time integrators whose per-step
nonlinear systems are solved by a matrix-free (Jacobian-free) Newton–Krylov method, and
streams self-describing results to HDF5.

## Overview

The code solves the periodic 1-D heat equation

$$\frac{\partial u}{\partial t} = \alpha\, \frac{\partial^2 u}{\partial x^2}, \qquad \alpha = 1,$$

on a uniform periodic grid. Space is discretised with a second-order central-difference
Laplacian ([src/rhs/laplacian.py](src/rhs/laplacian.py)); time is advanced *implicitly*,
so the stiff diffusion operator imposes no explicit stability limit on the step size. The
initial condition is a superposition of sine modes, each an eigenfunction of the
Laplacian, so the exact solution is known in closed form — every mode decays as
$\exp(-\alpha q_m^2 t)$ with $q_m = 2\pi k_m / x_n$ — which drives the accuracy and
convergence checks in [analysis/](analysis/).

Every run is driven by a single YAML config, and every result file embeds the exact
config text and git revision that produced it.

## Features

- Two implicit integrators: **backward Euler** (first order) and **implicit midpoint**
  (second order); both A-stable.
- **Jacobian-free Newton–Krylov** implicit solve: forward-mode `jax.linearize` applies
  the Jacobian without ever assembling it, and matrix-free GMRES solves each Newton
  system.
- **Inexact Newton** with an Eisenstat–Walker forcing term and grid-independent
  (RMS-based) convergence tolerances.
- End-to-end `float64` precision (JAX x64) and JIT-compiled, `lax.scan`-based time
  stepping.
- **Streaming HDF5 output** with buffered writes, so the full time history never has to
  reside in memory.
- Reproducible by construction: each output file stores the full config YAML, git
  commit / branch / dirty state, and run timing plus OS resource usage.
- Analysis suite: closed-form error comparison, spatial and temporal convergence
  studies, and error-movie rendering.

## Requirements & installation

The project targets Python 3.12 and pins its dependencies in
[environment.yml](environment.yml) (NumPy, SciPy, Matplotlib, h5py, PyYAML, Ruff,
ffmpeg, and `jax[cuda13]`). Create the conda environment once:

```bash
conda env create -f environment.yml
conda activate HeatEqnImpJAX
```

The JAX build targets CUDA 13 GPUs but falls back to CPU automatically when no GPU is
present. All commands below assume the `HeatEqnImpJAX` environment is active; equivalently
prefix any command with `conda run -n HeatEqnImpJAX`.

## Usage

Run a simulation by pointing the entry point at a YAML config, from the repository root:

```bash
python src/main.py configs/config.yml
```

`src/main.py` loads the config, writes the initial condition, JIT-compiles the stepper,
advances the implicit time loop while streaming snapshots to disk, and prints the compile
and run timings (warning on stderr if any step failed to converge). Running
`python src/main.py` puts `src/` on the import path automatically, so no `PYTHONPATH`
setup is needed.

### Configuration

The config is parsed into a nested, frozen `SimParams` tree by
[src/io_utils/input_config.py](src/io_utils/input_config.py). The six blocks of
[configs/config.yml](configs/config.yml):

```yaml
grid:                     # spatial domain (periodic); dx = (xn - x0) / size is derived
  size: 100
  x0: 0.0
  xn: 10.0

time:                     # integration window and output cadence
  start: 0.0
  end: 100.0
  step_size: 0.001        # dt; n_steps = round((end - start) / step_size) is derived
  save_rate:
    unit: steps           # "steps" or "time"
    value: 10             # save one snapshot every 10 steps

methods:
  integrator: implicit_midpoint   # or "backward_euler"
  solver: newton_method

solver:                   # Jacobian-free Newton–Krylov controls
  absolute_tolerance: 1.0e-8      # Newton stops at atol + rtol * ||R0||
  relative_tolerance: 1.0e-6
  max_iterations: 10              # cap on outer Newton iterations
  linear_rtol_init: 1.0e-3        # initial Eisenstat–Walker forcing term
  linear_rtol_min: 1.0e-9         # tightest inner GMRES tolerance (also the tangent solve)
  linear_rtol_max: 1.0e-2         # loosest inner GMRES tolerance
  linear_atol: 1.0e-12            # absolute floor for the inner GMRES solve
  gmres_restart: 20               # Krylov subspace size before restart
  gmres_maxiter: null             # restart cycles; null = library default

initial_state:            # u(x, 0) = offset + sum_m A_m sin(2*pi*k_m*(x - x0)/xn)
  offset: 0.0
  modes:
    - wavenumber: 1.0
      amplitude: 1.0
    - wavenumber: 3.0
      amplitude: 0.5

output:
  directory: output
  filename: heat_solution.h5
  buffer_rows: 100        # snapshots buffered in memory before each HDF5 flush
```

| Block | Keys | Meaning |
|---|---|---|
| `grid` | `size`, `x0`, `xn` | Grid points and domain bounds; spacing `dx` is derived (periodic). |
| `time` | `start`, `end`, `step_size`, `save_rate.{unit,value}` | Window, time step `dt`, and how often a snapshot is saved (`unit: steps` or `time`). |
| `methods` | `integrator`, `solver` | Scheme (`backward_euler` \| `implicit_midpoint`) and implicit solver (`newton_method`). |
| `solver` | tolerances, `linear_rtol_*`, `gmres_*` | Newton stopping test, Eisenstat–Walker forcing bounds, and GMRES controls. |
| `initial_state` | `offset`, `modes[].{wavenumber,amplitude}` | Initial condition as a sum of sine modes. |
| `output` | `directory`, `filename`, `buffer_rows` | Result destination and streaming buffer size. |

### Output

Results are written to a single self-describing HDF5 file (default
[output/heat_solution.h5](output/heat_solution.h5)) by
[src/io_utils/output.py](src/io_utils/output.py):

```text
output/heat_solution.h5
├── (root attrs)  config_yaml, created_utc, git_commit, git_branch,
│                 git_describe, git_dirty, nx, dx, dt
├── x        (nx,)         float64   spatial grid
├── u        (n_rows, nx)  float64   solution field per saved snapshot
├── t        (n_rows,)     float64   absolute time per snapshot
├── step     (n_rows,)     int64     global implicit-step index per snapshot
├── diagnostics/
│   ├── iterations     (n_rows,)  int64    Newton iterations (last step of interval)
│   ├── converged      (n_rows,)  bool     convergence flag (last step of interval)
│   └── residual_norm  (n_rows,)  float64  final RMS residual (last step of interval)
└── run_stats/ (attrs)  elapsed_s, compile_s, run_s, timing + OS resource usage
```

Row 0 is the initial condition (no solve; sentinel diagnostics `iterations=0`,
`converged=True`, `residual_norm=NaN`). The embedded `config_yaml` and `git_*` attributes
make each file independently reproducible.

### Analysis

Post-processing scripts live in [analysis/](analysis/) and read the HDF5 output directly:

- **Closed-form comparison** — [analysis/analytic_solution.py](analysis/analytic_solution.py)
  evaluates the exact solution on the run's grid and saved times for point-by-point error
  measurement.
- **Convergence studies** — [analysis/convergence_sweep.py](analysis/convergence_sweep.py)
  runs grid- and step-refinement sweeps (each run isolated in its own subprocess) and
  [analysis/plot_convergence.py](analysis/plot_convergence.py) renders the log-log order
  plots. A detached driver is provided in
  [analysis/run_convergence_sweep.sh](analysis/run_convergence_sweep.sh):

  ```bash
  bash analysis/run_convergence_sweep.sh
  ```

- **Error movie** — [analysis/make_error_movie.py](analysis/make_error_movie.py) renders a
  two-panel state / fractional-error mp4 (parallel frame rendering, H.264 via ffmpeg).

## Project structure

The `src/` layout deliberately separates physics from numerics:

```text
src/
  main.py            entry point: load config -> initialise -> time loop -> stream to HDF5
  state.py           State(u, t) NamedTuple carried through lax.scan
  initialize/        initial conditions (sine_waves)
  rhs/               spatial operators; rhs.py aggregates du/dt (currently laplacian_1d)
  integrators/       implicit-scheme residuals + nested-scan time-stepping driver
  solvers/           Jacobian-free Newton–Krylov solve, convergence norms/thresholds
  io_utils/          YAML config parsing, CLI, HDF5 output
```

Two design choices are worth knowing:

- **Physics is isolated in `rhs`.** Integrators depend only on the aggregate `rhs()`
  ([src/rhs/rhs.py](src/rhs/rhs.py)), so new terms (advection, sources, …) are added in
  one place. Each integrator expresses its scheme purely as a residual $R(u_{\text{new}})=0$
  ([src/integrators/backward_euler.py](src/integrators/backward_euler.py),
  [src/integrators/implicit_midpoint.py](src/integrators/implicit_midpoint.py)),
  decoupled from how the root is found.
- **Streaming time loop.** [src/integrators/stepping.py](src/integrators/stepping.py)
  builds a jitted, nested `lax.scan`: an inner scan takes `save_rate.steps` implicit steps
  per snapshot, and an outer scan stacks up to `buffer_rows` snapshots before they are
  flushed to disk.

## Numerical methods

Each implicit step requires solving the nonlinear system $R(u_{\text{new}}) = 0$ defined
by the integrator. This is done by an inexact **Jacobian-free Newton–Krylov** iteration
([src/solvers/newton_method.py](src/solvers/newton_method.py)):

- Every Newton step linearises the residual with forward-mode `jax.linearize` (no
  assembled Jacobian) and solves $J\,\delta = -R$ with matrix-free GMRES.
- The inner GMRES tolerance follows an **Eisenstat–Walker** forcing term, so early Newton
  steps are solved loosely and later ones tightly, bounded by `linear_rtol_min/max`.
- Convergence uses an RMS residual norm ([src/solvers/convergence.py](src/solvers/convergence.py)),
  so tolerances are independent of grid size.

The observed accuracy matches theory: backward Euler is first-order in time, implicit
midpoint is second-order, and both are second-order in space — as confirmed by the
convergence sweep.

### Differentiability (important caveat)

`differentiable_solve` ([src/solvers/newton_method.py](src/solvers/newton_method.py))
wraps the raw Newton–Krylov iteration in `jax.lax.custom_root`, reattaching derivatives
through the implicit function theorem rather than differentiating the data-dependent
`while_loop`. Verified behaviour (x64, CPU):

- ✅ **Forward-mode AD works** — `jvp` and `jacfwd` match the true Jacobian.
- ❌ **Reverse-mode AD is not supported** — `grad` / `jacrev` raise
  `NotImplementedError` from `jax.scipy.sparse.linalg.gmres`, because the matrix-free
  GMRES used as the `custom_root` tangent solve wraps `lax.custom_linear_solve`, which is
  **not reverse-mode transposable**.

Use forward-mode (`jacfwd`) for sensitivities. If reverse-mode is required, replace the
tangent solve with a dense `jnp.linalg.solve` of the small assembled Jacobian (the
idiomatic `custom_root` pattern), which enables both modes at the cost of forming that
Jacobian on the differentiated path only.

## Reproducibility

No extra bookkeeping is needed: every output file records the full config YAML, the git
commit / branch / describe / dirty state, a UTC timestamp, and a `run_stats` group with
wall-clock timing and OS resource usage. A result file therefore fully identifies the code
and configuration that produced it.

## Development

Code style is enforced by Ruff (88-column lines, NumPy-style docstrings, isort import
grouping) per [ruff.toml](ruff.toml) and
[.github/instructions/jax-style.instructions.md](.github/instructions/jax-style.instructions.md):

```bash
ruff format src/ analysis/
ruff check src/ analysis/
```
