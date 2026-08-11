---
description: "Use when writing or editing Python/JAX code in this scientific computing project (implicit heat-equation solver). Covers naming, 88-column Ruff formatting, type hints (jax.Array / ArrayLike), NumPy-style docstrings, import order, JAX functional idioms (jit / vmap / lax.scan / cond), float64 (x64) precision, PRNG keys, PyTrees, and the integrators / io / rhs / solvers module layout."
applyTo: "**/*.py"
---

# JAX Scientific Python Style Guide

Conventions for all Python code in this project — an implicit heat-equation
solver built on JAX (Python 3.12, GPU via `jax[cuda13]`, with NumPy/SciPy/
Matplotlib). Prefer clarity and numerical correctness over cleverness.

## 1. Formatting & naming

- Format and lint with **Ruff**; keep lines within **88 columns**.
- After creating or editing any `.py` file, run `ruff format <file>` then
  `ruff check --fix <file>` and resolve any remaining warnings.
- `snake_case` for functions and variables, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for module-level constants, leading `_` for private helpers.
- Name arrays for the physics, not the container: `u`, `u_new`, `laplacian_u`,
  `dx`, `dt` — never `arr`, `tmp`, `data`.
- Group imports in this order, one blank line between groups:
  1. Standard library
  2. Third-party scientific (`numpy as np`, `scipy`)
  3. JAX (`import jax`, `import jax.numpy as jnp`, `from jax import lax`)
  4. Local packages (`from rhs import laplacian_1d`)
- Use standard aliases only (`np`, `jnp`, `lax`). Never `from jax.numpy import *`.

## 2. Type hints

- Annotate every public function's parameters and return type.
- Use `jax.Array` for concrete array results; use `jax.typing.ArrayLike` for
  inputs that may also be Python/NumPy scalars.
- Type callables explicitly, e.g. `Callable[[jax.Array], jax.Array]`.
- Prefer Python 3.12 builtin generics: `list[float]`, `tuple[int, int]`,
  `float | None`.
- No mutable default arguments; pass configuration explicitly.

```python
from typing import Callable
import jax

def implicit_step(
    u: jax.typing.ArrayLike,
    dt: float,
    rhs: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    ...
```

## 3. Docstrings (NumPy style)

- Every public function and class gets a NumPy-style docstring: a one-line
  summary, then `Parameters`, `Returns`, and `Notes` when useful.
- Document each array's **shape and dtype** and name the numerical method.

```python
def laplacian_1d(u: jax.Array, dx: float) -> jax.Array:
    """Second-order central-difference Laplacian with periodic BCs.

    Parameters
    ----------
    u : jax.Array
        Field values, shape ``(nx,)``, dtype float64.
    dx : float
        Uniform grid spacing.

    Returns
    -------
    jax.Array
        Laplacian of ``u``, same shape and dtype as ``u``.
    """
```

## 4. JAX functional idioms

- Write **pure functions**: outputs depend only on inputs — no in-place
  mutation, global state, or I/O inside compute kernels.
- Arrays are immutable — update with `u = u.at[0].set(0.0)`, never `u[0] = 0.0`.
- Decorate hot paths with `@jax.jit`; mark shapes/flags static via
  `functools.partial(jax.jit, static_argnames=("n",))`.
- Batch with `jax.vmap` instead of Python loops over array rows.
- Use structured control flow on traced values: `lax.scan` for time stepping,
  `lax.cond` for branches, `lax.fori_loop` / `lax.while_loop` for iteration.
  Plain Python `for` / `if` only on static (non-traced) values.
- Never call `.item()`, `float()`, `bool()`, or `print` on a traced array inside
  a jitted function.
- Keep `rhs`, `integrators`, and `solvers` as functions (or PyTree-carrying
  callables), not stateful objects.

```python
from jax import lax

def integrate(u0: jax.Array, dt: float, n_steps: int) -> jax.Array:
    def body(u, _):
        return implicit_step(u, dt, rhs), None

    u_final, _ = lax.scan(body, u0, xs=None, length=n_steps)
    return u_final
```

## 5. Numerical & precision practices

- Enable double precision **once**, before any array is created (top of
  `main.py`):

  ```python
  import jax
  jax.config.update("jax_enable_x64", True)
  ```

- Be explicit with dtypes (`jnp.zeros(nx, dtype=jnp.float64)`) to avoid silent
  float32 downcasting on GPU.
- Randomness: thread a typed key explicitly and split before use —
  `key, sub = jax.random.split(key)`. Never reuse a key for two draws.
- Represent evolving solver state as a **PyTree** (a `NamedTuple`, or a dataclass
  registered with `jax.tree_util.register_pytree_node_class`) so it flows through
  `jit` and `scan` cleanly.

```python
from typing import NamedTuple
import jax

class State(NamedTuple):
    u: jax.Array   # solution field, shape (nx,)
    t: float       # current time
```

## 6. Module layout

Keep each package focused and its functions pure so they compose:

- `rhs/` — PDE spatial operators: Laplacian, boundary conditions, source terms;
  returns `du/dt` or the operator applied to a field.
- `integrators/` — time-stepping schemes (e.g. backward Euler); consume an `rhs`
  and a `solver` to advance state one step or over a `lax.scan`.
- `solvers/` — linear/nonlinear solves for implicit steps (`jnp.linalg.solve`,
  `jax.scipy.sparse.linalg.cg`, Newton); pure functions of operator + RHS.
- `io/` — load parameters/initial conditions, save results, plotting
  (Matplotlib). The **only** place side effects (file/stdout) belong.
- `main.py` — sets x64 config, wires the modules together, runs the simulation.
