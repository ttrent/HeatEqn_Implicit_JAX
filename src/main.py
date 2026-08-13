"""Run the implicit heat-equation simulation end to end from a YAML config."""

import sys
from time import perf_counter

import numpy as np

import jax
import jax.numpy as jnp

from initialize import sine_waves
from integrators import INTEGRATORS, make_buffer_runner
from io_utils import append_snapshots, create_output_file, load_config, parse_args
from solvers import SOLVERS

jax.config.update("jax_enable_x64", True)


def main() -> None:
    """Run the simulation end to end from a YAML config named on the command line.

    Reads the config, creates the output file, and writes the initial state,
    then advances the implicit time stepping in a nested ``lax.scan``: the inner
    scan takes ``save_rate.steps`` implicit steps per saved snapshot, and each
    jitted ``run_buffer`` call collects up to ``output.buffer_rows`` snapshots
    before they are streamed to the HDF5 file, so the whole history never has to
    reside in memory. A trailing partial buffer is run as a final shorter scan.
    Prints the compile and run timings and warns if any step failed to converge.
    """
    args = parse_args()

    params = load_config(args.config)
    out_path = create_output_file(params, args.config)

    state = sine_waves(params)
    append_snapshots(out_path, state, steps=0)

    residual_fn = INTEGRATORS[params.methods.integrator]
    solver = SOLVERS[params.methods.solver]
    run_buffer = make_buffer_runner(residual_fn, solver, params)

    save_steps = params.time.save_rate.steps
    n_saves = params.time.n_steps // save_steps
    buffer_rows = params.output.buffer_rows
    n_buffers, remainder = divmod(n_saves, buffer_rows)
    buffer_lengths = [buffer_rows] * n_buffers
    if remainder:
        buffer_lengths.append(remainder)

    # Warm the XLA cache for each distinct buffer length before timing the run.
    compile_start = perf_counter()
    for length in set(buffer_lengths):
        run_buffer.lower(state, length=length).compile()
    compile_time = perf_counter() - compile_start

    run_start = perf_counter()
    failed = 0
    save_offset = 0
    for length in buffer_lengths:
        state, (states, diagnostics, interval_converged) = run_buffer(
            state, length=length
        )
        jax.block_until_ready((states, diagnostics, interval_converged))
        steps = (save_offset + np.arange(1, length + 1)) * save_steps
        append_snapshots(out_path, states, steps, diagnostics)
        failed += int(jnp.sum(~interval_converged))
        save_offset += length
    run_time = perf_counter() - run_start

    print(f"Compiled in {compile_time:.3f} s")
    print(f"Ran in {run_time:.3f} s")
    if failed:
        print(
            f"WARNING: {failed} save interval(s) contained a non-converged step.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
