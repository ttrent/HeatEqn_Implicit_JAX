#!/usr/bin/env bash
# Launch the full convergence sweep and render its figures on a remote box.
#
# Runs the serial sweep followed by the plotting script inside a detached tmux
# session (with nohup, so it survives disconnects), logging everything to
# output/convergence/sweep.log. Both commands run in the HeatEqnImpJAX conda
# environment. Invoke from anywhere:  bash analysis/run_convergence_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.."  # repo root

SESSION="convergence_sweep"
LOG_FILE="output/convergence/sweep.log"
mkdir -p "$(dirname "${LOG_FILE}")"

# Single-quoted so nothing expands here; runs the sweep, then the plots.
CMD='conda run -n HeatEqnImpJAX python analysis/convergence_sweep.py --study all \
  && conda run -n HeatEqnImpJAX python analysis/plot_convergence.py'

# -l: login shell so the tmux session finds conda on PATH.
tmux new-session -d -s "${SESSION}" \
  "nohup bash -lc '${CMD}' > '${LOG_FILE}' 2>&1"

echo "Launched convergence sweep in tmux session '${SESSION}'."
echo "  Follow progress: tail -f ${LOG_FILE}"
echo "  Attach session:  tmux attach -t ${SESSION}"
echo "  List sessions:   tmux ls"
