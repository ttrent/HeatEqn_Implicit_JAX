# HeatEqn Implicit JAX — Agent Instructions

## Environment

- This project runs in the conda environment **`HeatEqnImpJAX`** (see `environment.yml`).
- Run **every** terminal command — tests, Ruff, scripts, Python — inside that
  environment. Prefix commands with `conda run -n HeatEqnImpJAX`, e.g.:
  - `conda run -n HeatEqnImpJAX python -m pytest`
  - `conda run -n HeatEqnImpJAX ruff format src/`
  - `conda run -n HeatEqnImpJAX python src/main.py`
- Alternatively, if the terminal session persists, run `conda activate HeatEqnImpJAX`
  once before other commands.
- Never use the base environment or a bare `python` / `pip` for this project.
