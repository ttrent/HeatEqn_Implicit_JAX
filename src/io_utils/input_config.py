"""Load simulation parameters from a YAML config into a ``SimParams`` tree."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GridParams:
    """Spatial grid definition.

    Attributes
    ----------
    size : int
        Number of grid points.
    x0 : float
        Left spatial boundary.
    xn : float
        Right spatial boundary.
    """

    size: int
    x0: float
    xn: float


@dataclass(frozen=True)
class SaveRate:
    """Cadence at which the solution is recorded.

    Attributes
    ----------
    unit : str
        Cadence unit, either ``"steps"`` or ``"time"``.
    value : int | float
        Interval between saves: an ``int`` step count when ``unit`` is
        ``"steps"``, or a ``float`` time interval when ``unit`` is ``"time"``.
    """

    unit: str
    value: int | float


@dataclass(frozen=True)
class TimeParams:
    """Time integration window and output cadence.

    Attributes
    ----------
    start : float
        Initial time.
    end : float
        Final time.
    step_size : float
        Time step ``dt``.
    save_rate : SaveRate
        How often the solution is recorded.
    """

    start: float
    end: float
    step_size: float
    save_rate: SaveRate


@dataclass(frozen=True)
class MethodParams:
    """Numerical scheme selection.

    Attributes
    ----------
    integrator : str
        Time-stepping scheme name (e.g. ``"backward_euler"``).
    solver : str
        Implicit-solve routine name (e.g. ``"newton_method"``).
    """

    integrator: str
    solver: str


@dataclass(frozen=True)
class SineMode:
    """A single sine-wave component of the initial state.

    Attributes
    ----------
    wavenumber : float
        Spatial wavenumber of the mode.
    amplitude : float
        Amplitude of the mode.
    """

    wavenumber: float
    amplitude: float


@dataclass(frozen=True)
class InitialState:
    """Initial condition built as a superposition of sine-wave modes.

    Attributes
    ----------
    offset : float
        Global offset applied to the initial condition.
    modes : tuple[SineMode, ...]
        Sine-wave components summed to form the initial field.
    """

    offset: float
    modes: tuple[SineMode, ...]


@dataclass(frozen=True)
class OutputParams:
    """Destination for simulation results.

    Attributes
    ----------
    directory : str
        Output directory path.
    filename : str
        Output file name.
    """

    directory: str
    filename: str


@dataclass(frozen=True)
class SimParams:
    """Full set of simulation parameters grouped by category.

    Attributes
    ----------
    grid : GridParams
        Spatial grid definition.
    time : TimeParams
        Time integration window and output cadence.
    methods : MethodParams
        Integrator and solver selection.
    initial_state : InitialState
        Initial condition specification.
    output : OutputParams
        Output destination.
    """

    grid: GridParams
    time: TimeParams
    methods: MethodParams
    initial_state: InitialState
    output: OutputParams


def load_config(path: str | Path) -> SimParams:
    """Read a YAML configuration file into a ``SimParams`` instance.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    SimParams
        Parsed simulation parameters grouped into nested dataclasses.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    time_config = config["time"]
    time = TimeParams(
        start=time_config["start"],
        end=time_config["end"],
        step_size=time_config["step_size"],
        save_rate=SaveRate(**time_config["save_rate"]),
    )

    initial_config = config["initial_state"]
    modes = tuple(SineMode(**mode) for mode in initial_config["modes"])
    initial_state = InitialState(offset=initial_config["offset"], modes=modes)

    return SimParams(
        grid=GridParams(**config["grid"]),
        time=time,
        methods=MethodParams(**config["methods"]),
        initial_state=initial_state,
        output=OutputParams(**config["output"]),
    )
