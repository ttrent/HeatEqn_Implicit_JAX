"""Load simulation parameters from a YAML config into a ``SimParams`` tree."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

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
    dx : float
        Grid spacing, computed as ``(xn - x0) / size`` (periodic grid).
    """

    size: int
    x0: float
    xn: float
    dx: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute the derived grid spacing."""
        object.__setattr__(self, "dx", (self.xn - self.x0) / self.size)


@dataclass(frozen=True)
class SaveRate:
    """Output cadence expressed in both steps and time.

    Attributes
    ----------
    steps : int
        Number of time steps between saves.
    time : float
        Time interval between saves.
    """

    steps: int
    time: float

    @classmethod
    def from_unit_value(cls, unit: str, value: int | float, dt: float) -> Self:
        """Build a cadence from one unit/value pair and the time step.

        Parameters
        ----------
        unit : str
            Cadence unit of ``value``, either ``"steps"`` or ``"time"``.
        value : int | float
            Save interval expressed in ``unit``.
        dt : float
            Time step used to convert between steps and time.

        Returns
        -------
        SaveRate
            Cadence populated with both ``steps`` and ``time``.
        """
        if unit == "steps":
            steps = int(value)
            return cls(steps=steps, time=steps * dt)
        if unit == "time":
            time = float(value)
            return cls(steps=round(time / dt), time=time)
        raise ValueError(f"unknown save_rate unit: {unit!r}")


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
    n_steps : int
        Number of time steps, computed as ``round((end - start) / step_size)``.
    """

    start: float
    end: float
    step_size: float
    save_rate: SaveRate
    n_steps: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute the derived number of time steps."""
        n_steps = round((self.end - self.start) / self.step_size)
        object.__setattr__(self, "n_steps", n_steps)


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
    save_rate_config = time_config["save_rate"]
    save_rate = SaveRate.from_unit_value(
        save_rate_config["unit"],
        save_rate_config["value"],
        time_config["step_size"],
    )
    time = TimeParams(
        start=time_config["start"],
        end=time_config["end"],
        step_size=time_config["step_size"],
        save_rate=save_rate,
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
