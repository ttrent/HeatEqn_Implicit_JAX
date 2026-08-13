"""Command-line interface for the simulation entry point."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments; ``config`` holds the path to the YAML config file.
    """
    parser = argparse.ArgumentParser(
        description="Run the implicit heat-equation simulation.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()
