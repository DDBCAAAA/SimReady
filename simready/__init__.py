"""SimReady: Automated Sim-Ready OpenUSD asset creation from CAD/CAE sources."""

from pathlib import Path

__version__ = "0.1.0"

# Absolute path to the project root (parent of this package directory).
# All default data/output paths are derived from here so the CLI works
# correctly regardless of the working directory it is invoked from.
PROJECT_ROOT = Path(__file__).parent.parent
