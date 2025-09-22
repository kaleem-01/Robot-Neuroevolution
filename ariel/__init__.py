"""Ariel."""

# Standard library
from pathlib import Path

# Pretty errors and console output
from rich.traceback import install

# Produce data output folder
CWD = Path.cwd()
DATA = Path(CWD / "__data__")
DATA.mkdir(exist_ok=True)

# Colorful errors
install()
