"""Make the implementation namespace available to synthetic test subprocesses."""
import os
from pathlib import Path

IMPLEMENTATION = str(Path(__file__).resolve().parent / "remote_edit")
paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
if IMPLEMENTATION not in paths:
    os.environ["PYTHONPATH"] = os.pathsep.join([IMPLEMENTATION] + [p for p in paths if p])
