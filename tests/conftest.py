"""Pytest configuration and environment setup."""

import sys
import json
import os
import tempfile
from pathlib import Path

# Ensure 'src' is discoverable when running bare pytest without package installation
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Install artificial inputs before test modules import the engine. Protocol tests
# explicitly pass this file to stdio children; real engineering data is never used.
from synthetic_data import make_dataset

_synthetic_directory = tempfile.TemporaryDirectory(prefix="cablesize-tests-")
_synthetic_path = Path(_synthetic_directory.name) / "synthetic.json"
_synthetic_path.write_text(json.dumps(make_dataset()), encoding="utf-8")
os.environ["CABLESIZE_DATA_FILE"] = str(_synthetic_path)
