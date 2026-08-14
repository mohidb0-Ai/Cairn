"""Where Cairn keeps its state: run history and registered models, all as
plain files under `.cairn/` in the current working directory. No database
to stand up — clone a repo, run a pipeline, and the history travels with it.

Author: Mohid Bin Farooq
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CAIRN_DIR = Path(".cairn")
RUNS_DIR = CAIRN_DIR / "runs"
MODELS_DIR = CAIRN_DIR / "models"


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
