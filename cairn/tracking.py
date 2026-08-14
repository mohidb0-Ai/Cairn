"""Experiment tracking: log params and metrics as you train, Cairn keeps a
timestamped record on disk so you can compare runs later.

    with track.run("baseline") as run:
        run.log_param("lr", 0.001)
        for epoch, loss in enumerate(train(...)):
            run.log_metric("loss", loss, step=epoch)

Author: Mohid Bin Farooq
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cairn.exceptions import RunNotFoundError
from cairn.logging import get_logger
from cairn.storage import RUNS_DIR, read_json, write_json

log = get_logger(__name__)


@dataclass
class Run:
    """A single tracked experiment run."""

    experiment: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def log_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metrics.setdefault(key, []).append({"value": value, "step": step, "at": time.time()})

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(str(path))

    def _path(self) -> Path:
        return Path(RUNS_DIR) / "experiments" / f"{self.run_id}.json"

    def save(self) -> None:
        write_json(
            self._path(),
            {
                "run_id": self.run_id,
                "experiment": self.experiment,
                "started_at": self.started_at,
                "params": self.params,
                "metrics": self.metrics,
                "artifacts": self.artifacts,
            },
        )


class _Tracker:
    """Module-level entry point: `from cairn import track`."""

    @contextmanager
    def run(self, experiment: str) -> Iterator[Run]:
        run_obj = Run(experiment=experiment)
        log.info("tracking run '%s' for experiment '%s'", run_obj.run_id, experiment)
        try:
            yield run_obj
        finally:
            run_obj.save()
            log.info(
                "run '%s' saved (%d param(s), %d metric(s))",
                run_obj.run_id,
                len(run_obj.params),
                len(run_obj.metrics),
            )

    @staticmethod
    def get(run_id: str) -> dict[str, Any]:
        path = Path(RUNS_DIR) / "experiments" / f"{run_id}.json"
        if not path.exists():
            raise RunNotFoundError(run_id)
        return read_json(path)

    @staticmethod
    def list(experiment: str | None = None) -> list[dict[str, Any]]:
        exp_dir = Path(RUNS_DIR) / "experiments"
        if not exp_dir.exists():
            return []
        runs = [read_json(p) for p in sorted(exp_dir.glob("*.json"))]
        if experiment:
            runs = [r for r in runs if r["experiment"] == experiment]
        return runs


track = _Tracker()
