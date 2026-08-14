"""A local model registry: version every model you save, keep its metrics
and params alongside it, and load any version back by name.

    registry.save(model, name="fraud-detector", metrics={"auc": 0.94})
    latest = registry.load("fraud-detector")
    v2 = registry.load("fraud-detector", version=2)

Author: Mohid Bin Farooq
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

from cairn.exceptions import ModelNotFoundError
from cairn.logging import get_logger
from cairn.storage import MODELS_DIR, ensure_dirs, read_json, write_json

log = get_logger(__name__)


class _Registry:
    """Module-level entry point: `from cairn import registry`."""

    @staticmethod
    def save(
        model: Any,
        name: str,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Serialize `model` as the next version under `name`. Returns the
        new version number.
        """
        ensure_dirs()
        model_dir = Path(MODELS_DIR) / name
        model_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(int(p.name) for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit())
        version = (existing[-1] + 1) if existing else 1

        version_dir = model_dir / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)

        with open(version_dir / "model.pkl", "wb") as fh:
            pickle.dump(model, fh)

        write_json(
            version_dir / "metadata.json",
            {
                "name": name,
                "version": version,
                "saved_at": time.time(),
                "metrics": metrics or {},
                "params": params or {},
            },
        )
        log.info("registered '%s' version %d", name, version)
        return version

    @staticmethod
    def load(name: str, version: int | None = None) -> Any:
        model_dir = Path(MODELS_DIR) / name
        if not model_dir.exists():
            raise ModelNotFoundError(name, version)

        if version is None:
            versions = sorted(int(p.name) for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit())
            if not versions:
                raise ModelNotFoundError(name, version)
            version = versions[-1]

        version_dir = model_dir / str(version)
        model_path = version_dir / "model.pkl"
        if not model_path.exists():
            raise ModelNotFoundError(name, version)

        with open(model_path, "rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def metadata(name: str, version: int | None = None) -> dict[str, Any]:
        model_dir = Path(MODELS_DIR) / name
        if version is None:
            versions = (
                sorted(int(p.name) for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit())
                if model_dir.exists()
                else []
            )
            if not versions:
                raise ModelNotFoundError(name, version)
            version = versions[-1]
        meta_path = model_dir / str(version) / "metadata.json"
        if not meta_path.exists():
            raise ModelNotFoundError(name, version)
        return read_json(meta_path)

    @staticmethod
    def list() -> list[str]:
        if not Path(MODELS_DIR).exists():
            return []
        return sorted(p.name for p in Path(MODELS_DIR).iterdir() if p.is_dir())

    @staticmethod
    def versions(name: str) -> list[int]:
        model_dir = Path(MODELS_DIR) / name
        if not model_dir.exists():
            return []
        return sorted(int(p.name) for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit())


registry = _Registry()
