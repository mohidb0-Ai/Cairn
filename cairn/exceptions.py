"""Cairn's error types.

Author: Mohid Bin Farooq
"""

from __future__ import annotations


class CairnError(Exception):
    """Base class for every error Cairn raises on purpose."""


class CyclicDependencyError(CairnError):
    """Raised when a pipeline's steps form a dependency cycle."""

    def __init__(self, cycle: list[str]):
        super().__init__(f"Cyclic dependency detected: {' -> '.join(cycle)}")
        self.cycle = cycle


class StepFailedError(CairnError):
    """Raised when a step exhausts its retries without succeeding."""

    def __init__(self, step_name: str, attempts: int, cause: Exception):
        super().__init__(f"Step '{step_name}' failed after {attempts} attempt(s): {cause}")
        self.step_name = step_name
        self.attempts = attempts
        self.cause = cause


class UnknownStepError(CairnError):
    """Raised when a step declares a dependency that was never added."""

    def __init__(self, step_name: str, missing: str):
        super().__init__(f"Step '{step_name}' depends on unknown step '{missing}'")


class ModelNotFoundError(CairnError):
    """Raised when the registry has no matching model / version."""

    def __init__(self, name: str, version: int | None = None):
        target = f"{name} (version {version})" if version else name
        super().__init__(f"No model registered as '{target}'")


class RunNotFoundError(CairnError):
    """Raised when a tracked run id doesn't exist in the local store."""

    def __init__(self, run_id: str):
        super().__init__(f"No tracked run with id '{run_id}'")
