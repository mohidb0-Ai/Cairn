"""The orchestration engine: build a DAG of training/eval steps, resolve
the right execution order, run independent steps in parallel, retry the
ones that fail, and pass each step's output into whatever depends on it.

    @step
    def load_data():
        return pd.read_csv("data.csv")

    @step
    def train(load_data):          # <- receives load_data's return value
        return fit_model(load_data)

    pipeline = Pipeline("training")
    pipeline.add(load_data)
    pipeline.add(train, depends_on=[load_data])
    results = pipeline.run()

Author: Mohid Bin Farooq
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cairn.exceptions import CyclicDependencyError, StepFailedError, UnknownStepError
from cairn.logging import get_logger
from cairn.storage import RUNS_DIR, write_json

log = get_logger(__name__)


def step(func: Callable | None = None, *, retries: int = 0, retry_delay: float = 1.0) -> Callable:
    """Decorator that marks a function as a pipeline step.

    Usable bare (`@step`) or with retry options (`@step(retries=3)`).
    """

    def wrap(fn: Callable) -> Step:
        return Step(fn, retries=retries, retry_delay=retry_delay)

    return wrap(func) if func is not None else wrap


@dataclass
class Step:
    """A single unit of work in a pipeline."""

    func: Callable
    retries: int = 0
    retry_delay: float = 1.0
    depends_on: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.func.__name__

    def run(self, inputs: dict[str, Any]) -> Any:
        """Call the wrapped function with whichever of its parameters we
        have values for, retrying on failure up to `self.retries` times.
        """
        import inspect

        params = inspect.signature(self.func).parameters
        kwargs = {name: inputs[name] for name in params if name in inputs}

        attempt = 0
        while True:
            attempt += 1
            try:
                return self.func(**kwargs)
            except Exception as exc:  # noqa: BLE001
                if attempt > self.retries:
                    raise StepFailedError(self.name, attempt, exc) from exc
                log.warning(
                    "step '%s' failed (attempt %d/%d), retrying: %s",
                    self.name,
                    attempt,
                    self.retries + 1,
                    exc,
                )
                time.sleep(self.retry_delay)


@dataclass
class StepResult:
    status: str  # "success" | "failed"
    duration_seconds: float
    error: str | None = None


class Pipeline:
    """A named, ordered collection of steps and the dependencies between them."""

    def __init__(self, name: str, max_workers: int = 4):
        self.name = name
        self.max_workers = max_workers
        self._steps: dict[str, Step] = {}

    def add(self, target: Step | Callable, depends_on: list[Step | Callable | str] | None = None) -> Step:
        """Register a step with the pipeline.

        `target` can be a bare function or one already wrapped with `@step`.
        `depends_on` accepts steps, their functions, or step names.
        """
        step_obj = target if isinstance(target, Step) else Step(target)
        step_obj.depends_on = [self._name_of(d) for d in (depends_on or [])]
        self._steps[step_obj.name] = step_obj
        return step_obj

    @staticmethod
    def _name_of(target: Step | Callable | str) -> str:
        if isinstance(target, str):
            return target
        if isinstance(target, Step):
            return target.name
        return target.__name__

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, **initial_inputs: Any) -> dict[str, Any]:
        """Execute every step in dependency order, returning a dict of
        each step's name -> return value.
        """
        order = self._topological_batches()
        outputs: dict[str, Any] = dict(initial_inputs)
        step_results: dict[str, StepResult] = {}
        run_id = f"{self.name}-{uuid.uuid4().hex[:8]}"
        started = time.time()

        log.info("starting run '%s' (%d step(s))", run_id, len(self._steps))

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for batch in order:
                futures: dict[str, Future] = {
                    name: pool.submit(self._run_one, self._steps[name], outputs) for name in batch
                }
                for name, future in futures.items():
                    step_started = time.time()
                    try:
                        outputs[name] = future.result()
                        step_results[name] = StepResult("success", time.time() - step_started)
                        log.info("  ✓ %s (%.2fs)", name, step_results[name].duration_seconds)
                    except StepFailedError as exc:
                        step_results[name] = StepResult("failed", time.time() - step_started, str(exc))
                        log.error("  ✗ %s: %s", name, exc)
                        self._persist_run(run_id, step_results, time.time() - started, ok=False)
                        raise

        self._persist_run(run_id, step_results, time.time() - started, ok=True)
        log.info("run '%s' complete in %.2fs", run_id, time.time() - started)
        return outputs

    @staticmethod
    def _run_one(step_obj: Step, outputs: dict[str, Any]) -> Any:
        return step_obj.run(outputs)

    def _persist_run(self, run_id: str, results: dict[str, StepResult], duration: float, ok: bool) -> None:
        write_json(
            Path(RUNS_DIR) / f"{run_id}.json",
            {
                "run_id": run_id,
                "pipeline": self.name,
                "status": "success" if ok else "failed",
                "duration_seconds": duration,
                "steps": {
                    name: {"status": r.status, "duration_seconds": r.duration_seconds, "error": r.error}
                    for name, r in results.items()
                },
            },
        )

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------

    def _topological_batches(self) -> list[list[str]]:
        """Kahn's algorithm, grouped into batches of steps that can run
        in parallel because nothing left to schedule depends on them
        being run sequentially.
        """
        for name, step_obj in self._steps.items():
            for dep in step_obj.depends_on:
                if dep not in self._steps:
                    raise UnknownStepError(name, dep)

        remaining = {name: set(s.depends_on) for name, s in self._steps.items()}
        batches: list[list[str]] = []

        while remaining:
            ready = sorted(name for name, deps in remaining.items() if not deps)
            if not ready:
                raise CyclicDependencyError(list(remaining.keys()))
            batches.append(ready)
            for name in ready:
                remaining.pop(name)
            for deps in remaining.values():
                deps.difference_update(ready)

        return batches
