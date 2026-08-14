"""`cairn` command-line entry point: inspect past pipeline runs and the
model registry without writing a script for it.

    cairn runs
    cairn models
    cairn show <run_id>

Author: Mohid Bin Farooq
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cairn.registry import registry
from cairn.storage import RUNS_DIR
from cairn.tracking import track


def _cmd_runs(args: argparse.Namespace) -> None:
    runs_dir = Path(RUNS_DIR)
    if not runs_dir.exists():
        print("No runs recorded yet.")
        return
    for path in sorted(runs_dir.glob("*.json")):
        data = json.loads(path.read_text())
        print(
            f"{data['run_id']:<28} {data['pipeline']:<20} {data['status']:<8} {data['duration_seconds']:.2f}s"
        )


def _cmd_experiments(args: argparse.Namespace) -> None:
    runs = track.list(experiment=args.experiment)
    if not runs:
        print("No tracked experiment runs yet.")
        return
    for run in runs:
        metrics_summary = ", ".join(f"{k}={v[-1]['value']}" for k, v in run["metrics"].items())
        print(f"{run['run_id']:<16} {run['experiment']:<20} {metrics_summary}")


def _cmd_models(args: argparse.Namespace) -> None:
    names = registry.list()
    if not names:
        print("No models registered yet.")
        return
    for name in names:
        versions = registry.versions(name)
        print(f"{name:<25} versions: {versions}")


def _cmd_show(args: argparse.Namespace) -> None:
    path = Path(RUNS_DIR) / f"{args.run_id}.json"
    if path.exists():
        print(json.dumps(json.loads(path.read_text()), indent=2))
        return
    print(json.dumps(track.get(args.run_id), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cairn", description="Inspect Cairn pipeline runs and registered models."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("runs", help="list pipeline runs").set_defaults(func=_cmd_runs)

    exp = sub.add_parser("experiments", help="list tracked experiment runs")
    exp.add_argument("--experiment", default=None, help="filter by experiment name")
    exp.set_defaults(func=_cmd_experiments)

    sub.add_parser("models", help="list registered models").set_defaults(func=_cmd_models)

    show = sub.add_parser("show", help="show full detail for a run id")
    show.add_argument("run_id")
    show.set_defaults(func=_cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
