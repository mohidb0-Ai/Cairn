# Cairn

Weave your training pipeline, experiment tracking, and model registry
together — without standing up Airflow, MLflow, and a database just to
run a `RandomForestClassifier`.

Built by [Mohid Bin Farooq](https://github.com/mohidbinfarooq).

## What it does

Most ML projects end up rebuilding the same three things by hand: a way to
chain training steps together, a way to remember what parameters produced
which metrics, and a way to keep track of which model version is actually
the good one. Cairn is those three things, as one small Python library with
**zero required dependencies**.

- **Pipelines** — declare steps as plain functions, wire up their
  dependencies, and Cairn figures out the execution order, runs independent
  steps in parallel, and retries the ones that fail.
- **Tracking** — log params and metrics inside a `with track.run(...)`
  block; every run is saved to disk so you can compare them later.
- **Registry** — every model you save gets a version number and its
  metrics/params saved alongside it. Load the latest, or pin a version.

## Install

```bash
pip install -e .
```

## Quickstart

```python
from cairn import step, Pipeline, track, registry
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


@step
def load_data():
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.2)


@step
def train(load_data, n_estimators=100):
    x_train, x_test, y_train, y_test = load_data
    with track.run("iris-rf") as run:
        run.log_param("n_estimators", n_estimators)
        model = RandomForestClassifier(n_estimators=n_estimators).fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))
        run.log_metric("accuracy", accuracy)
    return model


@step
def register(train):
    registry.save(train, name="iris-classifier")


pipeline = Pipeline("iris-training")
pipeline.add(load_data)
pipeline.add(train, depends_on=[load_data])
pipeline.add(register, depends_on=[train])

pipeline.run(n_estimators=150)
```

A step's parameters are matched by name — if you write `def train(load_data)`,
it automatically receives whatever `load_data` returned. Anything you pass
into `pipeline.run(**kwargs)` is available the same way.

A full runnable version of this is in [`examples/train_iris.py`](examples/train_iris.py).

## Inspecting past runs

```bash
cairn runs                    # pipeline run history
cairn experiments             # tracked experiment runs
cairn models                  # registered models and their versions
cairn show <run_id>           # full detail for one run
```

## Retries

```python
@step(retries=3, retry_delay=2.0)
def download_dataset(): ...
```

Flaky steps — a dataset download, a call to an external service — retry
automatically before the pipeline gives up on them.

## How it stores things

Everything lands under `.cairn/` in your working directory: `runs/` for
pipeline and experiment history, `models/<name>/<version>/` for registered
models and their metadata. No server, no database — it's just files, so it
travels with your repo (add `.cairn/` to `.gitignore` if you don't want
training artifacts committed).

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

MIT — see [LICENSE](LICENSE).
