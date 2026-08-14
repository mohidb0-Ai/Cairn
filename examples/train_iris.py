"""End-to-end example: a small training pipeline for the classic Iris
dataset, using Cairn for orchestration, experiment tracking, and the
model registry.

Run it:
    pip install -e ".[examples]"
    python examples/train_iris.py

Author: Mohid Bin Farooq
"""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cairn import Pipeline, registry, step, track


@step
def load_data():
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.2, random_state=42)


@step
def train(load_data, n_estimators: int = 100):
    x_train, x_test, y_train, y_test = load_data
    with track.run("iris-random-forest") as run:
        run.log_param("n_estimators", n_estimators)

        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(x_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(x_test))
        run.log_metric("accuracy", accuracy)

    return {"model": model, "accuracy": accuracy}


@step
def register(train):
    version = registry.save(
        train["model"],
        name="iris-classifier",
        metrics={"accuracy": train["accuracy"]},
    )
    print(f"registered iris-classifier v{version} (accuracy={train['accuracy']:.3f})")


if __name__ == "__main__":
    pipeline = Pipeline("iris-training")
    pipeline.add(load_data)
    pipeline.add(train, depends_on=[load_data])
    pipeline.add(register, depends_on=[train])

    pipeline.run(n_estimators=150)
