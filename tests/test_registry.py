import pytest

from cairn import registry
from cairn.exceptions import ModelNotFoundError


class DummyModel:
    def __init__(self, weight):
        self.weight = weight

    def predict(self, x):
        return x * self.weight


def test_save_and_load_latest_version():
    registry.save(DummyModel(2), name="dummy", metrics={"accuracy": 0.9})
    model = registry.load("dummy")
    assert model.predict(10) == 20


def test_versions_increment():
    registry.save(DummyModel(1), name="versioned")
    registry.save(DummyModel(2), name="versioned")
    v2 = registry.save(DummyModel(3), name="versioned")

    assert v2 == 3
    assert registry.versions("versioned") == [1, 2, 3]


def test_load_specific_version():
    registry.save(DummyModel(1), name="pinned")
    registry.save(DummyModel(99), name="pinned")

    model = registry.load("pinned", version=1)
    assert model.weight == 1


def test_missing_model_raises():
    with pytest.raises(ModelNotFoundError):
        registry.load("does-not-exist")


def test_metadata_round_trip():
    registry.save(DummyModel(1), name="with-metrics", metrics={"auc": 0.87}, params={"lr": 0.01})
    meta = registry.metadata("with-metrics")
    assert meta["metrics"]["auc"] == 0.87
    assert meta["params"]["lr"] == 0.01
