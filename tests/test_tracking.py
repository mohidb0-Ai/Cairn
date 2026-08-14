from cairn import track


def test_run_records_params_and_metrics():
    with track.run("unit-test-experiment") as run:
        run.log_param("lr", 0.01)
        run.log_metric("loss", 0.5, step=0)
        run.log_metric("loss", 0.3, step=1)
        run_id = run.run_id

    saved = track.get(run_id)
    assert saved["experiment"] == "unit-test-experiment"
    assert saved["params"]["lr"] == 0.01
    assert len(saved["metrics"]["loss"]) == 2
    assert saved["metrics"]["loss"][-1]["value"] == 0.3


def test_list_filters_by_experiment():
    with track.run("exp-a") as run:
        run.log_param("x", 1)
    with track.run("exp-b") as run:
        run.log_param("x", 2)

    only_a = track.list(experiment="exp-a")
    assert len(only_a) == 1
    assert only_a[0]["experiment"] == "exp-a"
