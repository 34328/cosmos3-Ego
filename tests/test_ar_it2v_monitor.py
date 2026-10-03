"""Actual update LR is recorded without replacing the official scheduler."""
from types import SimpleNamespace
from cosmos3_ar_it2v.monitor import VideoTrainingMonitor


def test_learning_rate_snapshot_reads_all_actual_groups():
    monitor=VideoTrainingMonitor()
    scheduler=SimpleNamespace(get_last_lr=lambda:[2e-5,1e-5])
    monitor.on_before_optimizer_step(None,None,scheduler,None,iteration=100)
    assert monitor.learning_rates==[2e-5,1e-5]
