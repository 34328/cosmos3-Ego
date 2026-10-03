"""Actual update LR is recorded without replacing the official scheduler."""
from types import SimpleNamespace
import json
import pytest
from cosmos3_ar_it2v.monitor import VideoTrainingMonitor, VideoStopPolicy


def test_learning_rate_snapshot_reads_all_actual_groups():
    monitor=VideoTrainingMonitor()
    scheduler=SimpleNamespace(get_last_lr=lambda:[2e-5,1e-5])
    monitor.on_before_optimizer_step(None,None,scheduler,None,iteration=100)
    assert monitor.learning_rates==[2e-5,1e-5]


def _history(path, rows):
    path.write_text(''.join(json.dumps(dict(step=step,video_loss=loss))+'\n' for step,loss in rows))
    return path


def test_resume_preserves_initial_baseline_and_loss_window(tmp_path):
    losses=[1.]*10+[2.]*7
    uninterrupted=VideoStopPolicy()
    for step,loss in enumerate(losses,1):
        uninterrupted.update(False,[loss],step/10,step=step)
    path=_history(tmp_path/'monitor.jsonl',list(enumerate(losses,1)))
    resumed=VideoStopPolicy.from_history(path,17)
    assert resumed.baseline==uninterrupted.baseline==1.
    assert list(resumed.losses)==list(uninterrupted.losses)
    assert not resumed.memory
    for step in range(18,28):
        assert resumed.update(False,[4.],0.,step=step)==uninterrupted.update(False,[4.],0.,step=step)


def test_resume_before_baseline_is_established(tmp_path):
    path=_history(tmp_path/'monitor.jsonl',[(i,1.) for i in range(1,6)])
    resumed=VideoStopPolicy.from_history(path,5)
    assert resumed.baseline is None
    for i in range(6,11):
        resumed.update(False,[2.],0.,step=i)
    assert resumed.baseline==1.5


def test_resume_uses_latest_branch_and_ignores_future(tmp_path):
    rows=[(i,1.) for i in range(1,21)]+[(11,2.),(12,3.),(13,99.)]
    path=_history(tmp_path/'monitor.jsonl',rows)
    resumed=VideoStopPolicy.from_history(path,12)
    assert resumed.baseline==1.
    assert list(resumed.losses)==[1.]*8+[2.,3.]


def test_resume_never_fills_branch_gaps_with_abandoned_rows(tmp_path):
    rows=[(i,1.) for i in range(1,21)]+[(11,2.)]
    path=_history(tmp_path/'monitor.jsonl',rows)
    with pytest.raises(ValueError,match='missing history steps'):
        VideoStopPolicy.from_history(path,12)


def test_resume_requires_complete_history(tmp_path):
    path=tmp_path/'monitor.jsonl'
    with pytest.raises(ValueError,match='missing'):
        VideoStopPolicy.from_history(path,12)
    _history(path,[(i,1.) for i in range(2,13)])
    with pytest.raises(ValueError,match='missing history steps'):
        VideoStopPolicy.from_history(path,12)
    assert VideoStopPolicy.from_history(path,0).baseline is None
