"""V0.2 must reclaim old cyclic graphs even after automatic GC is disabled."""

import gc
import weakref

from cosmos_framework.callbacks.manual_gc import ManualGarbageCollection
from cosmos3_joint_video_hand_pose.src.config import _ar_v02_experiment


def test_full_gc_reclaims_generation_two_cycles_from_first_update():
    callback = ManualGarbageCollection(**_ar_v02_experiment(True)["trainer"]["callbacks"]["manual_gc"])
    assert callback.every_n == 1 and callback.warm == 0 and callback.gc_level == 2

    class Graph:
        pass

    enabled = gc.isenabled()
    try:
        callback.on_training_step_start(None, {}, iteration=0)
        assert not gc.isenabled()
        graph = Graph()
        graph.cycle = graph
        reference = weakref.ref(graph)
        gc.collect(2)  # Promote the still-live object into the oldest generation.
        del graph
        gc.collect(1)
        assert reference() is not None  # The inherited shallow policy misses it.
        callback.every_n_impl(None, None, {}, {}, None, iteration=1)
        assert reference() is None
    finally:
        gc.collect(2)
        if enabled:
            gc.enable()
