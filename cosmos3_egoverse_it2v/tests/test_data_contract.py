from cosmos_framework.data.generator.sequence_packing import SequencePlan as OfficialSequencePlan

from cosmos3_egoverse_it2v.src.data import SequencePlan, _aligned, _indices


def test_temporal_indices_keep_first_and_4n_plus_1():
    indexes = _indices(1000, 601)
    assert indexes[0] == 0
    assert len(indexes) == 601
    assert (len(indexes) - 1) % 4 == 0
    assert len(set(indexes.tolist())) == len(indexes)


def test_dataset_uses_current_cosmos_sequence_plan():
    assert SequencePlan is OfficialSequencePlan
    plan = SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0])
    assert plan.vision_temporal_position_groups is None
