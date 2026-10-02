"""Pure video manifest boundaries, timebase, packing contract and exact resume."""
import csv
from io import BytesIO

import numpy as np
from PIL import Image
import pytest
import torch

from cosmos3_ar_it2v.dataset import EgoVerseIT2VDataset, IT2VIterableDataset, select_clip_frames


@pytest.fixture
def manifests(tmp_path):
    episodes = [dict(episode_hash='train_ep', split='train', total_frames=300,
                     fps=30, abs_zarr_path='/fake/train'),
                dict(episode_hash='test_ep', split='test', total_frames=300,
                     fps=30, abs_zarr_path='/fake/test')]
    segments = [dict(episode_hash=ep, split=split, span_index=i, start_idx=start,
                     end_idx=end, text_normalized='pick and place cup')
                for ep,split in [('train_ep','train'),('test_ep','test')]
                for i,(start,end) in enumerate([(0,200),(200,265),(265,298),(298,300)])]
    paths = []
    for name, rows in [('episodes',episodes),('segments',segments)]:
        path=tmp_path/f'{name}.csv'
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        paths.append(path)
    return paths


def dataset(manifests, **kw):
    return EgoVerseIT2VDataset(*manifests,caption_tokenizer=lambda text: [1,2,3], **kw)


@pytest.mark.parametrize('source_frames,expected',[(16,None),(17,17),(32,17),(33,33),(49,49),(65,65),(81,81),(97,97),(999,97)])
def test_largest_complete_tier(source_frames,expected):
    assert select_clip_frames(source_frames)==expected


def test_reject_partial_chunk():
    with pytest.raises(ValueError):
        select_clip_frames(100,tiers=(18,))


def test_original_splits_and_short_exclusions(manifests):
    train,test=dataset(manifests),dataset(manifests,split='test')
    assert len(train)==len(test)==3
    assert train.manifest_summary['segments_excluded_short']==1
    assert train.manifest_summary['clip_tier_counts']=={33:1,65:1,97:1}
    assert set(train.episodes).isdisjoint(test.episodes)
    assert all(r['episode_hash']=='train_ep' for r in train.rows)
    assert train.excluded[0]['reason']=='too_short_for_minimum_tier'


def test_windows_reproducible_and_within_caption(manifests):
    ds=dataset(manifests,seed=42)
    values=[]
    for epoch in range(12):
        ids=ds.window_indices(0,epoch=epoch)
        assert np.array_equal(ids,ds.window_indices(0,epoch=epoch))
        assert ids[0]>=0 and ids[-1]<200
        assert np.all(np.diff(ids)==1)
        values.append(ids[0])
    assert len(set(values))>1
    with pytest.raises(ValueError): ds.window_indices(0,window_start=150)
    ds.set_epoch(7)
    assert np.array_equal(ds.window_indices(0),ds.window_indices(0,epoch=7))


def test_native_video_only_sample_contract(manifests,monkeypatch):
    import zarr
    image=Image.fromarray(np.full((360,640,3),80,dtype=np.uint8))
    buf=BytesIO(); image.save(buf,format='JPEG'); encoded=buf.getvalue()
    class RGB:
        shape=(300,)
        def __getitem__(self,indices): return [encoded for _ in indices]
    accessed=[]
    class Group:
        def __getitem__(self,key):
            accessed.append(key)
            assert key=='images.front_1'
            return RGB()
    monkeypatch.setattr(zarr,'open_group',lambda *a,**k:Group())
    ds=dataset(manifests,cfg_dropout_rate=0)
    sample=ds.get_item_at_window(1,window_start=200)
    assert sample['video'].shape==(3,65,368,640)
    assert sample['video'].dtype==torch.uint8
    assert sample['conditioning_fps']==30
    assert sample['frame_start']==200 and sample['frame_end']==264
    assert torch.equal(sample['source_frame_indices'],torch.arange(200,265))
    assert sample['sequence_plan'].condition_frame_indexes_vision==[0]
    assert sample['sequence_plan'].has_vision and not sample['sequence_plan'].has_action
    assert not any('action' in k or 'state' in k for k in sample)
    assert accessed==['images.front_1']
    assert '30 FPS' in sample['ai_caption']
    assert '2.2 seconds' in sample['ai_caption']
    assert torch.equal(sample['image_size'],torch.tensor([368,640,368,640]))
    restored=ds.get_item_at_window(1,source_frame_indices=sample['source_frame_indices'])
    assert restored['window_start']==200
    assert torch.equal(restored['video'],sample['video'])
    with pytest.raises(ValueError,match='disagree'):
        ds.get_item_at_window(1,window_start=201,source_frame_indices=sample['source_frame_indices'])
    wrong=sample['source_frame_indices'].clone(); wrong[1]+=1
    with pytest.raises(ValueError,match='stride/tier'):
        ds.get_item_at_window(1,source_frame_indices=wrong)
    ds.cfg_dropout_rate=1
    assert ds.get_item_at_window(1)['ai_caption']==''


class MockDataset:
    def __len__(self): return 12
    def get_item_at_window(self,index,epoch=0): return index,epoch


def test_stream_disjoint_ranks_and_deterministic_order():
    streams=[]
    for rank in range(2):
        stream=IT2VIterableDataset(MockDataset(),seed=42)
        stream.shard_world_size=2; stream.shard_rank=rank
        it=iter(stream); streams.append([next(it) for _ in range(6)])
    assert set(streams[0]).isdisjoint(streams[1])
    assert {i for i,e in streams[0]+streams[1]}==set(range(12))
    a,b=iter(IT2VIterableDataset(MockDataset())),iter(IT2VIterableDataset(MockDataset()))
    assert [next(a) for _ in range(25)]==[next(b) for _ in range(25)]


@pytest.mark.parametrize('offset',[1,11,12,15])
def test_exact_resume_across_epoch(offset):
    stream=IT2VIterableDataset(MockDataset()); it=iter(stream)
    for _ in range(offset): next(it)
    restored=IT2VIterableDataset(MockDataset()); restored.load_state_dict(stream.state_dict())
    other=iter(restored)
    assert [next(it) for _ in range(20)]==[next(other) for _ in range(20)]


def test_resume_refuses_changed_sharding():
    stream=IT2VIterableDataset(MockDataset()); next(iter(stream))
    restored=IT2VIterableDataset(MockDataset()); restored.load_state_dict(stream.state_dict())
    restored.shard_world_size=2
    with pytest.raises(ValueError,match='topology'): next(iter(restored))
