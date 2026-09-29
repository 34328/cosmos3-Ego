"""Replay acceptance rejects divergence; tolerances are fixed before GPU runs."""
import importlib.util,json
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/"scripts/verify_ar_v02_replay.py"
spec=importlib.util.spec_from_file_location("replay_check",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(root):
    fields={"loss/video_raw":1.,"loss/action_raw":2.,"loss/total":3.}
    fields.update({f"loss/action_field_{i}":float(i) for i in range(8)})
    for folder in ("pretrain_probe","dataloader_trace"): (root/folder).mkdir(parents=True)
    row=dict(step=5,rank=0,data_sha256="abc",frame_counts=[129],clips=1,token_budget={},
             noise={"video_timesteps":[[123.]],"action_timesteps":[[321.]]},losses=fields)
    (root/"pretrain_probe/rank_00000.jsonl").write_text(json.dumps(row)+"\n")
    (root/"dataloader_trace/rank_00000.jsonl").write_text(json.dumps(dict(iteration=4,packed_global_id=5,sample_ids=["a"]))+"\n")
    (root/"loss_metrics.jsonl").write_text(json.dumps({"step":5,**fields})+"\n")

@pytest.mark.parametrize("mutation",[None,"data","sigma","loss","missing"])
def test_replay_assertions(tmp_path,mutation):
    a,b=tmp_path/"a",tmp_path/"b";fixture(a);fixture(b)
    path=b/"pretrain_probe/rank_00000.jsonl";row=json.loads(path.read_text())
    if mutation=="data":row["data_sha256"]="changed"
    if mutation=="sigma":row["noise"]["action_timesteps"]=[[322.]]
    if mutation=="loss":row["losses"]["loss/action_raw"]+=.01
    if mutation=="missing":row["losses"].pop("loss/action_field_0")
    path.write_text(json.dumps(row)+"\n")
    if mutation:
        with pytest.raises(AssertionError):m.compare_replay(a,b,steps=(5,),ranks=1)
    else:assert m.compare_replay(a,b,steps=(5,),ranks=1)["passed"]
