#!/usr/bin/env python3
"""Compare actual official-Trainer replay evidence, without running training."""
import argparse
import json
import math
from pathlib import Path

def read_rows(path, key):
    rows = {}
    for line in path.read_text().splitlines():
        row=json.loads(line)
        index=int(row[key])
        if index in rows:
            raise AssertionError(f"duplicate {key}={index}: {path}")
        rows[index]=row
    return rows

def compare_replay(reference, replay, *, steps=(5,6), ranks=8, atol=1e-6, rtol=1e-5):
    checks=[]
    for rank in range(ranks):
        name=f"rank_{rank:05d}.jsonl"
        before=read_rows(Path(reference)/"pretrain_probe"/name,"step")
        after=read_rows(Path(replay)/"pretrain_probe"/name,"step")
        a_trace=read_rows(Path(reference)/"dataloader_trace"/name,"iteration")
        b_trace=read_rows(Path(replay)/"dataloader_trace"/name,"iteration")
        for step in steps:
            a,b=before[step],after[step]
            assert a_trace[step-1]==b_trace[step-1], f"rank{rank} step{step}: dataloader cursor/IDs differ"
            for key in ("data_sha256","frame_counts","clips","token_budget","noise"):
                assert a[key]==b[key], f"rank{rank} step{step}: {key} differs"
            assert a["noise"]["action_timesteps"] is not None, "missing action sigma evidence"
            assert a["losses"].keys()==b["losses"].keys(), "metric fields differ"
            assert len([k for k in a["losses"] if k.startswith("loss/action_")])>=9, "missing field losses"
            max_error=0
            for key,x in a["losses"].items():
                y=b["losses"][key]
                assert math.isfinite(x) and math.isfinite(y), f"nonfinite {key}"
                assert math.isclose(x,y,abs_tol=atol,rel_tol=rtol), f"rank{rank} step{step}: {key} {x} != {y}"
                max_error=max(max_error,abs(x-y))
            checks.append(dict(rank=rank,step=step,max_metric_abs_error=max_error))
    # Check the actual durable globally reduced logs as well.
    a=read_rows(Path(reference)/"loss_metrics.jsonl","step")
    b=read_rows(Path(replay)/"loss_metrics.jsonl","step")
    for step in steps:
        for key,x in a[step].items():
            if key.startswith(("loss/","sigma/")):
                y=b[step][key]
                assert x is not None and y is not None and math.isclose(x,y,abs_tol=atol,rel_tol=rtol), (step,key,x,y)
    return dict(passed=True,ranks=ranks,steps=list(steps),atol=atol,rtol=rtol,
                data_and_sigma_exact=True,checks=checks)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--replay",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=compare_replay(a.reference,a.replay)
    a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
