"""Launch one approved node through MCP; atomic claim prevents duplicate runs."""
from pathlib import Path
import datetime
import getpass
import json
import os
import socket
import subprocess
import sys


def main():
    plan_path=Path(sys.argv[1]).resolve();node=sys.argv[2]
    plan=json.loads(plan_path.read_text());repo=Path(plan['repo']);pre=plan_path.parent
    assert getpass.getuser()=='lzh' and node in plan['nodes']
    assert socket.gethostname()==plan['nodes'][node]['hostname']
    node_dir=pre/node;node_dir.mkdir(exist_ok=True);claim=node_dir/'launch_claim.json'
    if claim.exists():
        print(json.dumps({'already_claimed':str(claim)}));return
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==plan['source_commit']
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    rows=[r.split(',') for r in gpu.strip().splitlines()]
    assert len(rows)==8 and not procs.strip() and all(float(r[2])<100 and float(r[3])==0 for r in rows),gpu+procs
    rank=plan['nodes'][node]['rank']
    if rank==0:
        assert not Path(plan['run_path']).exists(),'Output exists; inspect instead of relaunching'
        with socket.socket() as s:s.bind((plan['master_addr'],plan['master_port']))
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('WANDB_','TORCHELASTIC_')) or k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','EXTRA_TAIL_OVERRIDES','TOML_FILE','TRAINING_MODULE','TRAINING_PYTHONPATH','DATASET_PATH'):
            env.pop(k,None)
    env.update(PATH='/home/lzh/miniconda3/envs/cosmos3/bin:'+env.get('PATH',''),LD_LIBRARY_PATH='',
        PYTHONPATH=str(repo)+':'+str(repo/'packages/cosmos3'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NNODES=str(len(plan['nodes'])),NPROC_PER_NODE='8',
        NODE_RANK=str(rank),MASTER_ADDR=plan['master_addr'],MASTER_PORT=str(plan['master_port']),
        NCCL_IB_DISABLE='1',NCCL_NET='Socket',NCCL_SOCKET_IFNAME='eth0',GLOO_SOCKET_IFNAME='eth0',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_DEBUG='INFO',WANDB_MODE='online',WANDB_ENTITY=plan['entity'],
        WANDB_PROJECT=plan['project'],OUTPUT_ROOT=str(node_dir),IMAGINAIRE_OUTPUT_ROOT=plan['output_root'],
        LOG_FILENAME='formal_sft.log',BASE_CHECKPOINT_PATH='/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp',
        WAN_VAE_PATH='/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth',TEXT_TOKENIZER_PATH='/mnt/checkpoints/Cosmos3-Nano/text_tokenizer',
        TOML_FILE=plan['toml_file'],
        EXTRA_TAIL_OVERRIDES='job.name='+plan['run_name']+' job.wandb_mode=online')
    command=['bash',str(repo/'cosmos3_ar_it2v/launch.sh')]
    receipt=dict(node=node,rank=rank,hostname=socket.gethostname(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=plan['source_commit'],command=command,gpu=gpu,cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        affinity=len(os.sched_getaffinity(0)),loadavg=os.getloadavg(),
        environment={k:env[k] for k in ('NNODES','NPROC_PER_NODE','NODE_RANK','MASTER_ADDR','MASTER_PORT','WANDB_MODE','WANDB_ENTITY','WANDB_PROJECT','IMAGINAIRE_OUTPUT_ROOT','EXTRA_TAIL_OVERRIDES')})
    with claim.open('x') as f:json.dump(receipt,f,indent=2)
    supervisor='''import datetime,json,pathlib,subprocess,sys
pdir=pathlib.Path(sys.argv[1]);cmd=json.loads(sys.argv[2]);r=json.loads((pdir/'launch_claim.json').read_text())
p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL);r['launcher_pid']=p.pid
(pdir/'started.json').write_text(json.dumps(r,indent=2));code=p.wait()
(pdir/'exit.json').write_text(json.dumps(dict(exit_code=code,launcher_pid=p.pid,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
sys.exit(code)
'''
    with (node_dir/'console.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-u','-c',supervisor,str(node_dir),json.dumps(command)],cwd=repo,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt['supervisor_pid']=p.pid;claim.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(launched=node,rank=rank,supervisor_pid=p.pid,run_path=plan['run_path'])))

if __name__=='__main__':main()
