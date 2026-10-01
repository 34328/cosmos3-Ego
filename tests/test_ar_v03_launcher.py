import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cosmos3_joint_video_hand_pose/scripts/launch_ar_v0_3.sh"


def run_launcher(tmp_path, *, missing_tokenizer=False):
    assets = tmp_path / "assets"
    checkpoint = assets / "checkpoint"
    tokenizer = assets / "tokenizer"
    checkpoint.mkdir(parents=True)
    tokenizer.mkdir()
    (assets / "vae.pth").touch()
    if not missing_tokenizer:
        (tokenizer / "tokenizer_config.json").write_text("{}")
        (tokenizer / "tokenizer.json").write_text("{}")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    receipt = tmp_path / "torchrun.json"
    fake = bin_dir / "torchrun"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['TEST_TORCHRUN_RECEIPT']).write_text(json.dumps({"
        "'argv': sys.argv[1:], 'cwd': os.getcwd(), 'pythonpath': os.environ['PYTHONPATH']}))\n"
    )
    fake.chmod(0o755)
    env = {
        **os.environ, "PATH": str(bin_dir) + ":" + os.environ["PATH"],
        "BASE_CHECKPOINT_PATH": str(checkpoint), "WAN_VAE_PATH": str(assets / "vae.pth"),
        "TEXT_TOKENIZER_PATH": str(tokenizer), "OUTPUT_ROOT": str(tmp_path / "output"),
        "TEST_TORCHRUN_RECEIPT": str(receipt),
        "NPROC_PER_NODE": "8", "NNODES": "2", "NODE_RANK": "1",
        "MASTER_ADDR": "10.3.12.57", "MASTER_PORT": "51503",
        "EXTRA_TAIL_OVERRIDES": "trainer.max_iter=20 job.wandb_mode=disabled",
    }
    # Parent shells may carry variables for another active experiment.
    for key in ("TOML_FILE", "TRAINING_MODULE", "TRAINING_PYTHONPATH", "WORKDIR"):
        env.pop(key, None)
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)
    return result, receipt


def test_launcher_routes_v03_through_official_cli_and_preserves_topology_overrides(tmp_path):
    result, receipt = run_launcher(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    row = json.loads(receipt.read_text())
    assert row["argv"] == [
        "--nproc_per_node=8", "--master_port=51503", "--nnodes=2", "--node_rank=1",
        "--master_addr=10.3.12.57", "-m", "cosmos3_joint_video_hand_pose.src.train_ar_v03",
        "--sft-toml=" + str(ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_3.toml"),
        "--", "trainer.max_iter=20", "job.wandb_mode=disabled",
    ]
    assert row["cwd"] == str(ROOT)
    assert row["pythonpath"] == str(ROOT) + ":" + str(ROOT / "packages/cosmos3")


def test_launcher_rejects_missing_tokenizer_identity_before_torchrun(tmp_path):
    result, receipt = run_launcher(tmp_path, missing_tokenizer=True)
    assert result.returncode != 0
    assert "missing tokenizer_config.json" in result.stderr
    assert not receipt.exists()
