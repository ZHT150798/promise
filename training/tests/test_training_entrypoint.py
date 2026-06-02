import subprocess
import sys
from pathlib import Path


def test_train_entrypoint_runs_weight_free_stub_config(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "entrypoint"
    command = [
        sys.executable,
        "-m",
        "promise_train.entrypoints.train",
        "model=stub",
        "loss=stub",
        "data=stub",
        f"output_dir={output_dir}",
        f"hydra.run.dir={output_dir / 'hydra'}",
        "train.max_steps=1",
        "train.num_epochs=1",
        "train.batch_size=1",
        "train.grad_accum=1",
        "train.dataloader_num_workers=0",
        "train.ckpt_steps=99",
        "train.gan_start_step=null",
        "train.mixed_precision=no",
        "train.report_to=null",
        "train.val.skip_fid=true",
    ]

    process = subprocess.run(
        command,
        cwd=repo_root,
        env=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )

    assert process.returncode == 0, process.stdout[-4000:]
    assert (output_dir / "resolved_config.yaml").exists()
    assert (output_dir / "environment.txt").exists()
