import glob
import json
import os
import signal
import subprocess
import sys

import pytest

WORKER = os.path.join(os.path.dirname(__file__), "_dist_train_worker.py")
TIMEOUT = 300


def _run(tmp_path, name: str, *, gan_start_step, max_steps: int, checkpointing_steps: int = 99):
    out_dir = tmp_path / name
    out_dir.mkdir()
    gan_start_arg = -1 if gan_start_step is None else gan_start_step
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        WORKER,
        "--out_dir",
        str(out_dir),
        "--max_steps",
        str(max_steps),
        "--gan_start_step",
        str(gan_start_arg),
        "--checkpointing_steps",
        str(checkpointing_steps),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.Popen(
        command,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        log, _ = process.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.communicate()
        pytest.fail(f"分布式 trainer 超时，疑似 DDP/barrier 死锁，case={name}")

    if process.returncode != 0:
        pytest.fail(f"worker 退出码 {process.returncode}\n{log[-4000:]}")

    ranks = [json.load(open(path, encoding="utf-8")) for path in glob.glob(str(out_dir / "rank_*.json"))]
    assert len(ranks) == 2, f"rank 输出数量错误: {len(ranks)}"
    return out_dir, sorted(ranks, key=lambda item: item["rank"])


def test_gan_from_start_ddp_four_step_does_not_mark_ready_twice(tmp_path) -> None:
    _out_dir, ranks = _run(tmp_path, "gan_from_start", gan_start_step=0, max_steps=2)

    for rank in ranks:
        assert rank["global_step"] == 2
        assert rank["finite_losses"] is True
        assert rank["disc_step_at"] == [0, 1]


def test_cross_boundary_switches_lr_on_all_ranks_and_writes_stage1_once(tmp_path) -> None:
    out_dir, ranks = _run(tmp_path, "cross_boundary", gan_start_step=1, max_steps=2)

    for rank in ranks:
        assert rank["global_step"] == 2
        assert rank["finite_losses"] is True
        assert rank["disc_step_at"] == [1]
        assert rank["gen_lr"] == pytest.approx(2e-5)
        assert rank["disc_lr"] == pytest.approx(2e-5)
        assert rank["gen_sched_rebuilt"] is True
        assert rank["disc_sched_rebuilt"] is True

    checkpoints = glob.glob(str(out_dir / "checkpoints" / "stage1_final.pkl"))
    assert len(checkpoints) == 1


def test_periodic_checkpoint_is_written_by_main_process_once(tmp_path) -> None:
    out_dir, ranks = _run(tmp_path, "periodic_ckpt", gan_start_step=None, max_steps=3, checkpointing_steps=2)

    for rank in ranks:
        assert rank["global_step"] == 3
        assert rank["finite_losses"] is True
        assert rank["disc_step_at"] == []

    checkpoints = glob.glob(str(out_dir / "checkpoints" / "model_3.pkl"))
    assert len(checkpoints) == 1
