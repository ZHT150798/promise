import glob
import json
import os
import signal
import subprocess
import sys

import pytest

WORKER = os.path.join(os.path.dirname(__file__), "_dist_eval_worker.py")
TIMEOUT = 300


def _run(n_procs: int, n_samples: int, skip_fid: int, tmp_path):
    out_dir = tmp_path / f"p{n_procs}_n{n_samples}_f{skip_fid}"
    out_dir.mkdir()
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--cpu",
        "--num_processes",
        str(n_procs),
        WORKER,
        "--n_samples",
        str(n_samples),
        "--skip_fid",
        str(skip_fid),
        "--out_dir",
        str(out_dir),
    ]
    process = subprocess.Popen(
        command,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        log, _ = process.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.communicate()
        pytest.fail(f"分布式 eval 超时，疑似 barrier 死锁，n_procs={n_procs}")

    if process.returncode != 0:
        pytest.fail(f"worker 退出码 {process.returncode}\n{log[-2000:]}")

    ranks = [json.load(open(path, encoding="utf-8")) for path in glob.glob(str(out_dir / "rank_*.json"))]
    assert ranks, "无 rank 输出文件"
    return ranks


def test_deduplicates_odd_valset(tmp_path) -> None:
    n_samples = 5
    main = next(rank for rank in _run(2, n_samples, 1, tmp_path) if rank["rank"] == 0)

    assert sorted(main["gathered_values"]) == list(range(n_samples)), "去重后样本集错误"
    assert main["gathered_count"] == n_samples, f"pad 未去重: {main['gathered_count']} != {n_samples}"


def test_best_broadcast_agreement(tmp_path) -> None:
    bests = {rank["best"] for rank in _run(2, 5, 1, tmp_path)}

    assert len(bests) == 1, f"各 rank best 不一致: {bests}"


def test_single_vs_double_process_equivalence(tmp_path) -> None:
    rank1 = _run(1, 5, 1, tmp_path)[0]
    rank2 = next(rank for rank in _run(2, 5, 1, tmp_path) if rank["rank"] == 0)

    assert sorted(rank1["gathered_values"]) == sorted(rank2["gathered_values"])
    assert rank1["gathered_count"] == rank2["gathered_count"] == 5


def test_skip_fid_false_barriers_do_not_hang(tmp_path) -> None:
    _run(2, 6, 0, tmp_path)
