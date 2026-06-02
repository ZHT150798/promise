from pathlib import Path

import yaml


TRAINING_ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(path: str):
    return yaml.safe_load((TRAINING_ROOT / path).read_text(encoding="utf-8"))


def test_paper_training_stage_defaults() -> None:
    cfg = _read_yaml("conf/train/default.yaml")

    assert cfg["max_steps"] == 150000
    assert cfg["lr"] == "5e-5" or cfg["lr"] == 5e-5
    assert cfg["lr_stage2"] == "2e-5" or cfg["lr_stage2"] == 2e-5
    assert cfg["gan_start_step"] == 100000


def test_paper_loss_defaults() -> None:
    cfg = _read_yaml("conf/loss/default.yaml")

    assert cfg["lambda_l2"] == 2.0
    assert cfg["lambda_lpips"] == 4.0
    assert cfg["lambda_pe"] == 1.0
    assert cfg["lambda_gan"] == 0.05
    assert cfg["gan"] is True


def test_paper_lora_rank_defaults() -> None:
    cfg = _read_yaml("conf/model/morphodiff.yaml")

    assert cfg["lora_rank_unet"] == 32
    assert cfg["lora_rank_vae"] == 16
