import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sanity_train_curve.py"
SPEC = importlib.util.spec_from_file_location("sanity_train_curve", SCRIPT_PATH)
sanity_train_curve = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sanity_train_curve)


def _cfg(mode, **sanity_overrides):
    cfg = OmegaConf.create(
        {
            "sanity": {
                "mode": mode,
                "window": 3,
                "overfit_decline_ratio": 0.9,
                "overfit_min_delta": 0.0,
                "max_abs_gan_g": 100.0,
                "max_abs_gan_d": 100.0,
                "min_mean_abs_d": 1.0e-8,
            }
        }
    )
    for key, value in sanity_overrides.items():
        OmegaConf.update(cfg, f"sanity.{key}", value, merge=False)
    return cfg


def _row(step, stage, recon, loss_g=None, loss_d=None):
    return {
        "step": step,
        "stage": stage,
        "lr": 1.0e-4,
        "recon_total": recon,
        "loss_l2": recon * 0.1,
        "loss_lpips": recon * 0.8,
        "loss_pe": recon * 0.1,
        "loss_G": loss_g,
        "loss_D": loss_d,
    }


def test_stability_mode_does_not_require_recon_decline_across_gan_boundary():
    rows = [
        _row(0, "stage1", 1.0),
        _row(1, "stage1", 1.1),
        _row(2, "stage1", 1.2),
        _row(3, "stage2", 1.8, 0.1, 0.2),
        _row(4, "stage2", 1.7, 0.1, 0.2),
        _row(5, "stage2", 1.9, 0.1, 0.2),
    ]

    summary = sanity_train_curve._assert_curve(rows, _cfg("stability", require_stage2=True))

    assert summary["mode"] == "stability"
    assert summary["declined"] is False
    assert summary["gan"]["stage2_steps"] == 3


def test_overfit_mode_uses_stage1_window_decline_not_global_last_window():
    rows = [
        _row(0, "stage1", 2.0),
        _row(1, "stage1", 1.9),
        _row(2, "stage1", 1.8),
        _row(3, "stage1", 1.4),
        _row(4, "stage1", 1.3),
        _row(5, "stage1", 1.2),
        _row(6, "stage2", 3.0, 0.1, 0.2),
        _row(7, "stage2", 3.1, 0.1, 0.2),
        _row(8, "stage2", 3.2, 0.1, 0.2),
    ]

    summary = sanity_train_curve._assert_curve(rows, _cfg("overfit"))

    assert summary["mode"] == "overfit"
    assert summary["stage1"]["declined"] is True
    assert summary["stage1"]["first_mean"] == pytest.approx(1.9)
    assert summary["stage1"]["last_mean"] == pytest.approx(1.3)


def test_overfit_mode_fails_when_stage1_has_no_clear_downward_trend():
    rows = [
        _row(0, "stage1", 1.0),
        _row(1, "stage1", 1.1),
        _row(2, "stage1", 1.2),
        _row(3, "stage1", 1.1),
        _row(4, "stage1", 1.2),
        _row(5, "stage1", 1.3),
    ]

    with pytest.raises(AssertionError, match="Stage-1 recon loss did not clearly decline"):
        sanity_train_curve._assert_curve(rows, _cfg("overfit"))
