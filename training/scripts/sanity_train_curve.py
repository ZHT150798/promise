import csv
import json
import math
from pathlib import Path
from typing import Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from promise_train.trainer.trainer import Trainer, TrainingLoop
from promise_train.utils.env import write_environment_report


LOSS_KEYS = ("loss_l2", "loss_lpips", "loss_pe", "loss_G", "loss_D")


def _select(cfg: DictConfig, key: str, default: Any) -> Any:
    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _finite_row(row: dict[str, Any]) -> bool:
    for key in ("loss_l2", "loss_lpips", "loss_pe", "recon_total", "loss_G", "loss_D"):
        value = row.get(key)
        if value is not None and value != "" and not math.isfinite(float(value)):
            return False
    return True


def _has_value(value: Any) -> bool:
    return value is not None and value != ""


def _stage_summary(rows: list[dict[str, Any]], stage: str, window: int, decline_ratio: float) -> dict[str, Any]:
    stage_rows = [row for row in rows if row["stage"] == stage]
    if not stage_rows:
        return {"steps": 0}
    stage_window = max(1, min(window, len(stage_rows)))
    first_values = [float(row["recon_total"]) for row in stage_rows[:stage_window]]
    last_values = [float(row["recon_total"]) for row in stage_rows[-stage_window:]]
    first_mean = _mean(first_values)
    last_mean = _mean(last_values)
    return {
        "steps": len(stage_rows),
        "window": stage_window,
        "first_mean": first_mean,
        "last_mean": last_mean,
        "declined": last_mean < first_mean,
        "decline_ratio": last_mean / first_mean if first_mean != 0 else float("inf"),
        "reference_ratio_met": last_mean <= first_mean * decline_ratio,
    }


def _summarize_rows(rows: list[dict[str, Any]], cfg: DictConfig) -> dict[str, Any]:
    mode = str(_select(cfg, "sanity.mode", "overfit"))
    window = int(_select(cfg, "sanity.window", 50))
    window = max(1, min(window, len(rows))) if rows else 1
    first_recon = [float(row["recon_total"]) for row in rows[:window]]
    last_recon = [float(row["recon_total"]) for row in rows[-window:]]
    first_mean = _mean(first_recon) if first_recon else float("nan")
    last_mean = _mean(last_recon) if last_recon else float("nan")
    decline_margin = float(_select(cfg, "sanity.decline_margin", 0.0))
    overfit_decline_ratio = float(_select(cfg, "sanity.overfit_decline_ratio", 0.95))

    gan_rows = [row for row in rows if row["stage"] == "stage2" and _has_value(row.get("loss_G"))]
    gan_summary: dict[str, Any] = {"stage2_steps": len(gan_rows)}
    if gan_rows:
        loss_g = [float(row["loss_G"]) for row in gan_rows]
        loss_d = [float(row["loss_D"]) for row in gan_rows]
        gan_summary.update(
            {
                "loss_g_mean": _mean(loss_g),
                "loss_d_mean": _mean(loss_d),
                "loss_g_max_abs": max(abs(value) for value in loss_g),
                "loss_d_max_abs": max(abs(value) for value in loss_d),
                "loss_d_mean_abs": _mean([abs(value) for value in loss_d]),
            }
        )

    return {
        "mode": mode,
        "steps": len(rows),
        "window": window,
        "first_recon_mean": first_mean,
        "last_recon_mean": last_mean,
        "declined": last_mean < first_mean - decline_margin,
        "stage1": _stage_summary(rows, "stage1", window, overfit_decline_ratio),
        "stage2": _stage_summary(rows, "stage2", window, overfit_decline_ratio),
        "gan": gan_summary,
    }


def _assert_gan_bounded(summary: dict[str, Any], cfg: DictConfig) -> None:
    if summary["gan"]["stage2_steps"] <= 0:
        return
    max_abs_g = summary["gan"]["loss_g_max_abs"]
    max_abs_d = summary["gan"]["loss_d_max_abs"]
    mean_abs_d = summary["gan"]["loss_d_mean_abs"]
    max_abs_g_allowed = float(_select(cfg, "sanity.max_abs_gan_g", 1.0e4))
    max_abs_d_allowed = float(_select(cfg, "sanity.max_abs_gan_d", 1.0e4))
    min_mean_abs_d = float(_select(cfg, "sanity.min_mean_abs_d", 1.0e-8))
    if max_abs_g > max_abs_g_allowed:
        raise AssertionError(f"GAN G loss is too large: max_abs_g={max_abs_g:.6f}")
    if max_abs_d > max_abs_d_allowed:
        raise AssertionError(f"GAN D loss is too large: max_abs_d={max_abs_d:.6f}")
    if mean_abs_d <= min_mean_abs_d:
        raise AssertionError(f"GAN D loss collapsed near zero: mean_abs_d={mean_abs_d:.12f}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "stage", "lr", "recon_total", *LOSS_KEYS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _assert_curve(rows: list[dict[str, Any]], cfg: DictConfig) -> dict[str, Any]:
    if not rows:
        raise AssertionError("No training rows were recorded.")
    if not all(_finite_row(row) for row in rows):
        bad_steps = [row["step"] for row in rows if not _finite_row(row)]
        raise AssertionError(f"Non-finite loss values at steps: {bad_steps[:10]}")

    summary = _summarize_rows(rows, cfg)
    mode = summary["mode"]
    _assert_gan_bounded(summary, cfg)

    if mode == "stability":
        require_stage2 = bool(_select(cfg, "sanity.require_stage2", True))
        if require_stage2 and summary["gan"]["stage2_steps"] <= 0:
            raise AssertionError("Stability mode expected Stage-2 GAN rows, but none were recorded.")
        return summary

    if mode == "overfit":
        stage1 = summary["stage1"]
        if stage1["steps"] <= 0:
            raise AssertionError("Overfit mode requires Stage-1 rows.")
        min_delta = float(_select(cfg, "sanity.overfit_min_delta", 0.0))
        clearly_declined = stage1["last_mean"] < stage1["first_mean"] - min_delta
        if not clearly_declined:
            raise AssertionError(
                "Stage-1 recon loss did not clearly decline: "
                f"first_{stage1['window']}_mean={stage1['first_mean']:.6f}, "
                f"last_{stage1['window']}_mean={stage1['last_mean']:.6f}, "
                f"min_delta={min_delta:.6f}"
            )
        return summary

    raise ValueError(f"Unsupported sanity.mode: {mode}")

    return summary


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment_report(output_dir)
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    trainer = Trainer(cfg)
    components = trainer._build_components()
    loop = TrainingLoop(cfg, components)
    accelerator = components.accelerator
    curve_path = Path(_select(cfg, "sanity.curve_csv", str(output_dir / "sanity" / "loss_curve.csv")))
    summary_path = Path(_select(cfg, "sanity.summary_json", str(output_dir / "sanity" / "summary.json")))
    print_every = int(_select(cfg, "sanity.print_every", 10))

    rows: list[dict[str, Any]] = []
    global_step = 0
    max_steps = int(cfg.train.max_steps)
    for _epoch in range(int(cfg.train.num_epochs)):
        for batch in components.dl_train:
            loop._maybe_transition(global_step)
            logs = loop._train_batch(batch, global_step)
            if accelerator.sync_gradients:
                recon_total = float(logs["loss_l2"] + logs["loss_lpips"] + logs["loss_pe"])
                row = {
                    "step": global_step,
                    "stage": "stage2" if loop._in_stage2(global_step) else "stage1",
                    "lr": float(components.optimizer.param_groups[0]["lr"]),
                    "recon_total": recon_total,
                }
                for key in LOSS_KEYS:
                    row[key] = _as_float(logs.get(key))
                rows.append(row)
                if accelerator.is_main_process and (global_step % print_every == 0 or global_step == max_steps - 1):
                    gan_text = ""
                    if row.get("loss_G") is not None:
                        gan_text = f" loss_G={row['loss_G']:.6f} loss_D={row['loss_D']:.6f}"
                    print(
                        f"[sanity] step={global_step} {row['stage']} "
                        f"recon={row['recon_total']:.6f} "
                        f"l2={row['loss_l2']:.6f} lpips={row['loss_lpips']:.6f} pe={row['loss_pe']:.6f}"
                        f"{gan_text}"
                    )
                global_step += 1
            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    if accelerator.is_main_process:
        _write_csv(curve_path, rows)
        try:
            summary = _assert_curve(rows, cfg)
            summary["passed"] = True
        except AssertionError as exc:
            summary = _summarize_rows(rows, cfg)
            summary["passed"] = False
            summary["failure"] = str(exc)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[sanity] wrote curve: {curve_path}")
            print(f"[sanity] wrote summary: {summary_path}")
            print(f"[sanity] failed: {exc}")
            raise
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[sanity] wrote curve: {curve_path}")
        print(f"[sanity] wrote summary: {summary_path}")
        print(
            "[sanity] recon first/last mean: "
            f"{summary['first_recon_mean']:.6f} -> {summary['last_recon_mean']:.6f}; "
            f"stage2_steps={summary['gan']['stage2_steps']}"
        )

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
