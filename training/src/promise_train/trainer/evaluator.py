import csv
import gc
import os
import shutil
from collections.abc import Callable, Sequence
from math import nan
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torchvision import transforms


DEFAULT_VAL_SCALES = ("40x", "20x")


def resolve_val_scales(scales: Optional[Sequence[str]] = None, dl_count: Optional[int] = None) -> list[str]:
    """Resolve validation magnification labels.

    The default validation loaders cover 40x and 20x magnification.
    """

    resolved = list(scales or DEFAULT_VAL_SCALES)
    if dl_count is not None:
        resolved = resolved[:dl_count]
    return resolved


def maybe_calculate_fid(
    paths: Sequence[str],
    batch_size: int,
    device: str,
    dims: int = 2048,
    skip_fid: bool = False,
    fid_fn: Optional[Callable[..., float]] = None,
) -> float:
    if skip_fid:
        return nan

    if fid_fn is None:
        from pytorch_fid.fid_score import calculate_fid_given_paths

        fid_fn = calculate_fid_given_paths
    return float(fid_fn(paths, batch_size=batch_size, device=device, dims=dims))


def _mean_or_nan(values: Sequence[torch.Tensor]) -> float:
    if not values:
        return nan
    return torch.cat(list(values)).float().mean().item()


def _nanmean_or_nan(values: Sequence[float]) -> float:
    if not values or all(np.isnan(value) for value in values):
        return nan
    return float(np.nanmean(values))


class Evaluator:
    """Distributed evaluator for super-resolution metrics."""

    def __init__(
        self,
        accelerator: Any,
        output_dir: Union[str, Path],
        scales: Optional[Sequence[str]] = None,
        skip_fid: bool = False,
        fid_batch_size: int = 8,
    ) -> None:
        self.accelerator = accelerator
        self.output_dir = Path(output_dir)
        self.scales = resolve_val_scales(scales)
        self.skip_fid = skip_fid
        self.fid_batch_size = fid_batch_size

    def prepare_loaders(self, dataloaders: Sequence[Any]) -> list[Any]:
        if not dataloaders:
            return []
        prepared = self.accelerator.prepare(*dataloaders)
        if len(dataloaders) == 1:
            return [prepared]
        return list(prepared)

    def evaluate_scalar_loader(
        self,
        dataloader: Any,
        metric_key: str = "metric",
        prepare_loader: bool = True,
        run_fid_barriers: bool = False,
    ) -> dict[str, Any]:
        """Small production-code path for distributed aggregation tests."""

        loader = self.prepare_loaders([dataloader])[0] if prepare_loader else dataloader
        values: list[float] = []
        for batch in loader:
            gathered = self.accelerator.gather_for_metrics(batch[metric_key])
            values.extend(float(value) for value in gathered.tolist())

        best = torch.tensor([min(values)], device=self.accelerator.device)
        if self.accelerator.num_processes > 1:
            torch.distributed.broadcast(best, src=0)

        if run_fid_barriers:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                _ = sum(values)
            self.accelerator.wait_for_everyone()

        self.accelerator.wait_for_everyone()
        return {
            "rank": self.accelerator.process_index,
            "gathered_values": sorted(set(int(value) for value in values)),
            "gathered_count": len(values),
            "best": best.item(),
        }

    def evaluate(
        self,
        model: Any,
        dataloaders: Sequence[Any],
        net_lpips: Any,
        pe_loss_fn: Any,
        args: Any,
        global_step: int,
        best_lpips: float,
        logs: dict[str, float],
        save_vis: bool = False,
        prepare_loaders: bool = True,
    ) -> tuple[float, dict[str, float]]:
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        device = self.accelerator.device
        rank = self.accelerator.process_index

        loaders = self.prepare_loaders(dataloaders) if prepare_loaders else list(dataloaders)
        scales = resolve_val_scales(self.scales, len(loaders))
        val_results: dict[str, dict[str, float]] = {scale: {} for scale in scales}
        all_lpips: list[float] = []
        all_pe: list[float] = []
        all_fid: list[float] = []

        for scale, dataloader in zip(scales, loaders):
            gt_dir = self.output_dir / "temp_eval" / f"step{global_step}" / scale / "gt"
            sr_dir = self.output_dir / "temp_eval" / f"step{global_step}" / scale / "sr"
            if not self.skip_fid and self.accelerator.is_main_process:
                gt_dir.mkdir(parents=True, exist_ok=True)
                sr_dir.mkdir(parents=True, exist_ok=True)
            self.accelerator.wait_for_everyone()

            lpips_values: list[torch.Tensor] = []
            pe_values: list[torch.Tensor] = []
            vis_count = 0

            for step, batch in enumerate(dataloader):
                x_src = batch["conditioning_pixel_values"].to(device)
                x_tgt = batch["output_pixel_values"].to(device)
                batch_size = x_src.shape[0]

                with torch.no_grad():
                    prompt_prefix = batch["pos_prompt"][0]
                    batch["prompt"] = [prompt_prefix + ", " + args.pos_prompt for _ in range(batch_size)]
                    x_tgt_pred, _, _ = model(x_src, batch=batch, prompt=batch["prompt"], args=args)

                    loss_lpips = net_lpips(x_tgt_pred, x_tgt).mean()
                    lpips_values.append(self.accelerator.gather_for_metrics(loss_lpips.detach().reshape(1)))

                    if scale in {"20x", "20x_X8"}:
                        loss_pe = pe_loss_fn(x_tgt_pred, x_tgt, mag_prompt="20× magnification")
                    else:
                        loss_pe = pe_loss_fn(x_tgt_pred, x_tgt)
                    pe_values.append(self.accelerator.gather_for_metrics(loss_pe.detach().reshape(1)))

                    if not self.skip_fid:
                        for batch_idx in range(batch_size):
                            filename = f"r{rank}_s{step}_b{batch_idx}.png"
                            transforms.ToPILImage()(x_tgt[batch_idx].cpu().clamp(0, 1)).save(gt_dir / filename)
                            transforms.ToPILImage()(x_tgt_pred[batch_idx].cpu().clamp(0, 1)).save(sr_dir / filename)

                    if save_vis and self.accelerator.is_main_process and vis_count < 5:
                        to_vis = lambda x: x.cpu().detach() * 0.5 + 0.5
                        combined = torch.cat(
                            [to_vis(x_src), to_vis(x_tgt_pred), to_vis(torch.clamp(x_tgt, -1, 1))],
                            dim=3,
                        )
                        output_file = self.output_dir / "eval" / f"val_{scale}_{step}.png"
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        transforms.ToPILImage()(combined[0]).save(output_file)
                        vis_count += 1

            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                fid_value = maybe_calculate_fid(
                    [str(gt_dir), str(sr_dir)],
                    batch_size=self.fid_batch_size,
                    device=str(device),
                    dims=2048,
                    skip_fid=self.skip_fid,
                )
                lpips_mean = _mean_or_nan(lpips_values)
                pe_mean = _mean_or_nan(pe_values)

                val_results[scale] = {"lpips": lpips_mean, "pe": pe_mean, "fid": fid_value}
                all_lpips.append(lpips_mean)
                all_pe.append(pe_mean)
                all_fid.append(fid_value)
                if not self.skip_fid:
                    shutil.rmtree(gt_dir.parent, ignore_errors=True)

            self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            avg_lpips = _nanmean_or_nan(all_lpips)
            avg_pe = _nanmean_or_nan(all_pe)
            avg_fid = _nanmean_or_nan(all_fid)

            logs["val/avg_lpips"] = avg_lpips
            logs["val/avg_pe"] = avg_pe
            logs["val/avg_fid"] = avg_fid
            for scale in scales:
                result = val_results[scale]
                logs[f"val/{scale}_lpips"] = result["lpips"]
                logs[f"val/{scale}_pe"] = result["pe"]
                logs[f"val/{scale}_fid"] = result["fid"]

            print(f"[Step {global_step}] Avg: LPIPS={avg_lpips:.4f}, PE={avg_pe:.4f}, FID={avg_fid:.4f}")
            for scale in scales:
                result = val_results[scale]
                print(
                    f"  [{scale.upper()}] LPIPS={result['lpips']:.4f}, "
                    f"PE={result['pe']:.4f}, FID={result['fid']:.4f}"
                )

            unwrapped_model = self.accelerator.unwrap_model(model)
            if avg_lpips < best_lpips:
                best_lpips = avg_lpips
                if hasattr(unwrapped_model, "save_model"):
                    output_model = (
                        self.output_dir
                        / "checkpoints"
                        / f"bestmodel_{global_step}_{round(avg_lpips, 4)}_{round(avg_pe, 4)}.pkl"
                    )
                    output_model.parent.mkdir(parents=True, exist_ok=True)
                    unwrapped_model.save_model(str(output_model))

            self._write_csv(global_step, scales, val_results, avg_lpips, avg_pe, avg_fid)
            shutil.rmtree(self.output_dir / "temp_eval" / f"step{global_step}", ignore_errors=True)

        best_lpips_tensor = torch.tensor([best_lpips], device=device)
        if self.accelerator.num_processes > 1:
            torch.distributed.broadcast(best_lpips_tensor, src=0)
        best_lpips = best_lpips_tensor.item()

        self.accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        model.train()
        return best_lpips, logs

    def _write_csv(
        self,
        global_step: int,
        scales: Sequence[str],
        val_results: dict[str, dict[str, float]],
        avg_lpips: float,
        avg_pe: float,
        avg_fid: float,
    ) -> None:
        log_file = self.output_dir / "eval" / "val_log.csv"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_exists = log_file.is_file()
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                header = ["step", "avg_lpips", "avg_pe", "avg_fid"]
                for scale in scales:
                    header += [f"{scale}_lpips", f"{scale}_pe", f"{scale}_fid"]
                writer.writerow(header)
            row: list[Union[float, int]] = [global_step, avg_lpips, avg_pe, avg_fid]
            for scale in scales:
                result = val_results[scale]
                row += [result["lpips"], result["pe"], result["fid"]]
            writer.writerow(row)


__all__ = ["DEFAULT_VAL_SCALES", "Evaluator", "maybe_calculate_fid", "resolve_val_scales"]
