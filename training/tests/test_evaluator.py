import math
from argparse import Namespace

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from promise_train.trainer.evaluator import Evaluator, maybe_calculate_fid, resolve_val_scales


def test_default_val_scales_match_reference_active_dl_list() -> None:
    assert resolve_val_scales(dl_count=2) == ["40x", "20x"]


def test_fid_is_not_computed_when_disabled() -> None:
    called = False

    def fid_fn(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("FID should not be computed when skip_fid is enabled")

    value = maybe_calculate_fid(
        ["gt", "sr"],
        batch_size=1,
        device="cpu",
        skip_fid=True,
        fid_fn=fid_fn,
    )

    assert math.isnan(value)
    assert called is False


class TinyEvalDataset(Dataset):
    def __init__(self, n: int = 2, size: int = 4) -> None:
        self.n = n
        self.size = size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        image = torch.zeros(3, self.size, self.size)
        return {
            "conditioning_pixel_values": image,
            "output_pixel_values": image,
            "pos_prompt": "20× magnification, 4× super-resolution",
            "neg_prompt": "",
            "null_prompt": "",
        }


class StubModel:
    def __init__(self) -> None:
        self.is_train = True
        self.was_eval = False

    def eval(self) -> None:
        self.was_eval = True
        self.is_train = False

    def train(self) -> None:
        self.is_train = True

    def __call__(self, x_src, batch=None, prompt=None, args=None):
        return x_src, None, None


class ZeroLpips:
    def __call__(self, pred, target):
        return torch.zeros(pred.shape[0], 1, 1, 1, device=pred.device)


class RecordingPELoss:
    def __init__(self) -> None:
        self.mag_prompts = []

    def __call__(self, pred, target, mag_prompt=None):
        self.mag_prompts.append(mag_prompt)
        return torch.tensor(0.0, device=pred.device)


class PadLikePELoss:
    def __call__(self, pred, target, mag_prompt="40× magnification"):
        if isinstance(mag_prompt, list):
            mag_prompt = mag_prompt[0]
        if mag_prompt == "40× magnification":
            torch.nn.functional.pad(target, (256, 256, 256, 256), mode="reflect")
        return torch.tensor(0.0, device=target.device)


def test_evaluator_single_process_with_fid_disabled(tmp_path) -> None:
    accelerator = Accelerator(cpu=True)
    model = StubModel()
    pe_loss = RecordingPELoss()
    evaluator = Evaluator(
        accelerator=accelerator,
        output_dir=tmp_path,
        scales=["20x"],
        skip_fid=True,
    )

    best_lpips, logs = evaluator.evaluate(
        model=model,
        dataloaders=[DataLoader(TinyEvalDataset(), batch_size=1, shuffle=False)],
        net_lpips=ZeroLpips(),
        pe_loss_fn=pe_loss,
        args=Namespace(pos_prompt="A high-resolution"),
        global_step=1,
        best_lpips=1.0,
        logs={},
        save_vis=False,
    )

    assert model.was_eval is True
    assert model.is_train is True
    assert best_lpips == 0.0
    assert logs["val/20x_lpips"] == 0.0
    assert logs["val/20x_pe"] == 0.0
    assert math.isnan(logs["val/20x_fid"])
    assert pe_loss.mag_prompts == ["20× magnification", "20× magnification"]


def test_evaluator_40x_accepts_512_targets_for_pe_padding(tmp_path) -> None:
    accelerator = Accelerator(cpu=True)
    evaluator = Evaluator(
        accelerator=accelerator,
        output_dir=tmp_path,
        scales=["40x"],
        skip_fid=True,
    )

    best_lpips, logs = evaluator.evaluate(
        model=StubModel(),
        dataloaders=[DataLoader(TinyEvalDataset(size=512), batch_size=1, shuffle=False)],
        net_lpips=ZeroLpips(),
        pe_loss_fn=PadLikePELoss(),
        args=Namespace(pos_prompt="A high-resolution"),
        global_step=1,
        best_lpips=1.0,
        logs={},
        save_vis=False,
    )

    assert best_lpips == 0.0
    assert logs["val/40x_pe"] == 0.0
