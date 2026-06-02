"""CPU/gloo distributed Trainer control-flow check.

Launched by test_trainer_distributed.py with
`accelerate launch --cpu --num_processes 2`.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import nn
from torch.utils.data import DataLoader, Dataset

from promise_train.models.morphodiff import CHECKPOINT_KEYS, PAPER_RANK_UNET, PAPER_RANK_VAE
from promise_train.trainer.trainer import TrainComponents, TrainerDebug, TrainingLoop


class TinyTrainDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        value = float(idx + 1)
        return {
            "conditioning_pixel_values": torch.ones(3, 8, 8) * value,
            "output_pixel_values": torch.ones(3, 8, 8) * (value + 0.5),
            "pos_prompt": "pos",
            "neg_prompt": "neg",
            "mag_prompt": "40× magnification",
        }


class TinyGen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_weight = nn.Parameter(torch.tensor(0.2))
        self.neg_weight = nn.Parameter(torch.tensor(-0.1))
        self.bias = nn.Parameter(torch.tensor(0.05))

    def forward(self, x_src, batch=None, prompt=None, args=None):
        prompt0 = prompt[0] if isinstance(prompt, list) else str(prompt)
        weight = self.neg_weight if "neg" in prompt0 else self.pos_weight
        return x_src * (1.0 + weight) + self.bias, None, None

    def save_model(self, path: str) -> None:
        checkpoint = {key: {} for key in CHECKPOINT_KEYS}
        checkpoint["rank_unet"] = PAPER_RANK_UNET
        checkpoint["rank_vae"] = PAPER_RANK_VAE
        checkpoint["state_dict_unet"] = self.state_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)


class TinyDisc(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, for_G: bool = False, for_real=None):
        sign = 1.0 if (for_G or for_real) else -1.0
        return x.flatten(1).mean(dim=1) * self.weight * sign


class TinyLpips(nn.Module):
    def forward(self, pred, target):
        return (pred - target).abs().flatten(1).mean(dim=1)


class TinyPELoss(nn.Module):
    def forward(self, pred, target, mag_prompt=None):
        return (pred - target).pow(2).mean() * 0.1


class TinyScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self):
        self.steps += 1


def _scheduler_factory(_name, optimizer, num_warmup_steps, num_training_steps):
    return TinyScheduler()


def _build_loop(args, accelerator):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    assert any(getattr(handler, "find_unused_parameters", False) for handler in [ddp_kwargs])

    model = TinyGen()
    disc = TinyDisc()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-5)
    optimizer_disc = torch.optim.SGD(disc.parameters(), lr=5e-5)
    dataloader = DataLoader(TinyTrainDataset(args.dataset_len), batch_size=1, shuffle=False)

    model, disc, optimizer, optimizer_disc, dataloader = accelerator.prepare(
        model,
        disc,
        optimizer,
        optimizer_disc,
        dataloader,
    )
    layers_to_opt = list(accelerator.unwrap_model(model).parameters())
    debug = TrainerDebug()
    cfg = SimpleNamespace(
        train=SimpleNamespace(
            max_steps=args.max_steps,
            num_epochs=1,
            gan_start_step=args.gan_start_step,
            lr_stage2=2e-5,
        ),
        loss=SimpleNamespace(gan=True),
    )
    legacy_args = SimpleNamespace(
        output_dir=args.out_dir,
        pos_prompt="positive suffix",
        neg_prompt="negative suffix",
        neg_prob=args.neg_prob,
        lambda_l2=2.0,
        lambda_lpips=4.0,
        lambda_pe=1.0,
        lambda_gan=0.05,
        max_grad_norm=1.0,
        set_grads_to_none=False,
        checkpointing_steps=args.checkpointing_steps,
    )
    components = TrainComponents(
        accelerator=accelerator,
        model_gen=model,
        net_disc=disc,
        optimizer=optimizer,
        optimizer_disc=optimizer_disc,
        lr_scheduler=TinyScheduler(),
        lr_scheduler_disc=TinyScheduler(),
        dl_train=dataloader,
        dl_val_list=[],
        net_lpips=TinyLpips(),
        pe_loss_fn=TinyPELoss(),
        layers_to_opt=layers_to_opt,
        args=legacy_args,
        evaluator=None,
    )
    return TrainingLoop(cfg, components, scheduler_factory=_scheduler_factory, debug=debug), components, debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--gan_start_step", type=int, default=-1)
    parser.add_argument("--dataset_len", type=int, default=8)
    parser.add_argument("--checkpointing_steps", type=int, default=99)
    parser.add_argument("--neg_prob", type=float, default=0.0)
    args = parser.parse_args()
    if args.gan_start_step < 0:
        args.gan_start_step = None

    if not hasattr(torch.cpu, "device_count"):
        torch.cpu.device_count = lambda: 1
    if not hasattr(torch.cpu, "set_device"):
        torch.cpu.set_device = lambda _device: None

    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    loop, components, debug = _build_loop(args, accelerator)
    global_step = loop.train()
    logs = debug.last_logs
    finite_losses = bool(logs) and all(torch.isfinite(torch.tensor(value)).item() for value in logs.values())
    output = {
        "rank": accelerator.process_index,
        "global_step": global_step,
        "finite_losses": finite_losses,
        "disc_step_at": debug.disc_step_at,
        "gen_lr": components.optimizer.param_groups[0]["lr"],
        "disc_lr": components.optimizer_disc.param_groups[0]["lr"],
        "gen_sched_rebuilt": debug.gen_sched_rebuilt,
        "disc_sched_rebuilt": debug.disc_sched_rebuilt,
    }
    accelerator.wait_for_everyone()
    with open(os.path.join(args.out_dir, f"rank_{accelerator.process_index}.json"), "w", encoding="utf-8") as file:
        json.dump(output, file)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
