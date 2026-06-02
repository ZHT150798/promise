from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from promise_train.models.morphodiff import (
    CHECKPOINT_KEYS,
    PAPER_RANK_UNET,
    PAPER_RANK_VAE,
    validate_checkpoint_schema,
)
from promise_train.trainer.trainer import TrainComponents, TrainerDebug, TrainingLoop


class FakeAccelerator:
    device = torch.device("cpu")
    sync_gradients = True
    num_processes = 1
    is_main_process = True

    def __init__(self) -> None:
        self.clip_calls = 0

    @contextmanager
    def accumulate(self, *_models):
        yield

    def backward(self, loss):
        assert torch.isfinite(loss).all()

    def clip_grad_norm_(self, _parameters, _max_norm):
        self.clip_calls += 1

    def unwrap_model(self, model):
        return model


class RecordingOptimizer:
    def __init__(self, lr: float) -> None:
        self.param_groups = [{"lr": lr}]
        self.steps = 0
        self.zero_grad_calls = 0

    def step(self):
        self.steps += 1

    def zero_grad(self, set_to_none=False):
        self.zero_grad_calls += 1


class RecordingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self):
        self.steps += 1


class StubModelGen:
    def __init__(self) -> None:
        self.forward_prompts = []

    def __call__(self, x_src, batch=None, prompt=None, args=None):
        self.forward_prompts.append(prompt)
        return x_src + 0.25, None, None

    def save_model(self, path: str) -> None:
        checkpoint = {key: {} for key in CHECKPOINT_KEYS}
        checkpoint["rank_unet"] = PAPER_RANK_UNET
        checkpoint["rank_vae"] = PAPER_RANK_VAE
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)


class StubDiscriminator:
    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def __call__(self, x, for_G=False, for_real=None):
        self.calls.append({"for_G": for_G, "for_real": for_real})
        return torch.ones(x.shape[0], dtype=x.dtype, device=x.device)

    def parameters(self):
        return [self.param]


class ZeroLpips:
    def __call__(self, pred, target):
        return torch.zeros(pred.shape[0], dtype=pred.dtype, device=pred.device)


class MeanAbsLpips:
    def __call__(self, pred, target):
        return (pred - target).abs().flatten(1).mean(dim=1)


class RecordingPELoss:
    def __init__(self) -> None:
        self.mag_prompts = []

    def __call__(self, pred, target, mag_prompt=None):
        self.mag_prompts.append(mag_prompt)
        return torch.zeros((), dtype=pred.dtype, device=pred.device)


class MeanSquaredPELoss(RecordingPELoss):
    def __call__(self, pred, target, mag_prompt=None):
        self.mag_prompts.append(mag_prompt)
        return (pred - target).pow(2).mean() + 0.125


class TrainingLoopHarness:
    def __init__(self, tmp_path, gan_start_step, max_steps, checkpointing_steps=99):
        self.tmp_path = tmp_path
        self.debug = TrainerDebug()
        self.gen_opt = RecordingOptimizer(lr=5e-5)
        self.disc_opt = RecordingOptimizer(lr=5e-5)
        self.scheduler_factory_calls = []
        self.model = StubModelGen()
        self.disc = StubDiscriminator()
        self.pe_loss = RecordingPELoss()
        self.cfg = SimpleNamespace(
            train=SimpleNamespace(
                max_steps=max_steps,
                num_epochs=1,
                gan_start_step=gan_start_step,
                lr_stage2=2e-5,
            ),
            loss=SimpleNamespace(gan=True),
        )
        self.args = SimpleNamespace(
            output_dir=str(tmp_path),
            pos_prompt="positive suffix",
            neg_prompt="negative suffix",
            neg_prob=0.0,
            lambda_l2=2.0,
            lambda_lpips=4.0,
            lambda_pe=1.0,
            lambda_gan=0.05,
            max_grad_norm=1.0,
            set_grads_to_none=False,
            checkpointing_steps=checkpointing_steps,
        )
        self.components = TrainComponents(
            accelerator=FakeAccelerator(),
            model_gen=self.model,
            net_disc=self.disc,
            optimizer=self.gen_opt,
            optimizer_disc=self.disc_opt,
            lr_scheduler=RecordingScheduler(),
            lr_scheduler_disc=RecordingScheduler(),
            dl_train=[self._batch() for _ in range(max_steps)],
            dl_val_list=[],
            net_lpips=ZeroLpips(),
            pe_loss_fn=self.pe_loss,
            layers_to_opt=[torch.nn.Parameter(torch.tensor(0.0))],
            args=self.args,
            evaluator=None,
        )
        self.loop = TrainingLoop(
            self.cfg,
            self.components,
            scheduler_factory=self._scheduler_factory,
            debug=self.debug,
        )

    def _batch(self):
        return {
            "conditioning_pixel_values": torch.zeros(1, 3, 4, 4),
            "output_pixel_values": torch.ones(1, 3, 4, 4),
            "pos_prompt": ["pos"],
            "neg_prompt": ["neg"],
            "mag_prompt": ["40× magnification"],
        }

    def _scheduler_factory(self, name, optimizer, num_warmup_steps, num_training_steps):
        self.scheduler_factory_calls.append(
            {
                "name": name,
                "optimizer": optimizer,
                "num_warmup_steps": num_warmup_steps,
                "num_training_steps": num_training_steps,
            }
        )
        return RecordingScheduler()

    def train(self):
        return self.loop.train()


def build_trainer(tmp_path, gan_start_step, max_steps, checkpointing_steps=99):
    return TrainingLoopHarness(tmp_path, gan_start_step, max_steps, checkpointing_steps)


def _clone_batch(batch):
    cloned = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def _reference_stage1_recon_oracle(batch, args, model, lpips_fn, pe_loss_fn, device):
    x_src = batch["conditioning_pixel_values"]
    x_tgt = batch["output_pixel_values"]
    batch_size = x_src.shape[0]
    batch["pos_prompt"] = [batch["pos_prompt"][i] + ", " + args.pos_prompt for i in range(batch_size)]
    batch["neg_prompt"] = [batch["neg_prompt"][i] + ", " + args.neg_prompt for i in range(batch_size)]
    mag_prompt = batch["mag_prompt"]
    neg_probs = torch.rand(batch_size).to(device)
    mixed_tag_prompt = [
        neg_tag if prob < args.neg_prob else pos_tag
        for neg_tag, pos_tag, prob in zip(batch["neg_prompt"], batch["pos_prompt"], neg_probs)
    ]
    is_mix = neg_probs.reshape(batch_size, 1, 1, 1) < args.neg_prob
    mixed_tgt = torch.where(is_mix, x_src, x_tgt)

    pred, _, _ = model(x_src.detach(), batch=batch, prompt=mixed_tag_prompt, args=args)
    loss_l2 = torch.nn.functional.mse_loss(pred.float(), mixed_tgt.detach().float(), reduction="mean") * args.lambda_l2
    loss_lpips = lpips_fn(pred.float(), mixed_tgt.detach().float()).mean() * args.lambda_lpips
    loss_pe = pe_loss_fn(pred.float(), mixed_tgt.detach().float(), mag_prompt=mag_prompt) * args.lambda_pe
    return {
        "loss_l2": loss_l2.detach().item(),
        "loss_lpips": loss_lpips.detach().item(),
        "loss_pe": loss_pe.detach().item(),
        "mixed_tag_prompt": mixed_tag_prompt,
        "mixed_tgt": mixed_tgt,
    }


def test_cross_boundary_switch_rebuilds_schedulers_and_saves_stage1(tmp_path):
    t = build_trainer(tmp_path, gan_start_step=1, max_steps=2)

    assert t.train() == 2

    assert t.gen_opt.param_groups[0]["lr"] == pytest.approx(2e-5)
    assert t.disc_opt.param_groups[0]["lr"] == pytest.approx(2e-5)
    assert t.debug.gen_sched_rebuilt is True
    assert t.debug.disc_sched_rebuilt is True
    assert len(t.scheduler_factory_calls) == 2
    assert t.scheduler_factory_calls[0]["optimizer"] is t.gen_opt
    assert t.scheduler_factory_calls[1]["optimizer"] is t.disc_opt
    assert (tmp_path / "checkpoints" / "stage1_final.pkl").exists()
    assert t.debug.disc_step_at == [1]


def test_stage1_single_step_matches_reference_reconstruction_oracle(tmp_path):
    t = build_trainer(tmp_path, gan_start_step=None, max_steps=1)
    batch = {
        "conditioning_pixel_values": torch.zeros(2, 3, 4, 4),
        "output_pixel_values": torch.ones(2, 3, 4, 4) * 2.0,
        "pos_prompt": ["pos0", "pos1"],
        "neg_prompt": ["neg0", "neg1"],
        "mag_prompt": ["40× magnification", "20× magnification"],
    }
    t.args.neg_prob = 0.5
    t.components.net_lpips = MeanAbsLpips()
    t.components.pe_loss_fn = MeanSquaredPELoss()
    oracle_model = StubModelGen()
    oracle_pe = MeanSquaredPELoss()

    torch.manual_seed(0)
    expected = _reference_stage1_recon_oracle(
        _clone_batch(batch),
        t.args,
        oracle_model,
        MeanAbsLpips(),
        oracle_pe,
        t.components.accelerator.device,
    )
    torch.manual_seed(0)
    actual = t.loop._train_batch(_clone_batch(batch), global_step=0)

    assert actual["loss_l2"] == pytest.approx(expected["loss_l2"])
    assert actual["loss_lpips"] == pytest.approx(expected["loss_lpips"])
    assert actual["loss_pe"] == pytest.approx(expected["loss_pe"])
    assert t.model.forward_prompts == [expected["mixed_tag_prompt"]]
    assert expected["mixed_tag_prompt"] == ["neg0, negative suffix", "pos1, positive suffix"]
    assert t.components.pe_loss_fn.mag_prompts == [["40× magnification", "20× magnification"]]


def test_gan_from_start_runs_discriminator_every_step(tmp_path):
    t = build_trainer(tmp_path, gan_start_step=0, max_steps=2)

    t.train()

    assert t.debug.disc_step_at == [0, 1]


def test_gan_never_skips_discriminator(tmp_path):
    t = build_trainer(tmp_path, gan_start_step=None, max_steps=2)

    t.train()

    assert t.debug.disc_step_at == []


def test_stage2_four_step_order_and_second_forward_uses_positive_prompt(tmp_path):
    t = build_trainer(tmp_path, gan_start_step=0, max_steps=1)

    t.train()

    assert t.debug.call_order == [
        "gen_opt",
        "gen_sched",
        "gen_opt",
        "gen_sched",
        "disc_opt",
        "disc_sched",
        "disc_opt",
    ]
    assert len(t.model.forward_prompts) == 2
    assert t.model.forward_prompts[0] == ["pos, positive suffix"]
    assert t.model.forward_prompts[1] == ["pos, positive suffix"]


def test_stage1_loss_is_finite_and_periodic_checkpoint_has_schema(tmp_path):
    t = build_trainer(tmp_path, gan_start_step=None, max_steps=3, checkpointing_steps=2)

    t.train()

    assert t.debug.disc_step_at == []
    assert t.pe_loss.mag_prompts == [["40× magnification"]] * 3
    checkpoint_path = tmp_path / "checkpoints" / "model_3.pkl"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_checkpoint_schema(checkpoint)
