from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from promise_train.trainer.trainer import TrainComponents, TrainingLoop


K_STEPS = 20
TOL = 1e-5


class AutogradAccelerator:
    device = torch.device("cpu")
    sync_gradients = True
    num_processes = 1
    is_main_process = True

    @contextmanager
    def accumulate(self, *_models):
        yield

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)

    def unwrap_model(self, model):
        return model


class ConstantScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class TrainableStage1Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, 3, kernel_size=1)
        self.bias = torch.nn.Parameter(torch.tensor(0.05))

    def forward(self, x_src, batch=None, prompt=None, args=None):
        return self.proj(x_src) + self.bias, None, None


class MeanAbsLpips:
    def __call__(self, pred, target):
        return (pred - target).abs().flatten(1).mean(dim=1)


class MeanSquaredPELoss:
    def __call__(self, pred, target, mag_prompt=None):
        return (pred - target).pow(2).mean() + 0.125


class UnusedDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, for_G=False, for_real=None):
        return x.flatten(1).mean(dim=1) + self.param


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


def _parameter_vector(model):
    return torch.cat([param.detach().reshape(-1).cpu() for param in model.parameters()])


def _make_args():
    return SimpleNamespace(
        pos_prompt="positive suffix",
        neg_prompt="negative suffix",
        neg_prob=0.0,
        lambda_l2=2.0,
        lambda_lpips=4.0,
        lambda_pe=1.0,
        lambda_gan=0.05,
        max_grad_norm=1.0e6,
        set_grads_to_none=False,
        checkpointing_steps=999,
    )


def _make_cfg():
    return SimpleNamespace(
        train=SimpleNamespace(
            max_steps=K_STEPS,
            num_epochs=1,
            gan_start_step=None,
            lr_stage2=None,
        ),
        loss=SimpleNamespace(gan=False),
    )


def _make_cached_batches():
    generator = torch.Generator(device="cpu").manual_seed(2026)
    batches = []
    for step in range(K_STEPS):
        x_src = torch.rand((1, 3, 8, 8), generator=generator, dtype=torch.float32)
        x_tgt = torch.rand((1, 3, 8, 8), generator=generator, dtype=torch.float32)
        batches.append(
            {
                "conditioning_pixel_values": x_src,
                "output_pixel_values": x_tgt,
                "pos_prompt": [f"pos-{step}"],
                "neg_prompt": [f"neg-{step}"],
                "mag_prompt": ["40× magnification" if step % 2 == 0 else "20× magnification"],
            }
        )
    return batches


def _assert_batches_equal(left, right):
    for key in ("conditioning_pixel_values", "output_pixel_values"):
        torch.testing.assert_close(left[key], right[key], atol=0.0, rtol=0.0)
    for key in ("pos_prompt", "neg_prompt", "mag_prompt"):
        assert left[key] == right[key]


def _make_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )


def _reference_stage1_train_step(batch, args, model, optimizer, scheduler, lpips_fn, pe_loss_fn, device):
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
    loss = loss_l2 + loss_lpips + loss_pe

    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

    return {
        "loss_l2": float(loss_l2.detach().item()),
        "loss_lpips": float(loss_lpips.detach().item()),
        "loss_pe": float(loss_pe.detach().item()),
        "loss_total": float(loss.detach().item()),
    }


def _make_training_loop(model, optimizer, scheduler, args):
    components = TrainComponents(
        accelerator=AutogradAccelerator(),
        model_gen=model,
        net_disc=UnusedDiscriminator(),
        optimizer=optimizer,
        optimizer_disc=torch.optim.SGD(UnusedDiscriminator().parameters(), lr=1.0e-3),
        lr_scheduler=scheduler,
        lr_scheduler_disc=ConstantScheduler(),
        dl_train=[],
        dl_val_list=[],
        net_lpips=MeanAbsLpips(),
        pe_loss_fn=MeanSquaredPELoss(),
        layers_to_opt=list(model.parameters()),
        args=args,
        evaluator=None,
    )
    return TrainingLoop(_make_cfg(), components)


def test_stage1_multistep_trajectory_matches_reference_oracle():
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(1234)
        base_model = TrainableStage1Model()
        reference_model = deepcopy(base_model)
        new_model = deepcopy(base_model)
        reference_args = _make_args()
        new_args = _make_args()
        cached_batches = _make_cached_batches()

        reference_optimizer = _make_optimizer(reference_model)
        new_optimizer = _make_optimizer(new_model)
        reference_scheduler = ConstantScheduler()
        new_scheduler = ConstantScheduler()
        new_loop = _make_training_loop(new_model, new_optimizer, new_scheduler, new_args)

        torch.testing.assert_close(_parameter_vector(reference_model), _parameter_vector(new_model), atol=0.0, rtol=0.0)
        _assert_batches_equal(_clone_batch(cached_batches[0]), _clone_batch(cached_batches[0]))

        rows = []
        for step, cached_batch in enumerate(cached_batches):
            reference_batch = _clone_batch(cached_batch)
            new_batch = _clone_batch(cached_batch)
            _assert_batches_equal(reference_batch, new_batch)

            torch.manual_seed(10_000 + step)
            expected = _reference_stage1_train_step(
                reference_batch,
                reference_args,
                reference_model,
                reference_optimizer,
                reference_scheduler,
                MeanAbsLpips(),
                MeanSquaredPELoss(),
                torch.device("cpu"),
            )
            torch.manual_seed(10_000 + step)
            actual = new_loop._train_batch(new_batch, global_step=step)
            actual["loss_total"] = actual["loss_l2"] + actual["loss_lpips"] + actual["loss_pe"]

            for key in ("loss_l2", "loss_lpips", "loss_pe", "loss_total"):
                assert actual[key] == pytest.approx(expected[key], abs=TOL), (
                    f"step={step} key={key} expected={expected[key]:.8f} actual={actual[key]:.8f}"
                )
            torch.testing.assert_close(_parameter_vector(reference_model), _parameter_vector(new_model), atol=TOL, rtol=0.0)
            rows.append((step, expected, actual))

        assert len(rows) == K_STEPS
        assert reference_scheduler.steps == K_STEPS
        assert new_scheduler.steps == K_STEPS
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
