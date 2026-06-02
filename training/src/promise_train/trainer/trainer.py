import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from promise_train.data import build_dataset
from promise_train.losses.gan import build_discriminator
from promise_train.losses.pe_loss import build_pe_loss
from promise_train.models import build_model
from promise_train.trainer.evaluator import Evaluator
from promise_train.utils.legacy_args import build_legacy_args


@dataclass(frozen=True)
class OptimizerStepSpec:
    name: str
    optimizer: str
    scheduler_step: bool


GAN_UPDATE_SCHEDULE = (
    OptimizerStepSpec("reconstruction", "generator", True),
    OptimizerStepSpec("gan_generator", "generator", True),
    OptimizerStepSpec("discriminator_real", "discriminator", True),
    OptimizerStepSpec("discriminator_fake", "discriminator", False),
)


@dataclass(frozen=True)
class StageTransition:
    enabled: bool
    lr_stage2: Optional[float]
    remaining_steps: int
    save_tag: str = "stage1_final"


@dataclass
class TrainerDebug:
    call_order: list[str] = field(default_factory=list)
    disc_step_at: list[int] = field(default_factory=list)
    gen_sched_rebuilt: bool = False
    disc_sched_rebuilt: bool = False
    stage_transitions: list[int] = field(default_factory=list)
    last_logs: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainComponents:
    accelerator: Any
    model_gen: Any
    net_disc: Any
    optimizer: Any
    optimizer_disc: Any
    lr_scheduler: Any
    lr_scheduler_disc: Any
    dl_train: Any
    dl_val_list: list[Any]
    net_lpips: Any
    pe_loss_fn: Any
    layers_to_opt: list[Any]
    args: Any
    evaluator: Optional[Evaluator] = None


def in_stage2(global_step: int, gan_start_step: Optional[int]) -> bool:
    return gan_start_step is not None and global_step >= gan_start_step


def should_transition_to_stage2(global_step: int, gan_start_step: Optional[int]) -> bool:
    return gan_start_step is not None and global_step == gan_start_step


def build_stage2_transition(cfg: Any, global_step: int) -> StageTransition:
    should_transition = should_transition_to_stage2(global_step, cfg.train.gan_start_step)
    max_steps = int(cfg.train.max_steps)
    return StageTransition(
        enabled=should_transition,
        lr_stage2=cfg.train.lr_stage2 if should_transition else None,
        remaining_steps=max(max_steps - global_step, 0),
    )


def legacy_bridge_supports_config(cfg: Any) -> bool:
    if not cfg.loss.gan:
        return True
    return cfg.train.gan_start_step in (None, 0)


class TrainingLoop:
    def __init__(
        self,
        cfg: Any,
        components: TrainComponents,
        scheduler_factory: Optional[Any] = None,
        debug: Optional[TrainerDebug] = None,
    ) -> None:
        self.cfg = cfg
        self.components = components
        self.scheduler_factory = scheduler_factory
        self.debug = debug or TrainerDebug()

    def train(self) -> int:
        global_step = 0
        best_lpips = 1e10
        max_steps = int(self.cfg.train.max_steps)
        num_epochs = int(self.cfg.train.num_epochs)

        for _epoch in range(num_epochs):
            for batch in self.components.dl_train:
                self._maybe_transition(global_step)
                logs = self._train_batch(batch, global_step)

                if self.components.accelerator.sync_gradients:
                    global_step += 1
                    if self.components.accelerator.is_main_process:
                        self._maybe_save_periodic_checkpoint(global_step)
                    if self._should_evaluate(global_step):
                        best_lpips, logs = self._evaluate(global_step, best_lpips, logs)

                if global_step >= max_steps:
                    return global_step
        return global_step

    def _train_batch(self, batch: dict[str, Any], global_step: int) -> dict[str, float]:
        c = self.components
        accelerator = c.accelerator
        model_gen = c.model_gen
        net_disc = c.net_disc
        ctx = accelerator.accumulate(model_gen, net_disc) if hasattr(accelerator, "accumulate") else nullcontext()

        with ctx:
            x_src = batch["conditioning_pixel_values"]
            x_tgt = batch["output_pixel_values"]
            batch_size = x_src.shape[0]
            batch["pos_prompt"] = [batch["pos_prompt"][i] + ", " + c.args.pos_prompt for i in range(batch_size)]
            batch["neg_prompt"] = [batch["neg_prompt"][i] + ", " + c.args.neg_prompt for i in range(batch_size)]
            mag_prompt = batch["mag_prompt"]

            neg_probs = torch.rand(batch_size).to(accelerator.device)
            mixed_tag_prompt = [
                neg_tag if prob < c.args.neg_prob else pos_tag
                for neg_tag, pos_tag, prob in zip(batch["neg_prompt"], batch["pos_prompt"], neg_probs)
            ]
            is_mix = neg_probs.reshape(batch_size, 1, 1, 1) < c.args.neg_prob
            mixed_tgt = torch.where(is_mix, x_src, x_tgt)

            x_tgt_pred, _, _ = model_gen(x_src.detach(), batch=batch, prompt=mixed_tag_prompt, args=c.args)
            loss_l2 = F.mse_loss(x_tgt_pred.float(), mixed_tgt.detach().float(), reduction="mean") * c.args.lambda_l2
            loss_lpips = c.net_lpips(x_tgt_pred.float(), mixed_tgt.detach().float()).mean() * c.args.lambda_lpips
            loss_pe = c.pe_loss_fn(x_tgt_pred.float(), mixed_tgt.detach().float(), mag_prompt=mag_prompt) * c.args.lambda_pe
            loss = loss_l2 + loss_lpips + loss_pe

            self._backward_step(loss, c.optimizer, c.lr_scheduler, c.layers_to_opt, "gen_opt", "gen_sched")

            logs = {
                "loss_l2": float(loss_l2.detach().item()),
                "loss_lpips": float(loss_lpips.detach().item()),
                "loss_pe": float(loss_pe.detach().item()),
            }

            if self.cfg.loss.gan and self._in_stage2(global_step):
                self.debug.disc_step_at.append(global_step)
                x_tgt_pred, _, _ = model_gen(x_src.detach(), batch=batch, prompt=batch["pos_prompt"], args=c.args)
                loss_g = net_disc(x_tgt_pred, for_G=True).mean() * c.args.lambda_gan
                self._backward_step(loss_g, c.optimizer, c.lr_scheduler, c.layers_to_opt, "gen_opt", "gen_sched")

                loss_d_real = net_disc(x_tgt.detach(), for_real=True).mean() * c.args.lambda_gan
                self._backward_step(
                    loss_d_real.mean(),
                    c.optimizer_disc,
                    c.lr_scheduler_disc,
                    list(net_disc.parameters()),
                    "disc_opt",
                    "disc_sched",
                )

                loss_d_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * c.args.lambda_gan
                self._backward_step(
                    loss_d_fake.mean(),
                    c.optimizer_disc,
                    None,
                    list(net_disc.parameters()),
                    "disc_opt",
                    None,
                )
                logs["loss_G"] = float(loss_g.detach().item())
                logs["loss_D"] = float((loss_d_real + loss_d_fake).detach().item())

        self.debug.last_logs = logs
        return logs

    def _backward_step(
        self,
        loss: torch.Tensor,
        optimizer: Any,
        scheduler: Optional[Any],
        parameters: Any,
        optimizer_label: str,
        scheduler_label: Optional[str],
    ) -> None:
        accelerator = self.components.accelerator
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(parameters, self.components.args.max_grad_norm)
        optimizer.step()
        self.debug.call_order.append(optimizer_label)
        if scheduler is not None:
            scheduler.step()
            if scheduler_label is not None:
                self.debug.call_order.append(scheduler_label)
        optimizer.zero_grad(set_to_none=self.components.args.set_grads_to_none)

    def _maybe_transition(self, global_step: int) -> None:
        transition = build_stage2_transition(self.cfg, global_step)
        if not transition.enabled:
            return

        lr_stage2 = transition.lr_stage2
        if lr_stage2 is not None:
            for param_group in self.components.optimizer.param_groups:
                param_group["lr"] = lr_stage2
            for param_group in self.components.optimizer_disc.param_groups:
                param_group["lr"] = lr_stage2

        self.components.lr_scheduler = self._make_constant_scheduler(self.components.optimizer, global_step)
        self.components.lr_scheduler_disc = self._make_constant_scheduler(self.components.optimizer_disc, global_step)
        self.debug.gen_sched_rebuilt = True
        self.debug.disc_sched_rebuilt = True
        self.debug.stage_transitions.append(global_step)

        if global_step > 0 and self.components.accelerator.is_main_process:
            self._save_checkpoint(global_step, tag=transition.save_tag)
        print(f"[Step {global_step}] Stage 2: LR -> {lr_stage2}, GAN ON")

    def _make_constant_scheduler(self, optimizer: Any, global_step: int) -> Any:
        remaining = max(int(self.cfg.train.max_steps) - global_step, 0)
        total_steps = remaining * self.components.accelerator.num_processes
        if self.scheduler_factory is not None:
            return self.scheduler_factory(
                "constant",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps,
            )

        from diffusers.optimization import get_scheduler

        return get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

    def _maybe_save_periodic_checkpoint(self, global_step: int) -> None:
        if global_step % self.components.args.checkpointing_steps == 1 and global_step >= self.components.args.checkpointing_steps:
            self._save_checkpoint(global_step, tag=f"model_{global_step}")

    def _save_checkpoint(self, global_step: int, tag: str) -> None:
        output = Path(self.components.args.output_dir) / "checkpoints" / f"{tag}.pkl"
        output.parent.mkdir(parents=True, exist_ok=True)
        model = self.components.accelerator.unwrap_model(self.components.model_gen)
        if hasattr(model, "save_model"):
            model.save_model(str(output))
        else:
            torch.save({"global_step": global_step, "tag": tag}, output)

    def _should_evaluate(self, global_step: int) -> bool:
        return global_step % self.components.args.checkpointing_steps == 1 and global_step >= self.components.args.checkpointing_steps

    def _evaluate(self, global_step: int, best_lpips: float, logs: dict[str, float]) -> tuple[float, dict[str, float]]:
        if self.components.evaluator is None or not self.components.dl_val_list:
            return best_lpips, logs
        return self.components.evaluator.evaluate(
            self.components.accelerator.unwrap_model(self.components.model_gen),
            self.components.dl_val_list,
            self.components.net_lpips,
            self.components.pe_loss_fn,
            self.components.args,
            global_step,
            best_lpips,
            logs,
            save_vis=True,
        )

    def _in_stage2(self, global_step: int) -> bool:
        return in_stage2(global_step, self.cfg.train.gan_start_step)


class Trainer:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def _in_stage2(self, global_step: int) -> bool:
        return in_stage2(global_step, self.cfg.train.gan_start_step)

    def run(self) -> None:
        components = self._build_components()
        TrainingLoop(self.cfg, components).train()

    def _build_components(self) -> TrainComponents:
        import diffusers
        import transformers
        from accelerate import Accelerator, DistributedDataParallelKwargs
        from accelerate.utils import ProjectConfiguration, set_seed
        from diffusers.optimization import get_scheduler
        from diffusers.utils.import_utils import is_xformers_available
        from torch.utils.data import DataLoader

        args = build_legacy_args(self.cfg)
        logging_dir = Path(args.output_dir, args.logging_dir)
        project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=project_config,
            kwargs_handlers=[ddp_kwargs],
        )

        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if args.seed is not None:
            set_seed(args.seed)

        if accelerator.is_main_process:
            os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

        device = accelerator.device
        model_name = getattr(self.cfg.model, "builder", "morphodiff_gen")
        dataset_name = getattr(self.cfg.data, "builder", "wsi_sr")

        model_gen = build_model(model_name, args=args)
        model_gen.set_train()
        net_disc = build_discriminator(self.cfg.loss, device=device)
        net_lpips = self._build_lpips(device)
        net_lpips.requires_grad_(False)
        pe_loss_fn = build_pe_loss(self.cfg.loss, device=device)

        model_gen.vae.set_adapter(["default_encoder"])
        model_gen.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                model_gen.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        if args.gradient_checkpointing:
            model_gen.unet.enable_gradient_checkpointing()

        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        layers_to_opt = self._collect_trainable_layers(model_gen)
        optimizer = torch.optim.AdamW(
            layers_to_opt,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
        optimizer_disc = torch.optim.AdamW(
            net_disc.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        lr_scheduler_disc = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

        dataset_train = build_dataset(dataset_name, split="train", data_cfg=self.cfg.data)
        dl_train = DataLoader(
            dataset_train,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
        )
        dl_val_list = [
            DataLoader(
                build_dataset(
                    dataset_name,
                    split=self.cfg.data.val_split,
                    data_cfg=self.cfg.data,
                    mag=spec.mag,
                    scale=spec.scale,
                    max_samples=self.cfg.data.val_max_samples,
                ),
                batch_size=1,
                shuffle=False,
                num_workers=args.dataloader_num_workers,
            )
            for spec in self.cfg.data.val_specs
        ]

        model_gen, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
            model_gen,
            net_disc,
            optimizer,
            optimizer_disc,
            dl_train,
            lr_scheduler,
            lr_scheduler_disc,
        )
        net_lpips = accelerator.prepare(net_lpips)
        pe_loss_fn = accelerator.prepare(pe_loss_fn)
        for name, module in net_disc.named_modules():
            if "attn" in name:
                module.fused_attn = False

        evaluator = Evaluator(
            accelerator=accelerator,
            output_dir=args.output_dir,
            scales=self.cfg.train.val.scales,
            skip_fid=self.cfg.train.val.skip_fid,
            fid_batch_size=self.cfg.train.val.fid_batch_size,
        )

        return TrainComponents(
            accelerator=accelerator,
            model_gen=model_gen,
            net_disc=net_disc,
            optimizer=optimizer,
            optimizer_disc=optimizer_disc,
            lr_scheduler=lr_scheduler,
            lr_scheduler_disc=lr_scheduler_disc,
            dl_train=dl_train,
            dl_val_list=dl_val_list,
            net_lpips=net_lpips,
            pe_loss_fn=pe_loss_fn,
            layers_to_opt=layers_to_opt,
            args=args,
            evaluator=evaluator,
        )

    def _collect_trainable_layers(self, model_gen: Any) -> list[Any]:
        layers_to_opt = []
        for name, parameter in model_gen.unet.named_parameters():
            if "lora" in name:
                layers_to_opt.append(parameter)
        layers_to_opt += list(model_gen.unet.conv_in.parameters())
        for name, parameter in model_gen.vae.named_parameters():
            if "lora" in name:
                layers_to_opt.append(parameter)
        return layers_to_opt

    def _build_lpips(self, device: torch.device) -> Any:
        if getattr(self.cfg.loss, "lpips_type", "vgg") == "stub":
            class _StubLPIPS(torch.nn.Module):
                def forward(self, pred, target):
                    return (pred - target).abs().flatten(1).mean(dim=1) * 0.0

            return _StubLPIPS().to(device)
        import lpips

        lpips_module = lpips
        return lpips_module.LPIPS(net="vgg").to(device)


__all__ = [
    "GAN_UPDATE_SCHEDULE",
    "OptimizerStepSpec",
    "StageTransition",
    "TrainComponents",
    "Trainer",
    "TrainerDebug",
    "TrainingLoop",
    "build_stage2_transition",
    "in_stage2",
    "legacy_bridge_supports_config",
    "should_transition_to_stage2",
]
