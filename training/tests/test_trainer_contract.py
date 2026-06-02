from types import SimpleNamespace

from promise_train.trainer.trainer import (
    GAN_UPDATE_SCHEDULE,
    build_stage2_transition,
    in_stage2,
    legacy_bridge_supports_config,
    should_transition_to_stage2,
)


def test_gan_update_schedule_matches_reference() -> None:
    assert [(step.name, step.optimizer, step.scheduler_step) for step in GAN_UPDATE_SCHEDULE] == [
        ("reconstruction", "generator", True),
        ("gan_generator", "generator", True),
        ("discriminator_real", "discriminator", True),
        ("discriminator_fake", "discriminator", False),
    ]


def test_stage2_detection() -> None:
    assert in_stage2(global_step=99999, gan_start_step=100000) is False
    assert in_stage2(global_step=100000, gan_start_step=100000) is True
    assert in_stage2(global_step=0, gan_start_step=0) is True
    assert in_stage2(global_step=100000, gan_start_step=None) is False


def test_stage2_transition_only_fires_once() -> None:
    assert should_transition_to_stage2(global_step=99999, gan_start_step=100000) is False
    assert should_transition_to_stage2(global_step=100000, gan_start_step=100000) is True
    assert should_transition_to_stage2(global_step=100001, gan_start_step=100000) is False
    assert should_transition_to_stage2(global_step=0, gan_start_step=None) is False


def test_stage2_transition_payload() -> None:
    cfg = SimpleNamespace(
        train=SimpleNamespace(
            gan_start_step=100000,
            lr_stage2=2e-5,
            max_steps=150000,
        )
    )

    transition = build_stage2_transition(cfg, global_step=100000)

    assert transition.enabled is True
    assert transition.lr_stage2 == 2e-5
    assert transition.remaining_steps == 50000
    assert transition.save_tag == "stage1_final"


def test_legacy_bridge_rejects_staged_gan_config() -> None:
    cfg = SimpleNamespace(
        loss=SimpleNamespace(gan=True),
        train=SimpleNamespace(gan_start_step=100000),
    )

    assert legacy_bridge_supports_config(cfg) is False


def test_legacy_bridge_allows_non_staged_modes() -> None:
    no_gan_cfg = SimpleNamespace(
        loss=SimpleNamespace(gan=False),
        train=SimpleNamespace(gan_start_step=100000),
    )
    gan_from_start_cfg = SimpleNamespace(
        loss=SimpleNamespace(gan=True),
        train=SimpleNamespace(gan_start_step=0),
    )
    gan_disabled_by_stage_cfg = SimpleNamespace(
        loss=SimpleNamespace(gan=True),
        train=SimpleNamespace(gan_start_step=None),
    )

    assert legacy_bridge_supports_config(no_gan_cfg) is True
    assert legacy_bridge_supports_config(gan_from_start_cfg) is True
    assert legacy_bridge_supports_config(gan_disabled_by_stage_cfg) is True
