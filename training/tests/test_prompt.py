from promise_train.data.prompt import PromptParts, build_sr_prompt, magnification_prompt, scale_prompt


def test_prompt_format_matches_reference() -> None:
    prompt = build_sr_prompt(
        PromptParts(
            mag_prompt=magnification_prompt("40x"),
            scale_prompt=scale_prompt(4),
        )
    )

    assert prompt == "40× magnification, 4× super-resolution"
