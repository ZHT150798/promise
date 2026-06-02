from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PromptParts:
    mag_prompt: str
    scale_prompt: str
    q_prompt: Optional[str] = None
    disease_prompt: Optional[str] = None


def magnification_prompt(mag: str) -> str:
    if mag == "40x":
        return "40× magnification"
    if mag == "20x":
        return "20× magnification"
    raise ValueError(f"Unsupported magnification: {mag}")


def scale_prompt(scale: int) -> str:
    if scale in {4, 8, 16}:
        return f"{scale}× super-resolution"
    raise ValueError(f"Unsupported scale: {scale}")


def build_sr_prompt(parts: PromptParts) -> str:
    fields = [parts.mag_prompt, parts.scale_prompt]
    if parts.q_prompt:
        fields.append(parts.q_prompt)
    if parts.disease_prompt:
        fields.append(parts.disease_prompt)
    return ", ".join(fields)


__all__ = ["PromptParts", "build_sr_prompt", "magnification_prompt", "scale_prompt"]
