import platform
import subprocess
import sys
from pathlib import Path
from typing import Union


def write_environment_report(output_dir: Union[str, Path]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = output_path / "environment.txt"

    lines = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
    ]
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as exc:
        freeze = f"pip freeze failed: {exc}"

    report.write_text("\n".join(lines) + "\n\n[pip freeze]\n" + freeze, encoding="utf-8")


__all__ = ["write_environment_report"]
