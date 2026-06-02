from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from promise_train.trainer import Trainer
from promise_train.utils.env import write_environment_report


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_environment_report(output_dir)
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (output_dir / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
    Trainer(cfg).run()


if __name__ == "__main__":
    main()
