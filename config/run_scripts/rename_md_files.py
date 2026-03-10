import hydra
from omegaconf import DictConfig
import pathlib


def _run(cfg: DictConfig):
    data_dir = pathlib.Path(cfg.data_dir) / "md"
    for filepath in data_dir.iterdir():
        filename = filepath.stem
        if "_fc" in filename:
            continue
        else:
            # TODO
            return


# Run with hydra node configuration.
@hydra.main(config_path="../experiments/solvation/config/node/", config_name="desktop.yaml", version_base="1.1")
def run(cfg: DictConfig) -> None:
    _run(cfg)


if __name__ == "__main__":
    run()