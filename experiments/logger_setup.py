from fab.utils.logging import PandasLogger, WandbLogger, Logger, ListLogger
from omegaconf import DictConfig


def setup_logger(cfg: DictConfig, save_path: str) -> Logger:
    if hasattr(cfg.logger, "pandas_logger"):
        logger = PandasLogger(
            save=True, save_path=save_path + "logging_hist.csv", save_period=cfg.logger.pandas_logger.save_period
        )
    elif hasattr(cfg.logger, "wandb"):
        logger = WandbLogger(**cfg.logger.wandb, config=dict(cfg))
    elif hasattr(cfg.logger, "list_logger"):
        logger = ListLogger(save_path=save_path + "logging_hist.pkl")
    else:
        raise Exception("No logger specified, try adding the wandb or " "pandas logger to the config file.")
    return logger