import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from fab.utils.plotting import plot_contours, plot_marginal_pair
from experiments.setup_run import setup_trainer_and_run_flow, Plotter
from fab.target_distributions.h2o_in_h2o import H2OinH2O
import torch

# Removed params:
# dim = cfg.target.dim,
# n_mixes = cfg.target.n_mixes,
# loc_scaling = cfg.target.loc_scaling,
# log_var_scaling = cfg.target.log_var_scaling,
# use_gpu = cfg.training.use_gpu,


def setup_h2o_plotter(cfg: DictConfig, target: H2OinH2O, buffer=None) -> Plotter:
    raise NotImplementedError("Plotting not implemented for H2OinH2O")


def _run(cfg: DictConfig):
    torch.manual_seed(0)  # seed of 0 for setup. TODO: necessary? Maybe for simulation...
    target = H2OinH2O()
    torch.manual_seed(cfg.training.seed)
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
        target = target.double()
    setup_trainer_and_run_flow(cfg, setup_h2o_plotter, target)


@hydra.main(config_path="./config/", config_name="fab.yaml")
def run(cfg: DictConfig):
    _run(cfg)


if __name__ == "__main__":
    run()
