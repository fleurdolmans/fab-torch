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
    def plot(fab_model, n_samples: int = cfg.training.batch_size, dim: int = cfg.target.dim):
        print("  Plotting not yet implemented for H2OinH2O...")
        return []

    return plot


def _run(cfg: DictConfig):
    torch.manual_seed(0)  # seed of 0 for setup. TODO: necessary? Maybe for simulation...
    if cfg.target.solute == "water" and cfg.target.solvent == "water":
        assert cfg.target.dim % 3 == 0, "Dimension must be divisible by 3 for water in water."
        target = H2OinH2O(
            dim=cfg.target.dim,
            temperature=cfg.target.temperature,
            energy_cut=cfg.target.energy_cut,
            energy_max=cfg.target.energy_max,
            n_threads=cfg.target.n_threads,
            device="cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu",
            target_samples_path=cfg.target.samples_path,
            save_dir=cfg.evaluation.save_path,
        )
    else:
        raise NotImplementedError("Solute/solvent combination not implemented.")
    torch.manual_seed(cfg.training.seed)
    if cfg.training.use_64_bit:
        torch.set_default_dtype(torch.float64)
        target = target.double()
    setup_trainer_and_run_flow(cfg, setup_h2o_plotter, target)


@hydra.main(config_path="./config/", config_name="h2oinh2o_fab_pbuff.yaml", version_base="1.1")
def run(cfg: DictConfig):
    _run(cfg)


if __name__ == "__main__":
    run()
