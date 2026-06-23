"""
Training helpers for Boltzmann Generator models.

Provides a single low-level ``train_or_load_model`` function that handles
checkpoint save/load for one model. All looping, naming, and configuration
is done by the caller in the notebook.
"""

import os
import torch


def train_or_load_model(
    BG,
    model,
    model_path,
    x_solvent,
    z_kl,
    w_loss,
    energy_cap=None,
    cuttoff_factor=1.0,
    live_plot=True,
    plot_every=5,
    overwrite=False,
):
    """
    Train or load a single Boltzmann Generator model.

    Parameters
    ----------
    BG : BoltzmannGenerator2D
        Already built (``BG.build(solute_sys)`` called) BG wrapper.
    model : torch model
        Starting model weights (will be trained in-place or overwritten by
        the loaded checkpoint).
    model_path : str
        Full path to save/load the checkpoint.
    x_solvent : np.ndarray
        MC solvent configurations for the ML loss.
    z_kl : np.ndarray or torch.Tensor
        Prior samples for the KL loss.
    w_loss : list of float
        ``[w_ML, w_KL, w_overlap]`` loss weights.
    energy_cap : float or None
        Soft energy cap applied during KL training.
    cuttoff_factor : float
        Cutoff factor passed to the overlap penalty.
    live_plot : bool
        Show a live loss curve during training.
    plot_every : int
        Refresh live plot every this many epochs.
    overwrite : bool
        Re-train even if a checkpoint already exists.

    Returns
    -------
    model : torch model
        Trained (or loaded) model.
    loss : list of float
        Per-iteration loss values recorded during training.
    """
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    if os.path.isfile(model_path) and not overwrite:
        print(f"Loading model from {model_path}")
        model, loss = BG.load(model, model_path)
    else:
        print(f"Training model -> {model_path}")
        BG.train(
            model,
            w_loss=w_loss,
            x_samples=x_solvent,
            z_samples=z_kl,
            live_plot=live_plot,
            plot_every=plot_every,
            energy_cap=energy_cap,
            cuttoff_factor=cuttoff_factor,
        )
        BG.save(model, model_path)
        loss = BG.loss_iteration

    return model, loss


def sample_z_kl(model, n_kl):
    """Draw KL prior samples from a model's prior and return as a numpy array."""
    with torch.no_grad():
        z = model.prior.sample((n_kl,))
    return z.detach().cpu().numpy()
