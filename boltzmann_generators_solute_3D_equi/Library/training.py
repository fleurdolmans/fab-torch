import os
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils import data
from torch import distributions
from collections import OrderedDict

from generator import RealNVP, RealNVPCUDA

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())


def setup_yaml():
    yaml.add_representer(OrderedDict, represent_dictionary_order)


setup_yaml()


class BoltzmannGenerator3D:
    """
    Boltzmann generator for 3D particle systems.

    The flat input dimension is ``N * 3`` (N particles, 3 spatial dimensions).
    A checkerboard mask splits the flat vector into two halves for each
    affine coupling block.

    Parameters
    ----------
    model_params : dict, optional
        Keys: n_blocks, dimension (= N*3), n_nodes, n_layers, n_epochs,
              batch_size, LR, prior_sigma.
        ``reshape`` must also be provided: a tuple (N, 3).
    """

    defaults = {
        'n_blocks': 8,
        'dimension': 108,      # 36 solvent particles × 3  (solute system default)
        'reshape': (36, 3),
        'n_nodes': 256,
        'n_layers': 3,
        'n_epochs': 200,
        'batch_size': 512,
        'LR': 1e-4,
        'prior_sigma': 1.0,
        'w_overlap': 0.0,      # overlap penalty weight (0 = disabled)
        'patience': 50,
        'min_delta': 1e-4,
    }

    def __init__(self, model_params=None):
        params = dict(self.defaults)
        if model_params is not None:
            params.update(model_params)
        self.params = params
        for key, val in params.items():
            setattr(self, key, val)

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def affine_layers(self):
        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layers.append(nn.Linear(self.dimension, self.n_nodes))
            elif i == self.n_layers - 1:
                layers.append(nn.Linear(self.n_nodes, self.dimension))
            else:
                layers.append(nn.Linear(self.n_nodes, self.n_nodes))
            if i != self.n_layers - 1:
                layers.append(nn.ReLU())
        return layers

    def build_networks(self):
        self.s_net = lambda: nn.Sequential(*self.affine_layers(), nn.Tanh())
        self.t_net = lambda: nn.Sequential(*self.affine_layers())

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build(self, system):
        """
        Build and return an untrained RealNVP model.

        Parameters
        ----------
        system : potentials.DimerSimulation3D

        Returns
        -------
        model : RealNVP
        """
        self.system = system
        self.build_networks()

        # Checkerboard mask: first half ones, second half zeros, then flipped
        half = self.dimension // 2
        row_on = np.concatenate([np.ones(half), np.zeros(self.dimension - half)])
        row_off = 1 - row_on
        mask = np.array([row_on, row_off] * self.n_blocks, dtype=np.float32)
        self.mask = torch.from_numpy(mask)

        self.prior = distributions.MultivariateNormal(
            torch.zeros(self.dimension),
            torch.eye(self.dimension) * self.prior_sigma,
        )

        model = RealNVP(
            self.s_net, self.t_net, self.mask,
            self.prior, system, self.reshape,
        )
        for key, val in self.params.items():
            setattr(model, key, val)
        return model

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def preprocess_data(self, samples):
        """
        Wrap a numpy array (n_samples, dimension) in a DataLoader.
        """
        self.n_pts = len(samples)
        tensor = torch.from_numpy(samples.astype('float32'))
        return data.DataLoader(dataset=tensor, batch_size=self.batch_size,
                               num_workers=0, shuffle=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, model, w_loss, x_samples=None, z_samples=None,
              optimizer=None, energy_cap=None, patience=None, min_delta=1e-4,
              live_plot=False, plot_every=1):
        """
        Train the model.

        Parameters
        ----------
        model : RealNVP or CartesianSplineFlow
        w_loss : list [w_ML, w_KL]
            Weights for the forward-KL (ML) and reverse-KL losses.
        x_samples : np.ndarray, shape (n, dimension)
            Configuration-space training data (required if w_ML != 0).
        z_samples : np.ndarray, shape (n, dimension)
            Latent-space samples (required if w_KL != 0).
            Typically drawn from ``model.prior.sample((n,))``.
        optimizer : torch.optim, optional
            Defaults to Adam with self.LR.
        energy_cap : float or None
            If set, energies above this value are soft-capped in loss_KL
            to prevent explosion from overlapping configurations.
            A good starting value is 10–50× the typical MC energy.
        patience : int or None
            Early stopping: stop after this many epochs with no improvement
            greater than min_delta.  None disables early stopping.
        min_delta : float
            Minimum improvement in epoch loss to reset the patience counter.
        live_plot : bool
            If True, update a loss curve plot in-place after every
            ``plot_every`` epochs (requires a Jupyter / IPython environment).
        plot_every : int
            How many epochs to wait between live-plot updates.
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.LR,
            )

        if live_plot:
            try:
                from IPython.display import display, clear_output
                import matplotlib.pyplot as plt
                _ipython_display = display
                _ipython_clear = clear_output
            except ImportError:
                live_plot = False  # silently fall back if not in Jupyter

        # Build data loaders
        if w_loss[0] != 0:
            if x_samples is None:
                raise ValueError("w_ML != 0 but x_samples not provided.")
            loader_x = self.preprocess_data(x_samples)
        else:
            loader_x = None

        if w_loss[1] != 0:
            if z_samples is None:
                raise ValueError("w_KL != 0 but z_samples not provided.")
            loader_z = self.preprocess_data(z_samples)
        else:
            loader_z = None

        self.loss_iteration = []
        self.epoch_losses = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in tqdm(range(self.n_epochs)):
            # pair up the two loaders (cycle the shorter one)
            if loader_x is not None and loader_z is not None:
                pairs = zip(loader_x, loader_z)
            elif loader_x is not None:
                pairs = ((bx, None) for bx in loader_x)
            else:
                pairs = ((None, bz) for bz in loader_z)

            epoch_batch_losses = []
            for batch_x, batch_z in pairs:
                loss = torch.tensor(0.0)

                if w_loss[0] != 0 and batch_x is not None:
                    loss = loss + w_loss[0] * model.loss_ML(batch_x)
                if w_loss[1] != 0 and batch_z is not None:
                    loss = loss + w_loss[1] * model.loss_KL(
                        batch_z,
                        energy_cap=energy_cap,
                        w_overlap=getattr(self, 'w_overlap', 0.0),
                    )

                self.loss_iteration.append(loss.item())
                epoch_batch_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                print(f"Loss: {loss.item():.4f}", end='\r')

            epoch_loss = sum(epoch_batch_losses) / len(epoch_batch_losses)
            self.epoch_losses.append(epoch_loss)

            # Live loss plot
            if live_plot and (epoch + 1) % plot_every == 0:
                _ipython_clear(wait=True)
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(self.epoch_losses, lw=1.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'Training loss  (epoch {epoch + 1}/{self.n_epochs})')
                ax.set_yscale('symlog')
                plt.tight_layout()
                plt.show()

            # Early stopping
            if patience is not None:
                if epoch_loss < best_loss - min_delta:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        tqdm.write(
                            f'\nEarly stopping at epoch {epoch + 1} '
                            f'(no improvement for {patience} epochs).'
                        )
                        break

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_loss(self, use_epoch=True, smoothing=1, figsize=(8, 4)):
        """Plot the training loss curve.

        Parameters
        ----------
        use_epoch : bool
            If True, plot per-epoch average losses (smoother).
            If False, plot raw per-batch losses.
        smoothing : int
            Rolling-average window applied to per-batch losses when
            use_epoch=False.  Ignored when use_epoch=True.
        figsize : tuple
        """
        import matplotlib.pyplot as plt

        if use_epoch and hasattr(self, 'epoch_losses') and self.epoch_losses:
            losses = self.epoch_losses
            xlabel = 'Epoch'
        else:
            losses = self.loss_iteration
            xlabel = 'Batch iteration'
            if smoothing > 1:
                kernel = np.ones(smoothing) / smoothing
                losses = np.convolve(losses, kernel, mode='valid').tolist()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(losses)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Loss')
        ax.set_title('Training loss')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model, save_path, previous_loss=None):
        torch.save(model.state_dict(), save_path)

        yml_path = save_path + '.yml'
        if os.path.isfile(yml_path):
            os.remove(yml_path)

        with open(yml_path, 'a+', newline='') as f:
            f.write('# Parameters for training the model\n')
            saveable = {k: v for k, v in vars(self).items()
                        if k not in ('params', 's_net', 't_net', 'mask', 'prior',
                                     'loss_iteration', 'system')}
            yaml.dump(saveable, f, default_flow_style=False)
            f.write('\n# Training result\n')
            losses = (previous_loss or []) + self.loss_iteration
            f.write('loss: ' + str(losses))

    def load(self, model, load_path):
        model.load_state_dict(torch.load(load_path, map_location='cpu'))

        with open(load_path + '.yml', 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'loss: ' in line:
                loss_str = line.split('[')[1].split(']')[0].split(',')
                loss = [float(x) for x in loss_str]
                print(f'Loaded model — final loss: {loss[-1]:.4f}')
                return model, loss

        raise RuntimeError(f"No loss data found in {load_path}.yml")


# ---------------------------------------------------------------------------
# GPU variant
# ---------------------------------------------------------------------------

class BoltzmannGenerator3DCUDA(BoltzmannGenerator3D):
    """GPU version — moves masks, prior, and model to ``device``."""

    def affine_layers(self):
        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layers.append(nn.Linear(self.dimension, self.n_nodes).to(device))
            elif i == self.n_layers - 1:
                layers.append(nn.Linear(self.n_nodes, self.dimension).to(device))
            else:
                layers.append(nn.Linear(self.n_nodes, self.n_nodes).to(device))
            if i != self.n_layers - 1:
                layers.append(nn.ReLU().to(device))
        return layers

    def build_networks(self):
        self.s_net = lambda: nn.Sequential(*self.affine_layers(), nn.Tanh()).to(device)
        self.t_net = lambda: nn.Sequential(*self.affine_layers()).to(device)

    def build(self, system):
        self.system = system
        self.build_networks()

        half = self.dimension // 2
        row_on = np.concatenate([np.ones(half), np.zeros(self.dimension - half)])
        row_off = 1 - row_on
        mask = np.array([row_on, row_off] * self.n_blocks, dtype=np.float32)
        self.mask = torch.from_numpy(mask).to(device)

        self.prior = distributions.MultivariateNormal(
            torch.zeros(self.dimension).to(device),
            (torch.eye(self.dimension) * self.prior_sigma).to(device),
        )

        model = RealNVPCUDA(
            self.s_net, self.t_net, self.mask,
            self.prior, system, self.reshape,
        ).to(device)
        for key, val in self.params.items():
            setattr(model, key, val)
        return model

    def train(self, model, w_loss, x_samples=None, z_samples=None,
              optimizer=None, energy_cap=None, patience=None, min_delta=1e-4,
              live_plot=False, plot_every=1):
        if optimizer is None:
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.LR,
            )

        if live_plot:
            try:
                from IPython.display import display, clear_output
                import matplotlib.pyplot as plt
                _ipython_display = display
                _ipython_clear = clear_output
            except ImportError:
                live_plot = False

        if w_loss[0] != 0:
            if x_samples is None:
                raise ValueError("w_ML != 0 but x_samples not provided.")
            loader_x = self.preprocess_data(x_samples)
        else:
            loader_x = None

        if w_loss[1] != 0:
            if z_samples is None:
                raise ValueError("w_KL != 0 but z_samples not provided.")
            loader_z = self.preprocess_data(z_samples)
        else:
            loader_z = None

        self.loss_iteration = []
        self.epoch_losses = []
        best_loss = float('inf')
        patience_counter = 0

        for epoch in tqdm(range(self.n_epochs)):
            if loader_x is not None and loader_z is not None:
                pairs = zip(loader_x, loader_z)
            elif loader_x is not None:
                pairs = ((bx, None) for bx in loader_x)
            else:
                pairs = ((None, bz) for bz in loader_z)

            epoch_batch_losses = []
            for batch_x, batch_z in pairs:
                loss = torch.tensor(0.0, device=device)

                if w_loss[0] != 0 and batch_x is not None:
                    loss = loss + w_loss[0] * model.loss_ML(batch_x.to(device))
                if w_loss[1] != 0 and batch_z is not None:
                    loss = loss + w_loss[1] * model.loss_KL(
                        batch_z.to(device),
                        energy_cap=energy_cap,
                        w_overlap=getattr(self, 'w_overlap', 0.0),
                    )

                self.loss_iteration.append(loss.item())
                epoch_batch_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                print(f"Loss: {loss.item():.4f}", end='\r')

            epoch_loss = sum(epoch_batch_losses) / len(epoch_batch_losses)
            self.epoch_losses.append(epoch_loss)

            # Live loss plot
            if live_plot and (epoch + 1) % plot_every == 0:
                _ipython_clear(wait=True)
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(self.epoch_losses, lw=1.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'Training loss  (epoch {epoch + 1}/{self.n_epochs})')
                ax.set_yscale('symlog')
                plt.tight_layout()
                plt.show()

            if patience is not None:
                if epoch_loss < best_loss - min_delta:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        tqdm.write(
                            f'\nEarly stopping at epoch {epoch + 1} '
                            f'(no improvement for {patience} epochs).'
                        )
                        break
