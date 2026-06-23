import os
import sys
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from .flow.realnvp import RealNVP
from collections import OrderedDict
try:
    from .density_estimator import density_estimator
except ImportError:
    density_estimator = None

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



# ===========================================================================
# BoltzmannGenerator2D — modern interface compatible with experiments/setup_run.py
# ===========================================================================

class BoltzmannGenerator2D:
    """
    Boltzmann generator trainer for the 2D LJ-bath system.

    This class provides the interface expected by experiments/setup_run.py:
      trainer.train(flow, w_loss, x_samples, z_samples, optimizer, energy_cap, logger)

    The energy_cap parameter is forwarded to model.loss_KL, which applies a soft
    energy cap to prevent numerical blow-up during KL training.
    """

    _defaults = {
        "flow_type": "realnvp",
        'n_blocks': 8,
        'dimension': 64,         # 32 solvent x 2
        'reshape': (32, 2),
        'n_nodes': 256,
        'n_layers': 3,
        'n_epochs': 25,
        'batch_size': 512,
        'lr': 1e-4,
        'prior_sigma': 1.0,
        'patience': 20,
        'min_delta': 1e-4,
        'w_overlap': 0.0,
        "n_hidden": 3,
        "num_bins": 8,
        "tail_bound": 5.0,
        "hidden": 128
    }

    def __init__(self, model_params=None):
        merged = dict(self._defaults)
        if model_params:
            merged.update(model_params)
        self.params = merged
        for key, val in merged.items():
            setattr(self, key, val)
        if not hasattr(self, 'reshape'):
            self.reshape = (self.dimension // 2, 2)
    

    def preprocess_data(self, samples):
        from torch.utils import data as torch_data
        self.n_pts = len(samples)
        if isinstance(samples, torch.Tensor):
            training_set = samples.float()
        else:
            training_set = torch.from_numpy(samples.astype('float32'))
        return torch_data.DataLoader(
            dataset=training_set, batch_size=self.batch_size, num_workers=0
        )


    def build(self, system):
        """Build a RealNVP model for the 2D system."""
        self.system = system
        N, D = self.reshape
        if self.flow_type == 'realnvp':
            print("Building RealNVP flow...")
            from .build_system import build_realnvp_flow
            model = build_realnvp_flow(
                system,
                n_blocks=self.n_blocks,
                n_solvent=N,
                dim=D,
                n_nodes=self.n_nodes,
                n_layers=self.n_layers,
                prior_sigma=self.prior_sigma,
            )
        elif self.flow_type == 'spline' or self.flow_type == 'fab':
            print("Building spline flow...")
            from .build_system import build_spline_flow
            model = build_spline_flow(
                system,
                n_solvent=N,
                dim=D,
                n_blocks=self.n_blocks,
                n_nodes=self.n_nodes,
                n_hidden=self.n_hidden,
                num_bins=self.num_bins,
                tail_bound=self.tail_bound,
                hidden=self.hidden,
            )
        else:
            raise ValueError(f"Unsupported flow_type: {self.flow_type}")
        for key, val in self.params.items():
            setattr(model, key, val)
        return model

    def train(self, model, w_loss, x_samples=None, z_samples=None,
              optimizer=None, energy_cap=None, cuttoff_factor=1.0, logger=None,
              live_plot=False, plot_every=10):
        """
        Train the flow.

        Parameters
        ----------
        model : SoluteSplineFlow or RealNVP
        w_loss : list   [w_ML, w_KL]
        x_samples : np.ndarray or None   MC configs for ML loss
        z_samples : np.ndarray or None   latent samples for KL loss
        optimizer : torch.optim or None   (created internally if None)
        energy_cap : float or None   soft energy cap forwarded to model.loss_KL
        logger : logger or None   receives {'loss', 'epoch', 'iter'} each batch
        live_plot : bool   update a loss curve in-place every plot_every epochs (Jupyter)
        plot_every : int   refresh the plot every this many epochs
        """
        self.w_loss = w_loss

        if optimizer is None:
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=self.lr
            )

        # Build data loaders
        if w_loss[0] != 0:
            if x_samples is None:
                raise ValueError('w_ML is nonzero but no x_samples provided.')
            subdata_x = self.preprocess_data(x_samples)
            if w_loss[1] == 0 and z_samples is None:
                subdata_z = np.zeros(len(subdata_x))
        else:
            subdata_x = None

        if w_loss[1] != 0:
            if z_samples is None:
                raise ValueError('w_KL is nonzero but no z_samples provided.')
            subdata_z = self.preprocess_data(z_samples)
            if w_loss[0] == 0 and x_samples is None:
                subdata_x = [torch.from_numpy(np.zeros(1))] * len(subdata_z)

        # Live plot setup
        if live_plot:
            import matplotlib.pyplot as plt
            try:
                from IPython.display import display,  clear_output
                _in_jupyter = True
            except ImportError:
                _in_jupyter = False

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('Training loss')
            (line,) = ax.plot([], [], lw=1.2)
            if _in_jupyter:
                dh = display(fig, display_id=True)

        loss_ML_val = w_loss[0]
        loss_KL_val = w_loss[1]
        w_overlap_coeff = w_loss[2] if len(w_loss) > 2 else 0.0
        loss_overlap_val = 0.0

        self.loss_iteration = []
        best_loss = float('inf')
        best_state = None
        epochs_no_improve = 0
        global_iter = 0

        for epoch in tqdm(range(self.n_epochs)):
            epoch_losses = []
            for batch_x, batch_z in zip(subdata_x, subdata_z):
                if w_loss[0] != 0:
                    loss_ML_val = model.loss_ML(batch_x)
                if w_loss[1] != 0:
                    loss_KL_val = model.loss_KL(batch_z, w_overlap=0.0,
                                                energy_cap=energy_cap, cutoff_factor=cuttoff_factor)
                if w_overlap_coeff != 0:
                    with torch.no_grad():
                        z_ov = model.prior.sample((self.batch_size,))
                    z_ov = z_ov.detach()
                    loss_overlap_val = model.loss_overlap(z_ov, cutoff_factor=cuttoff_factor)

                loss = (w_loss[0] * loss_ML_val
                        + w_loss[1] * loss_KL_val
                        + w_overlap_coeff * loss_overlap_val)

                self.loss_iteration.append(loss.item())
                epoch_losses.append(loss.item())

                if logger is not None:
                    logger.write({'loss': loss.item(), 'epoch': epoch,
                                  'iter': global_iter})
                global_iter += 1

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                print("Total loss: %s" % loss.item(), end='\r')

            epoch_loss = np.mean(epoch_losses)
            if epoch_loss < best_loss - self.min_delta:
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                print(f'\nEarly stopping at epoch {epoch + 1}')
                break

            # Live plot update
            if live_plot and (epoch + 1) % plot_every == 0:
                iters = list(range(len(self.loss_iteration)))
                line.set_data(iters, self.loss_iteration)
                ax.relim()
                ax.autoscale_view()
                if _in_jupyter:
                    dh.update(fig)
                else:
                    fig.canvas.draw()
                    plt.pause(0.01)

        if live_plot:
            # Remove the temporary live plot after training
            if _in_jupyter:
                clear_output(wait=True)
            else:
                plt.close(fig)

        if best_state is not None:
            model.load_state_dict(best_state)
    
    def save(self, model, save_path, previous_loss=None):
        """
        Save the trained Boltzmann generator and the parameters used for training.

        model : object
            The Boltzmann generator model to be saved. 
        save_path : str
            The directory where the model will be saved (along with the filename).
        previous_loss : list
            A list of previus loss function values.
        """
        # save the trained model
        torch.save(model.state_dict(), save_path)

        # save the parameters for training the model
        for file in os.listdir('.'):
            if file == save_path + '.yml':
                # make sure that the output file is newly made
                os.remove(save_path + '.yml')

        outfile = open(save_path + '.yml', 'a+', newline='')
        outfile.write('# Parameters for training the model\n')
        model_params = copy.deepcopy(vars(self))
        del model_params['params']
        del model_params['loss_iteration']  # use outfile.write instead
        model_params.pop('system', None)    # not serialisable across module paths
        yaml.dump(model_params, outfile, default_flow_style=False)
        outfile.write('\n# Training result\n')

        if previous_loss is not None:
            self.loss_iteration = previous_loss + self.loss_iteration
        outfile.write('loss: ' + str(self.loss_iteration))

    def load(self, model, load_path):
        """
        Loads a trained Boltzmann generator and the training result

        model : objet
            The object of the model to be trained that is built by build method.
            Note that this model must have the same architecture as the trained model
            to be loaded.
        load_path : str
            The directory where the model was saved (along with the filename).
        """
        # load the trained model
        model.load_state_dict(torch.load(load_path))

        # load the training result (loss_iteration)
        f = open(load_path + '.yml', 'r')
        lines = f.readlines()
        f.close()

        loss_found = False
        for line in lines:
            if 'loss: ' in line:
                loss_found = True
                loss_str = line.split('[')[1].split(']')[0].split(',')
                loss = [float(i) for i in loss_str if i.strip()]

        if loss_found is False:
            print("Error! Incomplete results stored in %s" %
                  (load_path + '.yml'))
            sys.exit()

        print('Total loss: %s' % (loss[-1] if loss else 'n/a'))

        return model, loss

