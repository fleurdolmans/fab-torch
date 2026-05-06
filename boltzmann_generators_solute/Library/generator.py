import sys
import copy
import numpy as np 
import torch
import torch.nn as nn

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)


class RealNVP(nn.Module):  # inherit from nn.Module
    def __init__(self, s_net, t_net, mask, prior, system, sys_dim):
        """
        This is the init function for RealNVP class.

        Parameters
        ----------
        s_net : lambda function
            The scaling function as a neural network in the anonymous form. 
        t_net : lambda function
            The translation functione as a neural network in the anonymous form.
        mask : torch.Tensor
            The masking scheme for affine coupling layers. 
        prior : torch.distributions object
            The prior probability distribution in the latent space (generally a normal distribution)
        system : object
            The object of the system of interest. (For example, DoubleWellPotential)
        sys_dim : tuple
            The dimensionality of the system

        Attributes
        ----------
        prior       The prior probability distribution in the latent space (generally a normal distribution)
        mask        The masking scheme for affine coupling layers.
        s           The scaling function/network in the affine coupling layers for all the NVP blocks.
        t           The translation function/network in the affine coupling layers for all the NVP blocks.
        sys_dim     The dimensionality of the system
        """
        super().__init__()  # nn.Module.__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)  # could try requires_grad=False
        self.t = torch.nn.ModuleList([t_net() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([s_net() for _ in range(len(mask))])

        # nn.ModuleList is basically just like a Python list, used to store a desired number of nn.Module's.
        # Also note that t[i] and s[i] are entire sequences of operations
        self.system = system   # class of what molecular system are we considering.
        self.sys_dim = sys_dim # tuple describing original dim.
        # Scale factor applied to the coupling-layer s output.  Default 1.0 = no change.
        # Reduce (e.g. 0.5) to prevent scale blowup during KL training; set via model config.
        self.s_scale = 1.0

    def inverse_generator(self, x, process=False):
        """
        Inverse Boltzmann generator which transforms the samples drawn from the probability distribution
        in the configuration space to the real space. (Fxz in the Boltzmann generator paper)

        Parameters
        ---------
        x : torch.Tensor
            Samples drawn from the probability distribution in the configuration space

        Returns
        -------
        z : torch.Tensor
            Samples generated in the latent space
        log_R_xz : torch.Tensor
            Log of the determinant of the Jacobian of Fxz (invese generator)
        """
        z = x   # just for initialization
        log_R_xz = x.new_zeros(x.shape[0])
        # new_zeros(size) returns a tensor of size "size" filled with 0s
        # z_list = []
        # z_list.append(copy.deepcopy(x.detach().numpy()))
        for i in reversed(range(len(self.t))):   # move backwards through the layers
            # here we split the dataset into two channels (with 1:d and d+1:D dimensions)
            # See equation (9) in the original RealNVP papaer
            z_ = self.mask[i] * z    # b * x in equation (9)
            s = self.s[i](z_) * (1 - self.mask[i]) * self.s_scale  # s(b * x) in equation (9)
            t = self.t[i](z_) * (1 - self.mask[i])       # t(b * x) in equation (9)
            #z = z_ + (1 - self.mask[i]) * (z - t) * torch.exp(-s) +    # equation (9)
            #log_R_xz -= torch.sum(s, -1)
            z = z_ + (1 - self.mask[i]) * (z * torch.exp(s) + t)
            log_R_xz += torch.sum(s, -1)
            if process is True:
                z_list.append(copy.deepcopy(z.detach().numpy()))

        if process is False:
            return z, log_R_xz
        else:
            return z, log_R_xz, z_list

    def generator(self, z, process=False):
        """ 
        Boltzmann generator which transforms the samples drawn from the probability distribution 
        in the latent space to the configuration space. (Fzx in the Boltzmann generator paper)
        
        Parameters
        ----------
        z : torch.Tensor
            Samples drawn from the the probability distribution in the latent space

        Returns
        -------
        x : torch.Tensor
            Samples generated in the configuration space
        log_R_zx : torch.Tensore
            Log of the determinant of the Jacobian of Fzx (generator)
        """
        x = z   # just for initialization
        log_R_zx = z.new_zeros(z.shape[0])
        # new_zeros(size) returns a tensor of size "size" filled with 0s
        x_list = []
        # x_list.append(copy.deepcopy(z.detach().numpy()))

        for  i in range(len(self.t)):
            # here we split the dataset into two channels (with 1:d and d+1:D dimensions)
            # See equation (9) in the original RealNVP papaer
            x_ = self.mask[i] * x         # b * x in equation (9)
            s = self.s[i](x_) * (1 - self.mask[i]) * self.s_scale  # s(b * x) in equation (9)
            t = self.t[i](x_) * (1 - self.mask[i])            # t(b * x) in equation (9)
            #x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)   # equation (9)
            #log_R_zx += torch.sum(s, -1)
            x = x_ +  (1 - self.mask[i]) * (x - t) * torch.exp(-s)
            log_R_zx -= torch.sum(s, -1)    # equation (6)
            if process is True:
                x_list.append(copy.deepcopy(x.detach().numpy()))

        if process is True:
            return x, log_R_zx, x_list
        else:
            return x, log_R_zx
        
    def loss_ML(self, batch_x, weighted=False):
        """
        Calculates   the loss function when training by example (samples from the configuration space)
        J_ML = E[u_z(z) - log Rxz(x)], where u_z(z) = 0.5 * /(sigma^{2}) * z^{2} (sigma = 1)

        Parameters
        ----------
        batch_x : torch.Tensor
            A batch of samples in the configuration space.
        weighte : bool
            Whether the samples are already Boltzmann-weighted, i.e. drawn from MC simulations 
            without removing duplicates of configurations

        Returns
        -------
        J_ml : torch.Tensor
            The loss function J_ML
        """
        z, log_R_xz = self.inverse_generator(batch_x)
        u_z = self.calculate_energy(z, space='latent') 
        if weighted is not True:
            J_ml = self.expectation(u_z - log_R_xz)
        else:
            u_x = self.calculate_energy(batch_x, space='configuration')  
            weights_x = torch.exp(-u_x) 
            J_ml = self.expectation(u_z - log_R_xz, weights=weights_x)

        return J_ml

    def loss_KL(self, batch_z, weighted=False):
        """
        Calculates the loss function when training by energy (samples from the latent space)
        J_KL = E[u_x(x) - log Rzx(z)]

        Parameters
        ----------
        batch_z : torch.Tensor
            A batch of samples in the latent space

        Returns
        -------
        J_kl : torch.Tensor
            The loss function J_KL
        """
        x, log_R_zx = self.generator(batch_z)
        u_x = self.calculate_energy(x, space='configuration')   # we need this to calculate J_kl
        if weighted is not True:
            J_kl = self.expectation(u_x - log_R_zx)
        else:
            u_z = self.calculate_energy(batch_z, space='latent')
            weights_z = torch.exp(-u_z)  
            J_kl = self.expectation(u_x - log_R_zx, weights=weights_z)

        return J_kl

    def loss_RC(self, batch_RC, estimator, weighted=False):
        """
        Calculates the reaction coordinate loss function. J_RC = E[logp(RC)].

        Parameters
        ----------
        batch_RC : np.array
            A batch of samples along the reaction coordinate (in the configuration space).
        estimator : sklearn.neighbors.KernelDensity object
            A kernel density estimator to estimate the probability of the samples
        
        Returns
        -------
        J_rc : torch.Tensor
            The loss function J_RC

        Note
        ----
        At the current stage, this method might only work for DWP.
        """
        log_p = estimator.score_samples(batch_RC[:, 0][:, None])
        if weighted is not True:
            J_rc = self.expectation(log_p)
        else:
            u_rc = self.calculate_energy(batch_RC, space='configuration')
            weights_rc = torch.exp(-u_rc)
            J_rc = self.expectation(log_p, weights=weights_rc)

        return J_rc

    def calculate_energy(self, batch, space):
        """
        Calculate the energy of each each configuration in a batch of dataset.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of configurations
        
        Returns
        -------
        energy : torch.Tensor
            The energies of the configurations
        space : str
            Whether to calcualte the energy in the real space (x) or the 
            latent space (z). Available options: 'latent' or 'configuration'.
        """

        e_high, e_max = 10 ** 4, 10 ** 20
        energy = batch.new_zeros(batch.shape[0])  # like np.zeros, same length as batch_data

        if space == 'configuration':
            for i in range(batch.shape[0]):  # for each data point in the dataset
                config = batch[i, :].reshape(self.sys_dim)  # ensure correct dimensionality
                energy[i] = self.regularize_energy(self.system.get_energy(config))
        elif space == 'latent':
            for i in range(batch.shape[0]):  # for each data point in the dataset
                config = batch[i, :].reshape(self.sys_dim)  # ensure correct dimensionality
                # for 2D Gaussian distribution, u(z) = (1 / (2*sigma **2)) * z ** 2
                # in our case, sigma =1 and z ** 2 = z[0] ** 2 + z[1] ** 2
                energy[i] = self.regularize_energy(0.5 * torch.sum(config ** 2))
        else:
            print("Error! Unavailable option of parameter 'space' specificed.")
            sys.exit()
        return energy

    def regularize_energy(self, energy, e_high=10**8, e_max=10**20):
        if not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy, device=device, dtype=torch.float32)

        if energy.item() > e_max:
            cap = torch.tensor(e_max - e_high + 1.0, device=energy.device, dtype=energy.dtype)
            energy = torch.tensor(e_high, device=energy.device, dtype=energy.dtype) + torch.log10(cap)
        elif energy.item() > e_high:
            energy = torch.tensor(e_high, device=energy.device, dtype=energy.dtype) + torch.log10(
                energy - e_high + 1.0
            )
        return energy

    # def regularize_energy(self, energy, e_high = 10 ** 8, e_max = 10 ** 20):
    #     if energy.item() > e_high:
    #         energy = e_high + torch.log10(energy - e_high + 1)
    #     elif energy.item() > e_max:
    #         energy= e_high + torch.log10(e_max - e_high + 1)
    #     return energy


    def expectation(self, observable, weights = None):
        """ 
        Calculate the expectation value of an observable
        
        Parameters
        ----------
        observable : torch.Tensor
            Observable of interest.

        Returns
        -------
        e : torch.Tensor
            Expectation value as a one-element tensor
        """
        # e = torch.dot(observable, weights) / torch.sum(weights) #the same as below
        if weights is None:
            e = observable.mean()
        else:
            e = torch.sum(observable * weights) / torch.sum(weights)
        return e

class RealNVPCUDA(nn.Module):  # inherit from nn.Module
    def __init__(self, s_net, t_net, mask, prior, system, sys_dim):
        """
        This is the init function for RealNVP class.

        Parameters
        ----------
        s_net : lambda function
            The scaling function as a neural network in the anonymous form. 
        t_net : lambda function
            The translation functione as a neural network in the anonymous form.
        mask : torch.Tensor
            The masking scheme for affine coupling layers. 
        prior : torch.distributions object
            The prior probability distribution in the latent space (generally a normal distribution)
        system : object
            The object of the system of interest. (For example, DoubleWellPotential)
        sys_dim : tuple
            The dimensionality of the system

        Attributes
        ----------
        prior       The prior probability distribution in the latent space (generally a normal distribution)
        mask        The masking scheme for affine coupling layers.
        s           The scaling function/network in the affine coupling layers for all the NVP blocks.
        t           The translation function/network in the affine coupling layers for all the NVP blocks.
        sys_dim     The dimensionality of the system
        """

        super().__init__()  # nn.Module.__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False).to(device)  # could try requires_grad=False
        self.t = torch.nn.ModuleList([t_net() for _ in range(len(mask))]).to(device)
        self.s = torch.nn.ModuleList([s_net() for _ in range(len(mask))]).to(device)

        # nn.ModuleList is basically just like a Python list, used to store a desired number of nn.Module's.
        # Also note that t[i] and s[i] are entire sequences of operations
        self.system = system   # class of what molecular system are we considering.
        self.sys_dim = sys_dim # tuple describing original dim.
        # Scale factor applied to the coupling-layer s output.  Default 1.0 = no change.
        self.s_scale = 1.0

    def inverse_generator(self, x, process=False):
        """
        Inverse Boltzmann generator which transforms the samples drawn from the probability distribution
        in the configuration space to the real space. (Fxz in the Boltzmann generator paper)

        Parameters
        ---------
        x : torch.Tensor
            Samples drawn from the probability distribution in the configuration space

        Returns
        -------
        z : torch.Tensor
            Samples generated in the latent space
        log_R_xz : torch.Tensor
            Log of the determinant of the Jacobian of Fxz (invese generator)
        """
        z = x   # just for initialization
        log_R_xz = x.new_zeros(x.shape[0])
        # new_zeros(size) returns a tensor of size "size" filled with 0s
        # z_list = []
        # z_list.append(copy.deepcopy(x.cpu().detach().numpy()))
        for i in reversed(range(len(self.t))):   # move backwards through the layers
            # here we split the dataset into two channels (with 1:d and d+1:D dimensions)
            # See equation (9) in the original RealNVP papaer
            z_ = self.mask[i] * z    # b * x in equation (9)
            s = self.s[i](z_) * (1 - self.mask[i]) * self.s_scale  # s(b * x) in equation (9)
            t = self.t[i](z_) * (1 - self.mask[i])       # t(b * x) in equation (9)
            #z = z_ + (1 - self.mask[i]) * (z - t) * torch.exp(-s) +    # equation (9)
            #log_R_xz -= torch.sum(s, -1)
            z = z_ + (1 - self.mask[i]) * (z * torch.exp(s) + t)
            log_R_xz += torch.sum(s, -1)
            if process is True:
                z_list.append(copy.deepcopy(z.detach().numpy()))

        # if process is False:
        return z, log_R_xz
        # else:
        #    return z, log_R_xz, z_list

    def generator(self, z, process=False):
        """ 
        Boltzmann generator which transforms the samples drawn from the probability distribution 
        in the latent space to the configuration space. (Fzx in the Boltzmann generator paper)
        
        Parameters
        ----------
        z : torch.Tensor
            Samples drawn from the the probability distribution in the latent space

        Returns
        -------
        x : torch.Tensor
            Samples generated in the configuration space
        log_R_zx : torch.Tensore
            Log of the determinant of the Jacobian of Fzx (generator)
        """
        x = z   # just for initialization
        log_R_zx = z.new_zeros(z.shape[0])
        # new_zeros(size) returns a tensor of size "size" filled with 0s
        # x_list = []
        # x_list.append(copy.deepcopy(z.detach().numpy()))

        for  i in range(len(self.t)):
            # here we split the dataset into two channels (with 1:d and d+1:D dimensions)
            # See equation (9) in the original RealNVP papaer
            x_ = self.mask[i] * x         # b * x in equation (9)
            s = self.s[i](x_) * (1 - self.mask[i]) * self.s_scale  # s(b * x) in equation (9)
            t = self.t[i](x_) * (1 - self.mask[i])            # t(b * x) in equation (9)
            #x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)   # equation (9)
            #log_R_zx += torch.sum(s, -1)
            x = x_ +  (1 - self.mask[i]) * (x - t) * torch.exp(-s)
            log_R_zx -= torch.sum(s, -1)    # equation (6)
            if process is True:
                x_list.append(copy.deepcopy(x.detach().numpy()))

        return x, log_R_zx
        
    def loss_ML(self, batch_x, weighted=False):
        """
        Calculates   the loss function when training by example (samples from the configuration space)
        J_ML = E[u_z(z) - log Rxz(x)], where u_z(z) = 0.5 * /(sigma^{2}) * z^{2} (sigma = 1)

        Parameters
        ----------
        batch_x : torch.Tensor
            A batch of samples in the configuration space.
        weighte : bool
            Whether the samples are already Boltzmann-weighted, i.e. drawn from MC simulations 
            without removing duplicates of configurations

        Returns
        -------
        J_ml : torch.Tensor
            The loss function J_ML
        """
        z, log_R_xz = self.inverse_generator(batch_x)
        u_z = self.calculate_energy(z, space='latent') 
        if weighted is not True:
            J_ml = self.expectation(u_z - log_R_xz)
        else:
            u_x = self.calculate_energy(batch_x, space='configuration')  
            weights_x = torch.exp(-u_x) 
            J_ml = self.expectation(u_z - log_R_xz, weights=weights_x)

        return J_ml

    def loss_KL(self, batch_z, weighted=False):
        """
        Calculates the loss function when training by energy (samples from the latent space)
        J_KL = E[u_x(x) - log Rzx(z)]

        Parameters
        ----------
        batch_z : torch.Tensor
            A batch of samples in the latent space

        Returns
        -------
        J_kl : torch.Tensor
            The loss function J_KL
        """
        x, log_R_zx = self.generator(batch_z)
        u_x = self.calculate_energy(x, space='configuration')   # we need this to calculate J_kl
        if weighted is not True:
            J_kl = self.expectation(u_x - log_R_zx)
        else:
            u_z = self.calculate_energy(batch_z, space='latent')
            weights_z = torch.exp(-u_z)  
            J_kl = self.expectation(u_x - log_R_zx, weights=weights_z)

        return J_kl

    def loss_RC(self, batch_RC, estimator, weighted=False):
        """
        Calculates the reaction coordinate loss function. J_RC = E[logp(RC)].

        Parameters
        ----------
        batch_RC : np.array
            A batch of samples along the reaction coordinate (in the configuration space).
        estimator : sklearn.neighbors.KernelDensity object
            A kernel density estimator to estimate the probability of the samples
        
        Returns
        -------
        J_rc : torch.Tensor
            The loss function J_RC

        Note
        ----
        At the current stage, this method might only work for DWP.
        """
        log_p = estimator.score_samples(batch_RC[:, 0][:, None])
        if weighted is not True:
            J_rc = self.expectation(log_p)
        else:
            u_rc = self.calculate_energy(batch_RC, space='configuration')
            weights_rc = torch.exp(-u_rc)
            J_rc = self.expectation(log_p, weights=weights_rc)

        return J_rc

    def calculate_energy(self, batch, space):
        """
        Calculate the energy of each each configuration in a batch of dataset.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of configurations
        
        Returns
        -------
        energy : torch.Tensor
            The energies of the configurations
        space : str
            Whether to calcualte the energy in the real space (x) or the 
            latent space (z). Available options: 'latent' or 'configuration'.
        """

        e_high, e_max = 10 ** 4, 10 ** 20
        energy = batch.new_zeros(batch.shape[0])  # like np.zeros, same length as batch_data

        if space == 'configuration':
            for i in range(batch.shape[0]):  # for each data point in the dataset
                config = batch[i, :].reshape(self.sys_dim)  # ensure correct dimensionality
                energy[i] = self.regularize_energy(self.system.get_energy(config))
        elif space == 'latent':
            for i in range(batch.shape[0]):  # for each data point in the dataset
                config = batch[i, :].reshape(self.sys_dim)  # ensure correct dimensionality
                # for 2D Gaussian distribution, u(z) = (1 / (2*sigma **2)) * z ** 2
                # in our case, sigma =1 and z ** 2 = z[0] ** 2 + z[1] ** 2
                energy[i] = self.regularize_energy(0.5 * torch.sum(config ** 2))
        else:
            print("Error! Unavailable option of parameter 'space' specificed.")
            sys.exit()
        return energy

    def regularize_energy(self, energy, e_high = 10 ** 8, e_max = 10 ** 20):
        if energy.item() > e_high:
            energy = e_high + torch.log10(energy - e_high + 1)
        elif energy.item() > e_max:
            energy= e_high + torch.log10(e_max - e_high + 1)
        return energy


    def expectation(self, observable, weights = None):
        """
        Calculate the expectation value of an observable

        Parameters
        ----------
        observable : torch.Tensor
            Observable of interest.

        Returns
        -------
        e : torch.Tensor
            Expectation value as a one-element tensor
        """
        # e = torch.dot(observable, weights) / torch.sum(weights) #the same as below
        if weights is None:
            e = observable.mean()
        else:
            e = torch.sum(observable * weights) / torch.sum(weights)
        return e


# ===========================================================================
# Spline coupling flow for the 2-D solute system
# ===========================================================================
import math as _math

from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)


class BoxUniformPrior:
    """Uniform prior on [-bound, bound]^dim."""

    def __init__(self, dim, bound):
        self.dim = int(dim)
        self.bound = float(bound)
        self._log_prob_val = -self.dim * _math.log(2.0 * self.bound)

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        return torch.rand(*shape, self.dim) * (2 * self.bound) - self.bound

    def log_prob(self, z):
        return torch.full(
            (z.shape[0],), self._log_prob_val, device=z.device, dtype=z.dtype
        )


class GaussianPrior:
    """Isotropic Gaussian prior N(0, sigma^2 * I_dim)."""

    def __init__(self, dim, sigma=1.0):
        self.dim = int(dim)
        self.sigma = float(sigma)
        self._log_norm = -0.5 * self.dim * (_math.log(2 * _math.pi) + 2 * _math.log(sigma))

    def sample(self, shape):
        if isinstance(shape, int):
            shape = (shape,)
        return torch.randn(*shape, self.dim) * self.sigma

    def log_prob(self, z):
        return -0.5 * (z / self.sigma).pow(2).sum(-1) + self._log_norm


def overlap_penalty(x_flat, sys_dim, sigma):
    """
    Differentiable soft overlap penalty.

    Parameters
    ----------
    x_flat : (B, N*D) tensor
    sys_dim : tuple (N, D)
    sigma : float — minimum allowed distance

    Returns
    -------
    scalar — mean over batch of sum_{i<j} relu(sigma^2 - r_ij^2)^2
    """
    N, D = sys_dim
    B = x_flat.shape[0]
    coords = x_flat.reshape(B, N, D)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)   # (B, N, N, D)
    r2 = (diff * diff).sum(-1)                          # (B, N, N)
    idx = torch.triu_indices(N, N, offset=1, device=x_flat.device)
    r2_pairs = r2[:, idx[0], idx[1]]                   # (B, N*(N-1)/2)
    return torch.relu(sigma ** 2 - r2_pairs).pow(2).sum(-1).mean()


class SplineConditioner2D(nn.Module):
    """MLP that maps frozen-particle coordinates to spline parameters for active particles."""

    def __init__(self, n_frozen_coords, n_active_coords, num_bins, n_nodes, n_hidden):
        super().__init__()
        self.params_per_coord = 3 * num_bins - 1
        self.n_active_coords = n_active_coords
        self.num_bins = num_bins

        layers = []
        in_dim = n_frozen_coords
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, n_nodes), nn.ReLU()]
            in_dim = n_nodes
        final = nn.Linear(in_dim, n_active_coords * self.params_per_coord)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, x_frozen):
        B = x_frozen.shape[0]
        return self.net(x_frozen).view(B, self.n_active_coords, self.params_per_coord)



class SoluteSplineCoupling(nn.Module):
    """Rational-quadratic spline coupling layer for 2-D particle systems."""

    def __init__(self, conditioner, frozen_idx, active_idx, num_bins,
                 tail_bound=5.0, min_bin_width=1e-3, min_bin_height=1e-3,
                 min_derivative=1e-3):
        super().__init__()
        self.conditioner = conditioner
        self.register_buffer("frozen_idx", frozen_idx)
        self.register_buffer("active_idx", active_idx)
        self.num_bins = num_bins
        self.tail_bound = float(tail_bound)
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

    def _get_params(self, x):
        x_frozen_flat = x[:, self.frozen_idx, :].reshape(x.shape[0], -1)
        params = self.conditioner(x_frozen_flat)   # (B, N_A*2, ppc)
        B2, _N_A2, ppc = params.shape
        params = params.view(B2, -1, 2, ppc)       # (B, N_A, 2, ppc)
        nb = self.num_bins
        return params[..., :nb], params[..., nb:2*nb], params[..., 2*nb:]

    def forward(self, x):
        """x: (B, N, 2) — returns (y, logabsdet) where logabsdet is (B,)."""
        widths, heights, derivatives = self._get_params(x)
        x_active = x[:, self.active_idx, :]        # (B, N_A, 2)
        y_active, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=x_active,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=False,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        y = x.clone()
        y[:, self.active_idx, :] = y_active
        return y, logabsdet.sum(dim=(-1, -2))

    def inverse(self, y):
        """y: (B, N, 2) — returns (x, logabsdet) where logabsdet is (B,)."""
        widths, heights, derivatives = self._get_params(y)
        y_active = y[:, self.active_idx, :]
        x_active, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=y_active,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=True,
            tails="linear",
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        x = y.clone()
        x[:, self.active_idx, :] = x_active
        return x, logabsdet.sum(dim=(-1, -2))


class SoluteSplineFlow(nn.Module):
    """
    Neural spline flow for the 2-D solute-LJ-bath system.

    Same external API as RealNVP: loss_ML, loss_KL, generator, inverse_generator.
    Additionally exposes loss_KL(w_overlap=...) to penalise particle overlaps.
    """

    def __init__(self, layers, prior, system, n_particles):
        super().__init__()
        self.coupling_layers = nn.ModuleList(layers)
        self.prior = prior
        self.system = system
        self.n_particles = int(n_particles)
        self.sys_dim = (n_particles, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_particle(self, flat):
        return flat.view(flat.shape[0], self.n_particles, 2)

    def _to_flat(self, particle):
        return particle.reshape(particle.shape[0], -1)

    def forward_map(self, x):
        """x: (B, N, 2) — returns (z, sum_logdet)."""
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.coupling_layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, logdet

    def inverse_map(self, z):
        """z: (B, N, 2) — returns (x, sum_logdet)."""
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.coupling_layers):
            z, ld = layer.inverse(z)
            logdet = logdet + ld
        return z, logdet

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def loss_ML(self, batch_x, weighted=False):
        """Maximum-likelihood loss. batch_x: (B, N*2) in Cartesian coordinates."""
        x = self._to_particle(batch_x)
        z, logdet_fwd = self.forward_map(x)
        log_pz = self.prior.log_prob(self._to_flat(z))
        return -(log_pz + logdet_fwd).mean()
    
    def _add_fixed_solute(self, solvent_flat):
        B = solvent_flat.shape[0]
        solvent = solvent_flat.view(B, self.n_particles, 2)

        solute = torch.zeros(
            B, 1, 2,
            device=solvent.device,
            dtype=solvent.dtype,
        )

        full = torch.cat([solute, solvent], dim=1)  # (B, n_particles+1, 2)
        return full.reshape(B, -1)

    def loss_KL(self, batch_z, weighted=False, w_overlap=0.0):
        """
        Reverse-KL loss. batch_z: (B, N*2) samples from prior.

        Parameters
        ----------
        w_overlap : float
            Weight for the overlap penalty term (default 0 = disabled).
        """
        z = self._to_particle(batch_z)
        x, logdet_inv = self.inverse_map(z)
        x_cart = self._to_flat(x)

        x_full_flat = self._add_fixed_solute(x_cart)
        energies = torch.stack([
            self.system.get_energy(x_full_flat[i].reshape(37, 2))
            for i in range(x_full_flat.shape[0])
        ])
        energy_cap = 1e3
        u_x = torch.where(
            energies < energy_cap,
            energies,
            energy_cap + torch.log1p(energies - energy_cap),
        )
        loss = (u_x - logdet_inv).mean()

        if w_overlap > 0.0:
            # loss = loss + w_overlap * overlap_penalty(
            #     x_flat, self.sys_dim, self.system.sigma
            # )
            penalty = overlap_penalty(
                x_full_flat, (37, 2), self.system.sigma)
            print(penalty)
            loss = loss + w_overlap * penalty
        return loss

    # ------------------------------------------------------------------
    # Generator interface (same as RealNVP)
    # ------------------------------------------------------------------
    def generator(self, z):
        """z: (B, N*2) latent → x: (B, N*2) Cartesian."""
        x_particle, logdet_inv = self.inverse_map(self._to_particle(z))
        return self._to_flat(x_particle), logdet_inv

    def inverse_generator(self, x):
        """x: (B, N*2) Cartesian → z: (B, N*2) latent."""
        z_particle, logdet_fwd = self.forward_map(self._to_particle(x))
        return self._to_flat(z_particle), logdet_fwd


def build_solute_spline_flow(
    system,
    n_particles=36,
    n_blocks=8,
    n_nodes=256,
    n_hidden=3,
    num_bins=8,
    tail_bound=7.0,
    hidden=128,
):
    """
    Build a SoluteSplineFlow for the 2-D solute-LJ-bath system.

    Models only the 36 solvent particles (72 dimensions). The solute is fixed
    at the origin; loss_KL prepends it via _add_fixed_solute before computing energy.

    Parameters
    ----------
    system      : system object with .get_energy() and .sigma
    n_particles : number of solvent particles to model (default 36)
    n_blocks    : number of A→B + B→A coupling block pairs (total layers = 2*n_blocks)
    n_nodes     : hidden layer width in each conditioner MLP (only used when conditioner='mlp')
    n_hidden    : number of hidden layers in each conditioner MLP (only used when conditioner='mlp')
    num_bins    : number of spline bins
    tail_bound  : spline domain (-tail_bound, tail_bound); default 7.0 suits
                  Cartesian data in [-5, 5] with 40% margin.
    hidden      : hidden dimension for PairwiseInvariantConditioner (only used when
                  conditioner='pairwise'; default 128).
    """

    all_idx = torch.arange(n_particles)          # 0..35 (0-based solvent indices)
    group_A = all_idx[all_idx % 2 == 0]          # 18 particles: [0, 2, ..., 34]
    group_B = all_idx[all_idx % 2 == 1]          # 18 particles: [1, 3, ..., 35]
    n_A, n_B = len(group_A), len(group_B)

    layers = []
    for _ in range(n_blocks):
        cond_AB = SplineConditioner2D(n_B * 2, n_A * 2, num_bins, n_nodes, n_hidden)
        cond_BA = SplineConditioner2D(n_A * 2, n_B * 2, num_bins, n_nodes, n_hidden)
        layers.append(SoluteSplineCoupling(cond_AB, group_B, group_A, num_bins, tail_bound))
        layers.append(SoluteSplineCoupling(cond_BA, group_A, group_B, num_bins, tail_bound))

    prior = GaussianPrior(dim=n_particles * 2)
    return SoluteSplineFlow(
        layers=layers, prior=prior, system=system,
        n_particles=n_particles,
    )
