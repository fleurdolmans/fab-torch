import sys
import copy
import numpy as np 
import torch
import torch.nn as nn
from .. import utils as utils

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)


def _soft_cap(U, cap, alpha=0.05):
    """Differentiable soft energy cap matching SoluteSplineFlow.loss_KL.

    Below cap: identity.  Above cap: cap + alpha*(U-cap) + (1-alpha)*log1p(U-cap).
    Keeps gradients alive everywhere via torch.where.
    """
    excess = U - cap
    return torch.where(U < cap, U, cap + alpha * excess + (1.0 - alpha) * torch.log1p(excess))


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
        super().__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([t_net() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([s_net() for _ in range(len(mask))])
        self.system = system
        self.sys_dim = sys_dim
        # Scale factor applied to the coupling-layer s output.
        # Reduce (e.g. 0.5) to prevent scale blowup during KL training.
        self.s_scale = 1.0

    def _add_fixed_solute(self, x_flat):
        """Prepend a fixed solute at the origin to a batch of flat solvent coordinates.

        Parameters
        ----------
        x_flat : (B, N_solvent * 2) tensor

        Returns
        -------
        (B, (N_solvent + 1) * 2) tensor — solute at index 0, solvent at indices 1..N
        """
        B = x_flat.shape[0]
        N, D = self.sys_dim
        solvent = x_flat.view(B, N, D)
        solute  = torch.zeros(B, 1, D, device=x_flat.device, dtype=x_flat.dtype)
        return torch.cat([solute, solvent], dim=1).reshape(B, -1)

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
        z = x
        log_R_xz = x.new_zeros(x.shape[0])
        for i in reversed(range(len(self.t))):
            # See equation (9) in the RealNVP paper
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i]) * self.s_scale
            t = self.t[i](z_) * (1 - self.mask[i])
            z = z_ + (1 - self.mask[i]) * (z - t) * torch.exp(-s)
            log_R_xz -= torch.sum(s, -1)
        return z, log_R_xz

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
        log_R_zx : torch.Tensor
            Log of the determinant of the Jacobian of Fzx (generator)
        """
        x = z
        log_R_zx = z.new_zeros(z.shape[0])
        x_list = []

        for i in range(len(self.t)):
            # See equation (9) in the RealNVP paper
            x_ = self.mask[i] * x
            s = self.s[i](x_) * (1 - self.mask[i]) * self.s_scale
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_R_zx += torch.sum(s, -1)
            if process is True:
                x_list.append(copy.deepcopy(x.detach().numpy()))

        if process is True:
            return x, log_R_zx, x_list
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
    
    def loss_KL(self, batch_z, weighted=False, w_overlap=0.0, energy_cap=None, cutoff_factor=0.9):
        """
        Calculates the loss function when training by energy (samples from the latent space)
        J_KL = E[u_x(x) - log Rzx(z)]

        Parameters
        ----------
        batch_z : torch.Tensor
            A batch of samples in the latent space
        energy_cap : float or None
            Soft energy cap passed to calculate_energy (and on to system.get_energy).
        cutoff_factor : float
            Factor for determining the cutoff distance for overlap penalty.

        Returns
        -------
        J_kl : torch.Tensor
            The loss function J_KL
        """
        x, log_R_zx = self.generator(batch_z)
        u_x = self.calculate_energy(x, space='configuration', energy_cap=energy_cap)   # we need this to calculate J_kl
        if weighted is not True:
            J_kl = self.expectation(u_x - log_R_zx)
        else:
            u_z = self.calculate_energy(batch_z, space='latent')
            weights_z = torch.exp(-u_z)
            J_kl = self.expectation(u_x - log_R_zx, weights=weights_z)

        if w_overlap > 0.0:
            x_full = self._add_fixed_solute(x)
            N, D = self.sys_dim
            J_kl = J_kl + w_overlap * utils.overlap_penalty(x_full, (N + 1, D), self.system.sigma, cutoff_factor=cutoff_factor)
        return J_kl

    def loss_overlap(self, batch_z, cutoff_factor=0.9):
        """
        Overlap penalty on flow-generated samples, decoupled from the KL loss.

        Parameters
        ----------
        batch_z : (B, N*2) tensor — prior samples
        cutoff_factor : float

        Returns
        -------
        penalty : scalar tensor
        """
        x, _ = self.generator(batch_z)
        x_full = self._add_fixed_solute(x)
        N, D = self.sys_dim
        return utils.overlap_penalty(x_full, (N + 1, D), self.system.sigma, cutoff_factor=cutoff_factor)

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

    def calculate_energy(self, batch, space, energy_cap=None):
        """
        Calculate the energy of each configuration in a batch.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of configurations.
        space : str
            'latent' or 'configuration'.
        energy_cap : float or None
            When set, passed as energy_cut to system.get_energy and then a
            soft cap (_soft_cap) is applied to the returned batch energies.
            Has no effect for the latent space (Gaussian prior is well-behaved).
        """
        energy = batch.new_zeros(batch.shape[0])

        if space == 'configuration':
            N, D = self.sys_dim
            for i in range(batch.shape[0]):
                solvent = batch[i].view(N, D)
                solute  = torch.zeros(1, D, device=batch.device, dtype=batch.dtype)
                config  = torch.cat([solute, solvent], dim=0)  # (N+1, D)
                energy[i] = self.system.get_energy(config)
            if energy_cap is not None:
                cap = torch.tensor(energy_cap, device=batch.device, dtype=batch.dtype)
                energy = _soft_cap(energy, cap)
        elif space == 'latent':
            for i in range(batch.shape[0]):
                config = batch[i, :].reshape(self.sys_dim)
                energy[i] = 0.5 * torch.sum(config ** 2)
        else:
            print("Error! Unavailable option of parameter 'space' specificed.")
            sys.exit()
        return energy


    def expectation(self, observable, weights=None):
        """
        Calculate the expectation value of an observable.

        Parameters
        ----------
        observable : torch.Tensor
        weights : torch.Tensor or None

        Returns
        -------
        e : torch.Tensor
        """
        if weights is None:
            return observable.mean()
        return torch.sum(observable * weights) / torch.sum(weights)


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

        super().__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False).to(device)
        self.t = torch.nn.ModuleList([t_net() for _ in range(len(mask))]).to(device)
        self.s = torch.nn.ModuleList([s_net() for _ in range(len(mask))]).to(device)
        self.system = system
        self.sys_dim = sys_dim
        # Scale factor applied to the coupling-layer s output.
        # Reduce (e.g. 0.5) to prevent scale blowup during KL training.
        self.s_scale = 1.0

    def _add_fixed_solute(self, x_flat):
        """Prepend a fixed solute at the origin to a batch of flat solvent coordinates.

        Parameters
        ----------
        x_flat : (B, N_solvent * 2) tensor

        Returns
        -------
        (B, (N_solvent + 1) * 2) tensor — solute at index 0, solvent at indices 1..N
        """
        B = x_flat.shape[0]
        N, D = self.sys_dim
        solvent = x_flat.view(B, N, D)
        solute  = torch.zeros(B, 1, D, device=x_flat.device, dtype=x_flat.dtype)
        return torch.cat([solute, solvent], dim=1).reshape(B, -1)

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
        z = x
        log_R_xz = x.new_zeros(x.shape[0])
        for i in reversed(range(len(self.t))):
            # See equation (9) in the RealNVP paper
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i]) * self.s_scale
            t = self.t[i](z_) * (1 - self.mask[i])
            z = z_ + (1 - self.mask[i]) * (z - t) * torch.exp(-s)
            log_R_xz -= torch.sum(s, -1)
        return z, log_R_xz

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
        log_R_zx : torch.Tensor
            Log of the determinant of the Jacobian of Fzx (generator)
        """
        x = z
        log_R_zx = z.new_zeros(z.shape[0])

        for i in range(len(self.t)):
            # See equation (9) in the RealNVP paper
            x_ = self.mask[i] * x
            s = self.s[i](x_) * (1 - self.mask[i]) * self.s_scale
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_R_zx += torch.sum(s, -1)

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

    def loss_KL(self, batch_z, weighted=False, w_overlap=0.0, energy_cap=None, cutoff_factor=0.9):
        """
        Calculates the loss function when training by energy (samples from the latent space)
        J_KL = E[u_x(x) - log Rzx(z)]

        Parameters
        ----------
        batch_z : torch.Tensor
            A batch of samples in the latent space
        w_overlap : float
            Weight for the solute-aware overlap penalty (default 0 = disabled).
        energy_cap : float or None
            Soft energy cap passed to calculate_energy (and on to system.get_energy).
        cutoff_factor : float
            Factor for determining the cutoff distance for overlap penalty.

        Returns
        -------
        J_kl : torch.Tensor
            The loss function J_KL
        """
        x, log_R_zx = self.generator(batch_z)
        u_x = self.calculate_energy(x, space='configuration', energy_cap=energy_cap)   # we need this to calculate J_kl
        if weighted is not True:
            J_kl = self.expectation(u_x - log_R_zx)
        else:
            u_z = self.calculate_energy(batch_z, space='latent')
            weights_z = torch.exp(-u_z)
            J_kl = self.expectation(u_x - log_R_zx, weights=weights_z)

        if w_overlap > 0.0:
            x_full = self._add_fixed_solute(x)
            N, D = self.sys_dim
            J_kl = J_kl + w_overlap * utils.overlap_penalty(x_full, (N + 1, D), self.system.sigma, cutoff_factor=cutoff_factor)
        return J_kl

    def loss_overlap(self, batch_z, cutoff_factor=0.9):
        """
        Overlap penalty on flow-generated samples, decoupled from the KL loss.

        Parameters
        ----------
        batch_z : (B, N*2) tensor — prior samples
        cutoff_factor : float

        Returns
        -------
        penalty : scalar tensor
        """
        x, _ = self.generator(batch_z)
        x_full = self._add_fixed_solute(x)
        N, D = self.sys_dim
        return utils.overlap_penalty(x_full, (N + 1, D), self.system.sigma, cutoff_factor=cutoff_factor)

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

    def calculate_energy(self, batch, space, energy_cap=None):
        """
        Calculate the energy of each configuration in a batch.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of configurations.
        space : str
            'latent' or 'configuration'.
        energy_cap : float or None
            When set, passed as energy_cut to system.get_energy and then a
            soft cap (_soft_cap) is applied to the returned batch energies.
            Has no effect for the latent space (Gaussian prior is well-behaved).
        """
        energy = batch.new_zeros(batch.shape[0])

        if space == 'configuration':
            N, D = self.sys_dim
            for i in range(batch.shape[0]):
                solvent = batch[i].view(N, D)
                solute  = torch.zeros(1, D, device=batch.device, dtype=batch.dtype)
                config  = torch.cat([solute, solvent], dim=0)  # (N+1, D)
                energy[i] = self.system.get_energy(config)
            if energy_cap is not None:
                cap = torch.tensor(energy_cap, device=batch.device, dtype=batch.dtype)
                energy = _soft_cap(energy, cap)
        elif space == 'latent':
            for i in range(batch.shape[0]):
                config = batch[i, :].reshape(self.sys_dim)
                energy[i] = 0.5 * torch.sum(config ** 2)
        else:
            print("Error! Unavailable option of parameter 'space' specificed.")
            sys.exit()
        return energy


    def expectation(self, observable, weights=None):
        """
        Calculate the expectation value of an observable.

        Parameters
        ----------
        observable : torch.Tensor
        weights : torch.Tensor or None

        Returns
        -------
        e : torch.Tensor
        """
        if weights is None:
            return observable.mean()
        return torch.sum(observable * weights) / torch.sum(weights)

