import torch
from torch import nn
import numpy as np

import openmm as mm
from openmm import unit
from openmm import app

import torch.nn.functional as F

import multiprocessing as mp

"""
This file contains a number of classes that implement the Boltzmann distribution and OpenMM interface. Taken from the 
boltzmann-generators repo: https://github.com/VincentStimper/boltzmann-generators

This is necessary because we need a different integrator for OpenMMEnergyInterfaceParallel.
"""

# Gas constant in kJ / mol / K
R = 8.314e-3


class TransformedBoltzmann(nn.Module):
    """
    Boltzmann distribution with respect to transformed variables, uses OpenMM to get energy and forces.
    """

    def __init__(self, sim_context, temperature, energy_cut, energy_max, transform, force_groups, curriculum_type, curriculum_lambda, curriculum_soft_energy_cut):
        """
        Constructor
        :param sim_context: Context of the simulation object used for energy
        and force calculation
        :param temperature: Temperature of System
        :param energy_cut: Energy at which logarithm is applied
        :param energy_max: Maximum energy
        :param transform: Coordinate transformation
        :param curriculum_type: Type of curriculum
        :param curriculum_lambda: Lambda parameter for curriculum
        :param curriculum_soft_energy_cut: Soft energy cut parameter for curriculum
        """
        super().__init__()
        # Save input parameters
        self.sim_context = sim_context
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)

        # Set up functions
        self.openmm_energy = OpenMMEnergyInterface.apply
        self.regularize_energy = regularize_energy
        self.force_groups = force_groups
        self.curriculum_type = curriculum_type
        self.curriculum_lambda = curriculum_lambda
        self.curriculum_soft_energy_cut = curriculum_soft_energy_cut

        # self.norm_energy = lambda pos: self.regularize_energy(
        #     self.openmm_energy(pos, self.sim_context, temperature, self.force_groups)[:, 0], self.energy_cut, self.energy_max, 
        # )
        self.n_clipped = 0
        self.norm_energy = lambda pos: self.curriculum_energy(self.openmm_energy(pos, self.sim_context, temperature, self.force_groups)[:, 0])


        self.transform = transform

    def curriculum_energy(self, energy: torch.Tensor) -> torch.Tensor:
        if self.curriculum_type is None:
            energy_reg, n_clipped = self.regularize_energy(energy, self.energy_cut, self.energy_max)

        elif self.curriculum_type == "temperature":
            # lambda scaling: <1 means effectively higher temperature / flatter target
            energy = self.curriculum_lambda * energy
            energy_reg, n_clipped = self.regularize_energy(energy, self.energy_cut, self.energy_max)

        elif self.curriculum_type == "soft_cut":
            soft_cut = torch.tensor(
                self.curriculum_soft_energy_cut,
                device=energy.device,
                dtype=energy.dtype,
            )
            energy_reg, n_clipped = self.regularize_energy(energy, soft_cut, self.energy_max)

        else:
            raise ValueError(f"Unknown curriculum_type: {self.curriculum_type}")
        
        self.n_clipped = n_clipped
        return energy_reg


    def log_prob(self, z):
        """
        Log prob of system given in internal coordinates (e.g., Normalising Flow output).
        """
        if self.transform is not None:
            z, log_det = self.transform(z)  # I --> X
        else:
            log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        energy_term = -self.norm_energy(z)

        #  UNITS: We add logdetjac to energy, because energy is essentially log probability.
        #   Energy has units of kJ/mol by default (openMM). If we divide energy by N_A kBT = R * T,
        #   we get kJ/mol / (kJ/mol) = unitless!
        #   Check: R = N_A * kB, so R * T = N_A kBT, which has units of J / mol.
        #    because N_A (Avogadro) has units of /mol, and kB has units of J/K.
        return energy_term + log_det

    def log_prob_and_jac(self, z):
        """
        Log prob of system given in internal coordinates (e.g., Normalising Flow output) and the associated
        log determinant of the Jacobian of the transformation from internal to Cartesian coordinates.
        """
        if self.transform is not None:
            z, log_det = self.transform(z)  # I --> X
        else:
            log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return -self.norm_energy(z) + log_det, log_det

    def log_prob_x(self, x):
        """
        Log prob of system given in Cartesian coordinates.
        """
        return -self.norm_energy(x)


class TransformedBoltzmannParallel(nn.Module):
    """
    Boltzmann distribution with respect to transformed variables,
    uses OpenMM to get energy and forces and processes the batch of
    states in parallel
    """

    def __init__(self, system, temperature, energy_cut, energy_max, transform, platform_name="Reference", n_threads=None):
        """
        Constructor
        :param system: Molecular system
        :param temperature: Temperature of System
        :param energy_cut: Energy at which logarithm is applied
        :param energy_max: Maximum energy
        :param transform: Coordinate transformation
        :param platform_name: OpenMM platform to use
        :param n_threads: Number of threads to use to process batches, set to the number of cpus if None
        """
        super().__init__()
        # Save input parameters
        self.system = system
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)
        self.n_threads = mp.cpu_count() if n_threads is None else n_threads

        # Create pool for parallel processing
        self.pool = mp.Pool(self.n_threads, OpenMMEnergyInterfaceParallel.var_init, (system, temperature, platform_name))

        # Set up functions
        self.openmm_energy = OpenMMEnergyInterfaceParallel.apply
        self.regularize_energy = regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.pool)[:, 0], self.energy_cut, self.energy_max
        )

        self.transform = transform

    def log_prob(self, z):
        if self.transform is not None:
            z, log_det = self.transform(z)  # I --> X
        else:
            log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return -self.norm_energy(z) + log_det

    def log_prob_and_jac(self, z):
        if self.transform is not None:
            z, log_det = self.transform(z)  # I --> X
        else:
            log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        return -self.norm_energy(z) + log_det, log_det

    def log_prob_x(self, x):
        return -self.norm_energy(x)


class OpenMMEnergyInterface(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, openmm_context, temperature, force_groups):
        device = input.device
        n_batch = input.shape[0]
        input = input.view(n_batch, -1, 3)
        n_dim = input.shape[1]
        energies = torch.zeros((n_batch, 1), dtype=input.dtype)
        forces = torch.zeros_like(input)

        kBT = R * temperature
        input = input.cpu().detach().numpy()
        for i in range(n_batch):
            # reshape the coordinates and send to OpenMM
            x = input[i, :].reshape(-1, 3)
            # Handle nans and infinities
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                energies[i, 0] = np.nan
            else:
                openmm_context.setPositions(x)
                # state = openmm_context.getState(getForces=True, getEnergy=True, getPositions=True)

                if force_groups is None:
                    state = openmm_context.getState(getForces=True, getEnergy=True, getPositions=True)
                else:
                    state = openmm_context.getState(
                        getForces=True,
                        getEnergy=True,
                        getPositions=True,
                        groups=set(force_groups),
                    )
                # get energy
                # energies[i, 0] = state.getPotentialEnergy().value_in_unit(unit.kilojoule / unit.mole) / kBT
                energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule / unit.mole)
                energies[i, 0] = energy_kj / kBT

                # Printing energy components for debugging:
                # for j, frc in enumerate(openmm_context.getSystem().getForces()):
                #     stt = openmm_context.getState(getEnergy=True, groups={j})
                #     poten = stt.getPotentialEnergy()._value  # kJ/mol
                #     if np.abs(poten) > 1e-10:
                #         print(f" {frc.getName()}: {poten:.2f} kJ/mol")

                # get forces
                f = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule / unit.mole / unit.nanometer) / kBT
                forces[i, :] = torch.from_numpy(-f)

        forces = forces.view(n_batch, n_dim * 3)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        (forces,) = ctx.saved_tensors
        return forces * grad_output, None, None, None

class OpenMMEnergyInterfaceParallel(torch.autograd.Function):
    """
    Uses parallel processing to get the energies of the batch of states
    """

    @staticmethod
    def var_init(sys, temp, platform_name):
        """
        Method to initialize temperature and openmm context for workers
        of multiprocessing pool
        """
        global temperature, openmm_context
        temperature = temp
        sim = app.Simulation(
            sys.topology,
            sys.system,
            mm.LangevinMiddleIntegrator(temp * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
            platform=mm.Platform.getPlatformByName(platform_name),
        )
        openmm_context = sim.context

    @staticmethod
    def batch_proc(input):
        # Process state
        # openmm context and temperature are passed a global variables
        input = input.reshape(-1, 3)
        n_dim = input.shape[0]

        kBT = R * temperature
        # Handle nans and infinities
        if np.any(np.isnan(input)) or np.any(np.isinf(input)):
            energy = np.nan
            force = np.zeros_like(input)
        else:
            openmm_context.setPositions(input)
            openmm_context.applyConstraints(1e-6)
            state = openmm_context.getState(getForces=True, getEnergy=True)
            # get energy
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule / unit.mole) / kBT
            # get forces
            force = -state.getForces(asNumpy=True).value_in_unit(unit.kilojoule / unit.mole / unit.nanometer) / kBT
        force = force.reshape(n_dim * 3)
        return energy, force

    @staticmethod
    def forward(ctx, input, pool):
        device = input.device
        input_np = input.cpu().detach().numpy()
        energies_out, forces_out = zip(*pool.map(OpenMMEnergyInterfaceParallel.batch_proc, input_np))
        energies_np = np.array(energies_out)[:, None]
        forces_np = np.array(forces_out)
        energies = torch.from_numpy(energies_np)
        forces = torch.from_numpy(forces_np)
        energies = energies.type(input.dtype)
        forces = forces.type(input.dtype)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        (forces,) = ctx.saved_tensors
        return forces * grad_output, None, None

def smooth_min(u, cap, beta=1e-3):
    # smooth approximation of min(u, cap)
    return cap - F.softplus(beta * (cap - u)) / beta

def regularize_energy(energy, energy_cut, energy_max):
    # Cast inputs to same type
    # print("Energy before reg:", energy)
    energy_cut = energy_cut.type(energy.type())
    energy_max = energy_max.type(energy.type())
    # Check whether energy finite
    energy_finite = torch.isfinite(energy)

    n_clipped = (energy >= energy_cut).sum().item() # count clipped values (before clipping!)

    # Cap the energy at energy_max
    energy = torch.where(energy < energy_max, energy, energy_max)
    # Make it logarithmic above energy cut and linear below
    energy = torch.where(energy < energy_cut, energy, torch.log(energy - energy_cut + 1) + energy_cut)
    energy = torch.where(energy_finite, energy, torch.tensor(np.nan, dtype=energy.dtype, device=energy.device))

    # # Soft cap
    # energy = smooth_min(energy, energy_max)
    # energy = torch.where(energy < energy_cut, energy, energy_cut + torch.log1p((energy - energy_cut).clamp_min(0.0)))
    
    # energy = torch.where(energy_finite, energy, torch.full_like(energy, float("nan")))
    return energy, n_clipped
