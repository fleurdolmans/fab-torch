import torch
from torch import nn
import numpy as np
import tempfile
import pathlib
import os

from fab.target_distributions.base import TargetDistribution
from fab.transforms.global_3point_spherical_transform import Global3PointSphericalTransform

import boltzgen as bg
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools.testsystems import TestSystem, get_data_filename
from fab.utils.numerical import effective_sample_size_over_p


class WaterInWaterBox(TestSystem):

    """Water box with water micro-solvation system.

    Parameters
    ----------
    dim: dimensionality of system (num_atoms x 3).

    """

    def __init__(self, dim, **kwargs):
        TestSystem.__init__(self, **kwargs)
        # http://docs.openmm.org/latest/userguide/application/02_running_sims.html
        # Two methods: either create system from pdb and FF with forcefield.createSystems() or use prmtop and crd files,
        #  as in the openmmtools testsystems examples:
        #  https://openmmtools.readthedocs.io/en/stable/_modules/openmmtools/testsystems.html#AlanineDipeptideImplicit

        # prmtop + crd example:
        # prmtop_filename = get_data_filename("data/alanine-dipeptide-gbsa/alanine-dipeptide.prmtop")
        # crd_filename = get_data_filename("data/alanine-dipeptide-gbsa/alanine-dipeptide.crd")
        #
        # # Initialize system.
        # prmtop = app.AmberPrmtopFile(prmtop_filename)
        # system = prmtop.createSystem(
        #     implicitSolvent=app.OBC1, constraints=constraints, nonbondedCutoff=None, hydrogenMass=hydrogenMass
        # )
        # # Extract topology
        # topology = prmtop.topology
        # # Read positions.
        # inpcrd = app.AmberInpcrdFile(crd_filename)
        # positions = inpcrd.getPositions(asNumpy=True)
        # self.topology, self.system, self.positions = topology, system, positions

        # TODO: Other parameters? HydrogenMass, cutoffs, etc.?
        # TODO: Add radial restraint force to keep solvation shells close
        # TODO: Add implicit solvent to the system
        self.num_atoms_per_solute = 3  # Water
        self.num_atoms_per_solvent = 3  # Water
        self.num_solvent_molecules = (dim - self.num_atoms_per_solute) // (self.num_atoms_per_solvent * 3)

        # pdb example:
        # Initial solute molecule
        pdb = app.PDBFile("/home/timsey/HDD/data/molecules/solvents/water.pdb")
        # NOTE: Add solute force field if not water!
        # Add solvent
        modeller = app.modeller.Modeller(pdb.topology, pdb.positions)
        forcefield = app.ForceField("amber14/tip3p.xml")  # tip3pfb
        # ‘tip3p’, ‘spce’, ‘tip4pew’, ‘tip5p’, ‘swm4ndp’
        modeller.addSolvent(forcefield, model="tip3p", numAdded=self.num_solvent_molecules)
        # Create system
        self.system = forcefield.createSystem(modeller.topology)

        self.topology, self.positions = modeller.getTopology(), modeller.getPositions()
        # self.topology.atoms() yields the atom order, which is OHH OHH OHH etc.
        # This is the order in which the coordinates are stored in the positions array.
        self.atoms = [atom.name for atom in self.topology.atoms()]


class H2OinH2O(nn.Module, TargetDistribution):
    def __init__(
        self,
        dim=3 * (3 + 3 * 8),  # 3 atoms in solute, 3 atoms in solvent, 8 solvent molecules. 3 dimensions per atom (xyz)
        temperature=300,
        energy_cut=1.0e8,  # TODO: Does this still make sense? Originally for 1000K ALDP.
        energy_max=1.0e20,  # TODO: Does this still make sense? Originally for 1000K ALDP.
        n_threads=4,
        data_path=None,
        device="cpu",
        target_samples_path=None,
        save_dir=None,
    ):
        """
        Boltzmann distribution of H2O in H2O.
        :param temperature: Temperature of the system
        :param energy_cut: Value after which the energy is logarithmically scaled
        :param energy_max: Maximum energy allowed, higher energies are cut
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :param data_path: Path to the data used for coordinate transform
        :param device: Device on which the model is run
        :param target_samples_path: Path to the target samples (e.g., MD samples).
        :param save_dir: Directory being used for saving models, metric, etc.
        """
        super(H2OinH2O, self).__init__()

        self.dim = dim
        self.temperature = temperature
        self.energy_cut = energy_cut
        self.energy_max = energy_max
        self.n_threads = n_threads
        self.data_path = data_path
        self.device = device
        self.target_samples_path = target_samples_path
        self.save_dir = save_dir
        self.metric_dir = os.path.join(self.save_dir, f"metrics")

        # Steps to take:
        # 1. Load topology of solute.
        # 2. Solvate the solute.
        # 3. Add the solute and solvent force fields.
        # 4. Add the implicit solvent force field / external potential term.

        self.system = WaterInWaterBox(dim)
        # TODO: What filetype are MD samples usually saved in? Support this.
        if target_samples_path is None:
            raise ValueError("Cannot evaluate H20inH20 system without target samples.")
        assert target_samples_path.suffix == ".pt", "Target samples path must be a .pt file"
        self.target_data = torch.load(target_samples_path)

        # Generate trajectory for coordinate transform if no data path is specified
        if data_path is None:
            traj_sim = app.Simulation(
                self.system.topology,
                self.system.system,
                mm.LangevinIntegrator(temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                platform=mm.Platform.getPlatformByName("Reference"),
            )
            traj_sim.context.setPositions(self.system.positions)
            traj_sim.minimizeEnergy()
            state = traj_sim.context.getState(getPositions=True)
            position = state.getPositions(True).value_in_unit(unit.nanometer)
            tmp_dir = pathlib.Path(tempfile.gettempdir())
            data_path = tmp_dir / f"h2o_in_h2o_{dim}.pt"

            torch.save(torch.tensor(position.reshape(1, dim).astype(np.float64)), data_path)
            del traj_sim

        data_path = pathlib.Path(data_path)
        assert data_path.suffix == ".pt", "Data path must be a .pt file"
        self.transform_data = torch.load(data_path)
        assert self.transform_data.shape[-1] == dim, (
            f"Data shape ({self.transform_data.shape}) does not match number of " 
            f"coordinates in current system ({dim})."
        )

        self.coordinate_transform = Global3PointSphericalTransform(self.system, self.transform_data.to(device))

        if n_threads > 1:
            self.p = bg.distributions.TransformedBoltzmannParallel(
                self.system,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,
                n_threads=n_threads,
            )
        else:
            # Need to define sim, since the non-parallel version does not take a system as input (parallel builds the
            #  sim from system in exactly this way).
            sim = app.Simulation(
                self.system.topology,
                self.system.system,
                mm.LangevinIntegrator(temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                mm.Platform.getPlatformByName("Reference"),
            )

            self.p = bg.distributions.TransformedBoltzmann(
                sim.context,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,
            )

    def log_prob(self, x: torch.tensor):
        # Add global potential for implicit solvent here?
        return self.p.log_prob(x)

    def performance_metrics(self, samples, log_w, log_q_fn, batch_size):
        if log_q_fn:  # Evaluate base flow samples, but this is weird because why would AIS samples not need q?
            # TODO: This currently only evaluates test samples using the flow and target p.
            #  This should be expanded to estimate with generated samples as well (e.g. log(p(x)), with x ~ q(x)).
            #  Also, in the current code log_q_fn is not given when doing evaluation on AIS samples, which means we
            #  don't actually ever evaluate on those samples.
            #  So we need:
            #  1. Evaluate flow base on target samples. (Done, but probably bad estimate? Use the KL histogram stuff?)
            #  2. Evaluate flow with AIS sampling on target samples.
            #  3. Evaluate flow base samples: e.g. by checking the target log prob of these samples.
            #  4. Evaluate flow AIS samples: e.g. by checking the target log prob of these samples.
            log_q_test = log_q_fn(self.target_data)
            log_p_test = self.log_prob(self.target_data)
            test_mean_log_prob = torch.mean(log_q_test)
            kl_forward = torch.mean(log_p_test - log_q_test)
            ess_over_p = effective_sample_size_over_p(log_p_test - log_q_test)
            summary_dict = {
                "base_test_set_mean_log_prob": test_mean_log_prob.cpu().item(),
                "base_ess_over_p": ess_over_p.detach().cpu().item(),
                "base_kl_forward": kl_forward.detach().cpu().item(),
            }
        else:  # Evaluate AIS samples, but this is weird because why would AIS samples not need q?
            summary_dict = {}
        return summary_dict

        # z_test = self.target_data  # TODO: Should this be in z-space somehow???
        # z_sample = samples
        #
        # # Determine likelihood of test data and transform it (mostly taken from aldp.py)
        # z_d_np = z_test.cpu().data.numpy()
        # x_d_np = np.zeros((0, self.dim))
        # log_p_sum = 0
        # n_batches = int(np.ceil(len(z_test) / batch_size))
        # for i in range(n_batches):
        #     if i == n_batches - 1:
        #         end = len(z_test)
        #     else:
        #         end = (i + 1) * batch_size
        #     z = z_test[(i * batch_size): end, :]
        #     x, log_det = self.coordinate_transform(z.double())  # Z --> X
        #     x_d_np = np.concatenate((x_d_np, x.cpu().data.numpy()))
        #     log_p = self.log_prob(z)
        #     log_p_sum = log_p_sum + torch.sum(log_p).detach() - torch.sum(log_det).detach().float()
        # log_p_avg = log_p_sum.cpu().data.numpy() / len(z_test)
        #
        # # Transform samples
        # z_np = np.zeros((0, 60))
        # x_np = np.zeros((0, 66))
        # n_batches = int(np.ceil(len(samples) / batch_size))
        # for i in range(n_batches):
        #     if i == n_batches - 1:
        #         end = len(z_sample)
        #     else:
        #         end = (i + 1) * batch_size
        #     z = z_sample[(i * batch_size): end, :]
        #     x, _ = self.transform(z.double())  # Z --> X
        #     x_np = np.concatenate((x_np, x.cpu().data.numpy()))
        #     z, _ = self.transform.inverse(x)  # X --> Z
        #     z_np = np.concatenate((z_np, z.cpu().data.numpy()))
        #
        # # Estimate density of marginals
        # nbins = 200
        # hist_range = [-5, 5]
        # ndims = z_np.shape[1]
        #
        # hists_test = np.zeros((nbins, ndims))
        # hists_gen = np.zeros((nbins, ndims))
        #
        # for i in range(ndims):
        #     htest, _ = np.histogram(z_d_np[:, i], nbins, range=hist_range, density=True)
        #     hgen, _ = np.histogram(z_np[:, i], nbins, range=hist_range, density=True)
        #     hists_test[:, i] = htest
        #     hists_gen[:, i] = hgen
        #
        # # Compute KLD of marginals
        # eps = 1e-10
        # kld_unscaled = np.sum(hists_test * np.log((hists_test + eps) / (hists_gen + eps)), axis=0)
        # kld = kld_unscaled * (hist_range[1] - hist_range[0]) / nbins
        #
        # # Split KLD into groups
        # r_ind = np.arange(0, self.dim, 3)
        # phi_ind = np.arange(1, self.dim, 3)
        # theta_ind = np.arange(2, self.dim, 3)
        #
        # kld_r = kld[r_ind]
        # kld_phi = kld[phi_ind]
        # kld_theta = kld[theta_ind]
        #
        # # Save metrics
        # if self.metric_dir is not None:
        #     # Calculate and save KLD stats of marginals
        #     kld = (kld_r, kld_phi, kld_theta)
        #     kld_labels = ["bond", "angle", "dih"]
        #     kld_ = np.concatenate(kld)
        #     kld_append = np.array([[iteration + 1, np.median(kld_), np.mean(kld_)]])
        #     kld_path = os.path.join(self.metric_dir, "kld.csv")
        #     if os.path.exists(kld_path):
        #         kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=",")
        #         if len(kld_hist.shape) == 1:
        #             kld_hist = kld_hist[None, :]
        #         kld_hist = np.concatenate([kld_hist, kld_append])
        #     else:
        #         kld_hist = kld_append
        #     np.savetxt(kld_path, kld_hist, delimiter=",", header="it,kld_median,kld_mean", comments="")
        #
        #     # Save KLD per coordinate (r, phi, theta)
        #     for kld_label, kld_ in zip(kld_labels, kld):
        #         kld_append = np.concatenate([np.array([iteration + 1, np.median(kld_), np.mean(kld_)]), kld_])
        #         kld_append = kld_append[None, :]
        #         kld_path = os.path.join(self.metric_dir, "kld_" + kld_label + ".csv")
        #         if os.path.exists(kld_path):
        #             kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=",")
        #             if len(kld_hist.shape) == 1:
        #                 kld_hist = kld_hist[None, :]
        #             kld_hist = np.concatenate([kld_hist, kld_append])
        #         else:
        #             kld_hist = kld_append
        #         header = "it,kld_median,kld_mean"
        #         for kld_ind in range(len(kld_)):
        #             header += ",kld" + str(kld_ind)
        #         np.savetxt(kld_path, kld_hist, delimiter=",", header=header, comments="")
        #
        #     # Save log probability
        #     log_p_append = np.array([[iteration + 1, log_p_avg]])
        #     log_p_path = os.path.join(self.metric_dir, "log_p_test.csv")
        #     if os.path.exists(log_p_path):
        #         log_p_hist = np.loadtxt(log_p_path, skiprows=1, delimiter=",")
        #         if len(log_p_hist.shape) == 1:
        #             log_p_hist = log_p_hist[None, :]
        #         log_p_hist = np.concatenate([log_p_hist, log_p_append])
        #     else:
        #         log_p_hist = log_p_append
        #     np.savetxt(log_p_path, log_p_hist, delimiter=",", header="it,log_p", comments="")
        #
        # info = {
        #     "test_set_mean_log_prob": None,
        #     "test_set_kld": None,
        #     "eval_batch_size": None,
        # }
        # return info
