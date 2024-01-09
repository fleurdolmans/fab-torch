import torch
import h5py
import warnings
from torch import nn
from torch import Tensor
from typing import Optional, List, Dict, Any, Callable, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import tempfile
import pathlib
import os

from fab.target_distributions.base import TargetDistribution
from fab.transforms.global_3point_spherical_transform import Global3PointSphericalTransform

from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools.testsystems import TestSystem, get_data_filename
from fab.utils.numerical import effective_sample_size_over_p
from fab.target_distributions.boltzmann import TransformedBoltzmann, TransformedBoltzmannParallel


class WaterInWaterBox(TestSystem):

    """Water box with water micro-solvation system.

    Parameters
    ----------
    dim: dimensionality of system (num_atoms x 3).

    """

    def __init__(self, solvent_pdb_path: str, dim: int, **kwargs):
        TestSystem.__init__(self, **kwargs)
        # http://docs.openmm.org/latest/userguide/application/02_running_sims.html
        # Two methods: either create system from pdb and FF with forcefield.createSystems() or use prmtop and crd files,
        #  as in the openmmtools testsystems examples:
        #  https://openmmtools.readthedocs.io/en/stable/_modules/openmmtools/testsystems.html#AlanineDipeptideImplicit

        # TODO: Other parameters? HydrogenMass, cutoffs, etc.?
        self.num_atoms_per_solute = 3  # Water
        self.num_atoms_per_solvent = 3  # Water
        self.num_solvent_molecules = (dim - self.num_atoms_per_solute) // (self.num_atoms_per_solvent * 3)

        # Steps to take:
        # 1. Load topology of solute.
        # 2. Solvate the solute.
        # 3. Add the solute and solvent force fields.
        # 4. Add the implicit solvent force field / external potential term.

        # Initial solute molecule
        pdb = app.PDBFile(solvent_pdb_path)
        # TODO: Add solute force field if not water!
        # Add solvent
        # This pdb file has a single water molecule, where the OH bonds are approx 0.1 nm in length.
        modeller = app.modeller.Modeller(pdb.topology, pdb.positions)  # In nanometers
        forcefield = app.ForceField("amber14/tip3p.xml")  # tip3pfb
        # ‘tip3p’, ‘spce’, ‘tip4pew’, ‘tip5p’, ‘swm4ndp’
        modeller.addSolvent(
            forcefield, model="tip3p", numAdded=self.num_solvent_molecules
        )
        # Create system
        # Default nonbondedCutoff is 1.0 nm, but this is too big:
        #  openmm.OpenMMException:
        #   NonbondedForce: The cutoff distance cannot be greater than half the periodic box size.
        self.system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1 * unit.nanometers,
            constraints=app.HBonds,
        )

        # add origin restraint for the central water (156)
        # this should actually be a rigid constraint!
        # constraints require dummy atoms
        # but topology doesn't play well. Will check tomorrow
        center = mm.CustomExternalForce('100000*max(0, r)^2; r=sqrt(x*x+y*y+z*z)')
        self.system.addForce(center)
        center.addParticle(0, [])

        # add spherical restraint to hold the droplet
        force = mm.CustomExternalForce('100*max(0, r-1.0)^2; r=sqrt(x*x+y*y+z*z)')
        self.system.addForce(force)
        for i in range(self.system.getNumParticles()):
            force.addParticle(i, [])

        self.topology, self.positions = modeller.getTopology(), modeller.getPositions()
        # self.topology.atoms() yields the atom order, which is OHH OHH OHH etc.
        # This is the order in which the coordinates are stored in the positions array.
        self.atoms = [atom.name for atom in self.topology.atoms()]


class H2OinH2O(nn.Module, TargetDistribution):
    def __init__(
        self,
        solvent_pdb_path: str,
        dim: int = 3 * (3 + 3 * 8),  # 3 atoms in solute, 3 atoms in solvent, 8 solvent molecules. 3 dimensions per atom (xyz)
        temperature: float = 300,
        energy_cut: float = 1.0e8,  # TODO: Does this still make sense? Originally for 1000K ALDP.
        energy_max: float = 1.0e20,  # TODO: Does this still make sense? Originally for 1000K ALDP.
        n_threads: int = 4,
        train_samples_path: Optional[str] = None,
        val_samples_path: Optional[str] = None,
        test_samples_path: Optional[str] = None,
        eval_mode: Literal["val", "test"] = "val",
        use_val_data_for_transform: bool = False,
        device: str = "cpu",
        save_dir: Optional[str] = None,
    ):
        """
        Boltzmann distribution of H2O in H2O.
        :param temperature: Temperature of the system
        :param energy_cut: Value after which the energy is logarithmically scaled
        :param energy_max: Maximum energy allowed, higher energies are cut
        :param n_threads: Number of threads used to evaluate the log
            probability for batches
        :param train_samples_path: Path to the target samples for training (e.g., MD samples).
        :param val_samples_path: Path to the target samples for evaluation (e.g., MD samples).
        :param test_samples_path: Path to the target samples for testing (e.g., MD samples).
        :param eval_mode: Whether to use the validation or test set for evaluation.
        :param device: Device on which the model is run

        :param save_dir: Directory being used for saving models, metric, etc.
        """
        super(H2OinH2O, self).__init__()

        self.dim = dim
        self.temperature = temperature
        self.energy_cut = energy_cut
        self.energy_max = energy_max
        self.n_threads = n_threads
        self.device = device

        self.save_dir = save_dir
        self.metric_dir = os.path.join(self.save_dir, f"metrics")
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)

        # Initialise system
        self.system = WaterInWaterBox(solvent_pdb_path, dim)

        # Load any MD data
        # TODO: This data is not put in the standard coordinate form with the solute at the origin. Should use the
        #  `setup_coordinate_system` function on this data to make sure it matches what the flow outputs after the
        #  z --> x transform.
        self.eval_mode = eval_mode
        self.raw_train_data, self.raw_val_data, self.raw_test_data = None, None, None
        self.train_data, self.val_data, self.test_data = None, None, None
        self.train_samples_path, self.val_samples_path, self.test_samples_path = None, None, None
        if train_samples_path:
            self.train_samples_path = pathlib.Path(train_samples_path)
            # OH bonds still ~0.1 nm in length for this data.
            self.raw_train_data = self.load_target_data(self.train_samples_path, dim).double()
        if val_samples_path:
            self.val_samples_path = pathlib.Path(val_samples_path)
            self.raw_val_data = self.load_target_data(self.val_samples_path, dim).double()
        if test_samples_path:
            self.test_samples_path = pathlib.Path(test_samples_path)
            self.raw_test_data = self.load_target_data(self.test_samples_path, dim).double()

        # Generate trajectory for coordinate transform if no data path is specified
        integrator = mm.LangevinMiddleIntegrator
        if not val_samples_path or not use_val_data_for_transform:
            traj_sim = app.Simulation(
                self.system.topology,
                self.system.system,
                integrator(temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                platform=mm.Platform.getPlatformByName("Reference"),
            )
            traj_sim.context.setPositions(self.system.positions)
            traj_sim.minimizeEnergy()
            state = traj_sim.context.getState(getPositions=True)
            position = state.getPositions(True).value_in_unit(unit.nanometer)  # TODO: Are these the same units as MD samples?
            transform_data = torch.tensor(position.reshape(1, dim)).double()
            del traj_sim
            self.transform_data = transform_data
        else:
            self.transform_data = self.raw_val_data.clone()[0].reshape(1, dim)

        assert self.transform_data.shape[-1] == dim, (
            f"Data shape ({self.transform_data.shape}) does not match number of coordinates in current system ({dim})."
        )

        self.coordinate_transform = Global3PointSphericalTransform(self.system, self.transform_data.to(device))

        # Use this transform to setup the coordinate system for loaded MD data as well.
        # raw_train/val/test_data = original x data loaded from dist
        # train/val/test_data = x data in centralised coordinate system
        if self.raw_train_data is not None:
            # OH bonds are still ~0.1 nm apart
            self.train_data, _, _, _ = self.coordinate_transform.setup_coordinate_system(
                self.raw_train_data.reshape(-1, self.dim)  # Setup expects flattened coordinates
            )
            # Store new coordinates for visualisation tests (notebook).
            fr = h5py.File(str(self.train_samples_path), "r")
            orig_coords = torch.from_numpy(fr["coordinates"][()])
            new_coords = self.train_data.numpy()
            new_filepath = str(self.train_samples_path.parent / self.train_samples_path.stem) + "_centered.h5"
            with h5py.File(new_filepath, "w") as fw:
                for obj in fr.keys():
                    fr.copy(obj, fw)
                assert np.isclose(fw["coordinates"][...], orig_coords).all(), (
                    "Original coordinates not copied correctly."
                )
                fw["coordinates"][...] = new_coords
            fr.close()

        if self.raw_val_data is not None:
            self.val_data, _, _, _ = self.coordinate_transform.setup_coordinate_system(
                self.raw_val_data.reshape(-1, self.dim)
            )
        if self.raw_test_data is not None:
            self.test_data, _, _, _ = self.coordinate_transform.setup_coordinate_system(
                self.raw_test_data.reshape(-1, self.dim)
            )

        if n_threads > 1:
            self.p = TransformedBoltzmannParallel(
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
                integrator(temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                mm.Platform.getPlatformByName("Reference"),
            )
            self.p = TransformedBoltzmann(
                sim.context,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,
            )

    @staticmethod
    def load_target_data(data_path: pathlib.Path, dim: int):
        # TODO: What filetype are MD samples usually saved in? Support this.
        if data_path.suffix == ".h5":
            with h5py.File(str(data_path), "r") as f:
                target_data = torch.from_numpy(f["coordinates"][()])
        elif data_path.suffix == ".pt":
            target_data = torch.load(str(data_path))
            assert len(target_data.shape) == 2, "Data must be of shape (num_frames, dim)."
        elif data_path.suffix == ".pdb":  # TODO: This seems very slow for big files?
            warnings.warn("Loading MD samples from .pdb file. This is very slow. Use .pt or .h5 instead.")
            pdb = app.PDBFile(str(data_path))
            target_data = []
            for i in range(pdb.getNumFrames()):
                # in nanometers
                frame = torch.from_numpy(np.array(pdb.getPositions(asNumpy=True, frame=i))).reshape(-1, dim)
                target_data.append(frame)
            target_data = torch.cat(target_data, dim=0)
        else:
            raise ValueError(
                "Cannot load MD samples file with suffix: {}. Must be .pt or .pdb".format(data_path.suffix)
            )
        return target_data

    def log_prob(self, z: Tensor):
        # Add global potential for implicit solvent here?
        return self.p.log_prob(z)

    def log_prob_x(self, x: Tensor):
        return self.p.log_prob_x(x)

    def performance_metrics(
            self,
            samples: Optional[Tensor] = None,
            log_w: Optional[Tensor] = None,
            log_q_fn: Callable = None,
            batch_size: int = 1000,
            iteration: Optional[int] = None
    ):
        # This function is typically called both with Flow (likelihood available) and with Flow+AIS
        # samples (no likelihood available).
        summary_dict = {}

        # TODO: Fix self.get_kld_info() to do some kind of eval when we don't have a flow likelihood. Maybe use
        #  aldp.py's evaluate_aldp as an example.
        # These are saved to disk, rather than to a dictionary.
        # if self.metric_dir is not None and samples is not None and log_w is not None:
        #     with torch.no_grad():
        #         self.get_kld_info(samples, log_w, batch_size=batch_size, iteration=iteration)

        if log_q_fn:  # Evaluate base flow samples: likelihood available.
            if self.eval_mode == "val":  # TODO: batch this?
                target_data = self.val_data.reshape(-1, self.dim).to(self.device)
            elif self.eval_mode == "test":
                target_data = self.test_data.reshape(-1, self.dim).to(self.device)
            with torch.no_grad():  # TODO: Is this correct?
                log_q_test = log_q_fn(target_data)  # log_q_fn is the log_prob(x) function of the flow.
                # TODO: This gives huge negative numbers. Better if we multiply by factor 10: units are wrong
                #  between MD and simulator? Seems unlikely, since both use the same WaterInWaterBox system...
                log_p_test = self.log_prob_x(target_data)  # logprob of MD data under target distribution.
                kl_forward = torch.mean(log_p_test - log_q_test)
                # ESS normalised by true p samples: this presumably gives a metric of how well the flow is covering p.
                #  In particular, this is the version of ESS that should be less spurious if the flow is missing modes.
                ess_over_p = effective_sample_size_over_p(log_p_test - log_q_test)
                test_mean_log_prob = torch.mean(log_q_test)
            summary_dict.update({
                "flow_test_log_prob": test_mean_log_prob.cpu().item(),
                "flow_ess_over_p": ess_over_p.cpu().item(),
                "flow_kl_forward": kl_forward.cpu().item(),
            })
        else:  # Evaluate Flow+AIS samples: no likelihood available.
            pass  # There is no evaluation currently that only works for Flow+AIS samples.
        return summary_dict

    def get_kld_info(
            self,
            samples: Optional[Tensor] = None,
            log_w: Optional[Tensor] = None,
            batch_size: int = 1000,
            iteration: Optional[int] = None,
    ):
        """
        Computes the KLD between the target distribution and the flow distribution, and saves the KLD histogram to
        disk. Uses sample histograms to estimate the KLD, as in the original ALDP code. Using samples means we don't
        need the likelihood, so we can actually evaluate Flow+AIS samples as well.
        """
        raise NotImplementedError("This function is not yet correctly implemented for H2OinH2O.")

        # NOTE: These are x, but they call it z. Note when comparing the original code in utils/aldp.py that their
        #  transforms have the opposite forward/inverse convention to ours.
        assert iteration, "Must pass iteration number for doing KLD histogram."

        z_test = self.target_data
        z_sample = samples  # TODO: might need to fix PeriodicWrap for this (see last layer in Flow and Vincent email).

        # Determine likelihood of test data and transform it (mostly taken from aldp.py)
        z_d_np = z_test
        x_d_np = torch.zeros(0, self.dim)
        log_p_sum = 0
        n_batches = int(np.ceil(len(z_test) / batch_size))
        for i in range(n_batches):
            if i == n_batches - 1:
                end = len(z_test)
            else:
                end = (i + 1) * batch_size
            z = z_test[(i * batch_size): end, :]
            # TODO: I think this fixes the PeriodicWrap issue (see email Vincent).
            # TODO: This gives an error in get_angle_and_normal() that finds a rotation around and axis with x=0. This
            #  presumably happens because one of the test data points has a non-solute atom with y=0. This can indeed
            #  happen (although exactly 0 is kind of strange), so we should figure out how to deal with it. What would
            #  work, is to pick a convention for the case where x=0 (i.e., pick a rotation direction,
            #  such as the x>0 direction), and make sure that this convention is used when the code rotates our
            #  molecule (otherwise we rotate wrong!).
            # TODO: Error in batch i=190?
            x, log_det = self.coordinate_transform.inverse(z.double())  # Actually: X --> Z  # TODO: Check phi/theta.
            x_d_np = torch.cat((x_d_np, x), dim=0)
            log_p = self.log_prob(z)  # TODO: This is the logprob under the target, of the z-space samples?! Weird.
            log_p_sum = log_p_sum + torch.sum(log_p).detach() - torch.sum(log_det).detach().float()
        log_p_avg = log_p_sum / len(z_test)

        # Transform samples
        z_np = torch.zeros(0, self.dim)
        x_np = torch.zeros(0, self.dim)
        n_batches = int(np.ceil(len(samples) / batch_size))
        for i in range(n_batches):
            if i == n_batches - 1:
                end = len(z_sample)
            else:
                end = (i + 1) * batch_size
            z = z_sample[(i * batch_size): end, :]
            # TODO: I think this fixes the PeriodicWrap issue (see email Vincent).
            x, _ = self.coordinate_transform.inverse(z.double())  # Actually: X --> Z  # TODO: Check phi/theta.
            x_np = torch.cat((x_np, x), dim=0)
            z, _ = self.coordinate_transform.forward(x)  # Actually: Z --> X
            z_np = torch.cat((z_np, z), dim=0)

        # To numpy
        z_np = z_np.cpu().data.numpy()
        z_d_np = z_d_np.cpu().data.numpy()

        # Estimate density of marginals
        nbins = 200
        hist_range = [-5, 5]
        ndims = z_np.shape[1]

        hists_test = np.zeros((nbins, ndims))
        hists_gen = np.zeros((nbins, ndims))

        for i in range(ndims):
            htest, _ = np.histogram(z_d_np[:, i], nbins, range=hist_range, density=True)
            hgen, _ = np.histogram(z_np[:, i], nbins, range=hist_range, density=True)
            hists_test[:, i] = htest
            hists_gen[:, i] = hgen

        # Compute KLD of marginals
        eps = 1e-10
        kld_unscaled = np.sum(hists_test * np.log((hists_test + eps) / (hists_gen + eps)), axis=0)
        kld = kld_unscaled * (hist_range[1] - hist_range[0]) / nbins

        # Split KLD into groups
        r_ind = np.arange(0, self.dim, 3)
        phi_ind = np.arange(1, self.dim, 3)
        theta_ind = np.arange(2, self.dim, 3)

        kld_r = kld[r_ind]
        kld_phi = kld[phi_ind]
        kld_theta = kld[theta_ind]

        # Calculate and save KLD stats of marginals
        kld = (kld_r, kld_phi, kld_theta)
        kld_labels = ["bond", "angle", "dih"]
        kld_ = np.concatenate(kld)
        kld_append = np.array([[iteration + 1, np.median(kld_), np.mean(kld_)]])
        kld_path = os.path.join(self.metric_dir, "kld.csv")
        if os.path.exists(kld_path):
            kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=",")
            if len(kld_hist.shape) == 1:
                kld_hist = kld_hist[None, :]
            kld_hist = np.concatenate([kld_hist, kld_append])
        else:
            kld_hist = kld_append
        np.savetxt(kld_path, kld_hist, delimiter=",", header="it,kld_median,kld_mean", comments="")

        # Save KLD per coordinate (r, phi, theta)
        for kld_label, kld_ in zip(kld_labels, kld):
            kld_append = np.concatenate([np.array([iteration + 1, np.median(kld_), np.mean(kld_)]), kld_])
            kld_append = kld_append[None, :]
            kld_path = os.path.join(self.metric_dir, "kld_" + kld_label + ".csv")
            if os.path.exists(kld_path):
                kld_hist = np.loadtxt(kld_path, skiprows=1, delimiter=",")
                if len(kld_hist.shape) == 1:
                    kld_hist = kld_hist[None, :]
                kld_hist = np.concatenate([kld_hist, kld_append])
            else:
                kld_hist = kld_append
            header = "it,kld_median,kld_mean"
            for kld_ind in range(len(kld_)):
                header += ",kld" + str(kld_ind)
            np.savetxt(kld_path, kld_hist, delimiter=",", header=header, comments="")

        # Save log probability
        log_p_append = np.array([[iteration + 1, log_p_avg]])
        log_p_path = os.path.join(self.metric_dir, "log_p_test.csv")
        if os.path.exists(log_p_path):
            log_p_hist = np.loadtxt(log_p_path, skiprows=1, delimiter=",")
            if len(log_p_hist.shape) == 1:
                log_p_hist = log_p_hist[None, :]
            log_p_hist = np.concatenate([log_p_hist, log_p_append])
        else:
            log_p_hist = log_p_append
        np.savetxt(log_p_path, log_p_hist, delimiter=",", header="it,log_p", comments="")

        return  # TODO: Return anything to save in the dict or just save to disk?
