import h5py
import warnings
import pathlib
import json
import os
from typing import Optional, List, Dict, Any, Callable, Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from torch import nn
from torch import Tensor
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools.testsystems import TestSystem

from fab.utils.logging import Logger
from fab.target_distributions.base import TargetDistribution
from fab.target_distributions.boltzmann import TransformedBoltzmann, TransformedBoltzmannParallel
from fab.transforms.global_3point_spherical_transform import Global3PointSphericalTransform


constraints_dict = {
    "hbonds": app.HBonds,
    "none": None,
}


class TriatomicInWaterSys(TestSystem):
    """
    Triatomic molecule in water OpenMM system setup.

    Parameters
    ----------
    solute_pdb_path: str, path to solute pdb file. If provided, incprd and prmtop files will be ignored.
    solute_xml_path: str, path to solute xml file. If provided, this forcefield will be used in addition to the
        base forcefield (tip3p).
    solute_inpcrd_path: str, path to solute inpcrd file. Only used if pdb file is not provided. Usage is currently not
        fully implemented.
    solute_prmtop_path: str, path to solute prmtop file. Only used if pdb file is not provided. Usage is currently not
        fully implemented.
    dim: int, dimensionality of system (num_atoms x 3).
    external_constraints: bool, whether to use external force constraints for keeping the system in place.
    internal_constraints: str, internal constraints to use. E.g. "hbonds" (restricts hydrogen atom bond lengths)
        or "none". Should be "none" during training, since otherwise the energy of flow samples cannot be properly
        computed (will only be able to compute a projection of these samples onto 'valid' configurations).
    rigidwater: bool, whether to use rigid water molecules. If False, the water molecules will be flexible. Should
        False during training, for reasons mentioned in `internal_constraints` docstring.
    """
    def __init__(
            self,
            solute_pdb_path: str,
            solute_xml_path: str,
            solute_inpcrd_path: str,
            solute_prmtop_path: str,
            dim: int,
            external_constraints: bool,
            internal_constraints: str,
            rigidwater: bool,
            **kwargs,
    ):
        TestSystem.__init__(self, **kwargs)
        # http://docs.openmm.org/latest/userguide/application/02_running_sims.html
        self.solute_pdb_path = solute_pdb_path
        self.solute_xml_path = solute_xml_path
        self.solute_inpcrd_path = solute_inpcrd_path
        self.solute_prmtop_path = solute_prmtop_path
        self.dim = dim
        self.external_constraints = external_constraints
        self.internal_constraints = internal_constraints
        self.rigidwater = rigidwater

        self.num_atoms_per_solute = 3  # Triatomic
        self.num_atoms_per_solvent = 3  # Water
        self.num_solvent_molecules = (dim - self.num_atoms_per_solute) // (self.num_atoms_per_solvent * 3)

        # Steps to take:
        # 1. Load topology of solute.
        # 2. Solvate the solute.
        # 3. Add the solute and solvent force fields.
        # 4. Add the implicit solvent force field / external potential term.

        if solute_pdb_path is not None and solute_inpcrd_path is not None and solute_prmtop_path is not None:
            warnings.warn("Found path to .pdb, .inpcrd and .prmtop files. Will use .pdb file.")
        # Initial solute molecule
        if solute_pdb_path is not None:
            pdb = app.PDBFile(solute_pdb_path)  # This can be any triatomic solute
            # This pdb file has a single water molecule, where the OH bonds are 0.0957 nm in length.
            modeller = app.modeller.Modeller(pdb.topology, pdb.positions)  # In nanometers
            forcefield = app.ForceField("amber14/tip3p.xml")  # tip3pfb
            # ‘tip3p’, ‘spce’, ‘tip4pew’, ‘tip5p’, ‘swm4ndp’
            if solute_xml_path is not None:
                forcefield.loadFile(solute_xml_path)
            # Add solvent
            if self.num_solvent_molecules > 0:
                # TODO: set padding=1.0 * unit.nanometers ?
                # TODO: set boxSize=mm.Vec3(3.105, 3.105, 3.105) * unit.nanometers ?
                modeller.addSolvent(forcefield, model="tip3p", numAdded=self.num_solvent_molecules)
            # Create system
            self.system = forcefield.createSystem(  # Create system from forcefield
                modeller.topology,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=1.0 * unit.nanometers,
                constraints=constraints_dict[internal_constraints],  # `"none"` for flexible H2O
                rigidWater=rigidwater,  # `False` for flexible H2O
            )
        elif solute_inpcrd_path is not None and solute_prmtop_path is not None:
            # TODO: Not fully implemented!
            #  After adding solvent, the system can be created in two ways:
            #  1) Using the solventForceField.createSystem() method to ensure that the force field parameters are
            #     applied not only to the solute but also to the solvent.
            #  2) Using the .prmtop.createSystem() to ensure consistency with AMBER parameters.
            #  Neither works, because neither method uses a forcefield that contains parameters for both SO2 and H2O.
            #  Potential solution: create xml file that contains both forcefields. Or create AMBER system with solvent
            #  molecules already present (latter is easier, but less flexible, because we have to predetermine the
            #  number of solvent molecules).
            # Input coordinates
            inpcrd = app.AmberInpcrdFile(solute_inpcrd_path)
            # Parameters/topology
            prmtop = app.AmberPrmtopFile(solute_prmtop_path)
            # Create modeller
            modeller = app.Modeller(prmtop.topology, inpcrd.positions)
            raise NotImplementedError("Using inpcrd+prmtop file is not fully implemented.")
        else:
            raise ValueError("Must provide either a .pdb file with optional .xml file, or .inpcrd and .prmtop files.")

        if external_constraints:
            # This keeps the first atom around the origin.
            center = mm.CustomExternalForce('k*r^2; r=sqrt(x*x+y*y+z*z)')
            center.addGlobalParameter("k", 100000.0)
            # center.addGlobalParameter("k", 1.0)
            self.system.addForce(center)
            center.addParticle(0, [])

            # Add spherical restraint to hold the droplet
            force = mm.CustomExternalForce('w*max(0, r-1.0)^2; r=sqrt(x*x+y*y+z*z)')
            force.addGlobalParameter("w", 100.0)
            self.system.addForce(force)
            for i in range(self.system.getNumParticles()):
                force.addParticle(i, [])

        self.topology, self.positions = modeller.getTopology(), modeller.getPositions()
        # self.topology.atoms() yields the atom order, which is OHH OHH OHH etc.
        # This is the order in which the coordinates are stored in the positions array.
        self.atoms = [atom.name for atom in self.topology.atoms()]


class SoluteInWater(nn.Module, TargetDistribution):
    """
    Boltzmann distribution of a solute in water.

    Parameters
    ----------
    solute_pdb_path: str, path to solute pdb file. If provided, incprd and prmtop files will be ignored.
    solute_xml_path: str, path to solute xml file. If provided, this forcefield will be used in addition to the
        base forcefield (tip3p).
    solute_inpcrd_path: str, path to solute inpcrd file. Only used if pdb file is not provided. Usage is currently not
        fully implemented.
    solute_prmtop_path: str, path to solute prmtop file. Only used if pdb file is not provided. Usage is currently not
        fully implemented.
    dim: int, dimensionality of system (num_atoms x 3).
    temperature: float, temperature of system in Kelvin.
    energy_cut: float, energy cut-off for Boltzmann distribution (logarithmic above this value).
    energy_max: float, maximum energy for Boltzmann distribution (capped at this value).
    n_threads: int, number of threads to use for parallel Boltzmann evaluation.
    train_samples_path: str, path to MD training samples file.
    val_samples_path: str, path to MD validation samples file.
    test_samples_path: str, path to MD test samples file.
    eval_mode: Literal["val", "test"], evaluation mode for performance metrics.
    use_val_data_for_transform: bool, whether to use validation data to set scale for coordinate transform.
    device: str, device to use for computation.
    logger: Logger, logger object for logging.
    save_dir: str, directory to save metrics and plots.
    plot_MD_energies: bool, whether to plot the energies of the MD data as a sanity check. Mostly used for debugging.
    plot_marginal_hists: bool, whether to plot the marginal histograms of the MD data vs Flow data. Mostly used for
        debugging.
    external_constraints: bool, whether to use external force constraints for keeping the system in place.
    internal_constraints: str, internal constraints to use. E.g. "hbonds" (restricts hydrogen atom bond lengths)
        or "none". Should be "none" during training, since otherwise the energy of flow samples cannot be properly
        computed (will only be able to compute a projection of these samples onto 'valid' configurations).
    rigidwater: bool, whether to use rigid water molecules. If False, the water molecules will be flexible. Should
        False during training, for reasons mentioned in `internal_constraints` docstring.
    """
    def __init__(
        self,
        solute_pdb_path: str,
        solute_xml_path: str,
        solute_inpcrd_path: str,
        solute_prmtop_path: str,
        dim: int = 3 * (3 + 3 * 8),
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
        logger: Logger = None,
        save_dir: Optional[str] = None,
        plot_MD_energies: bool = False,
        plot_marginal_hists: bool = False,
        external_constraints: bool = True,
        internal_constraints: str = "none",
        rigidwater: bool = False,
    ):
        super(SoluteInWater, self).__init__()

        self.cartesian_dim = dim
        self.internal_dim = dim - 6
        self.temperature = temperature
        self.energy_cut = energy_cut
        self.energy_max = energy_max
        self.n_threads = n_threads
        self.device = device

        self.plot_MD_energies = plot_MD_energies
        self.plot_marginal_hists = plot_marginal_hists

        self.logger = logger
        self.save_dir = save_dir
        self.metric_dir = os.path.join(self.save_dir, f"metrics")
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)

        # Load any MD data
        self.eval_mode = eval_mode
        self.train_samples_path, self.val_samples_path, self.test_samples_path = None, None, None
        self.train_data_config, self.val_data_config, self.test_data_config = None, None, None
        self.train_data_x, self.val_data_x, self.test_data_x = None, None, None
        self.train_data_i, self.val_data_i, self.test_data_i = None, None, None
        self.train_logdet_xi, self.val_logdet_xi, self.test_logdet_xi = None, None, None
        if train_samples_path:
            self.train_samples_path = pathlib.Path(train_samples_path)
            # OH bonds still ~0.1 nm in length for this data.
            self.train_data_x = self.load_target_data(self.train_samples_path, self.cartesian_dim).double()
            # Load associated config
            with open(self.train_samples_path.with_suffix(".json"), "r") as f:
                self.train_data_config = json.load(f)
        if val_samples_path:
            self.val_samples_path = pathlib.Path(val_samples_path)
            self.val_data_x = self.load_target_data(self.val_samples_path, self.cartesian_dim).double()
            # Load associated config
            with open(self.val_samples_path.with_suffix(".json"), "r") as f:
                self.val_data_config = json.load(f)
        if test_samples_path:
            self.test_samples_path = pathlib.Path(test_samples_path)
            self.test_data_x = self.load_target_data(self.test_samples_path, self.cartesian_dim).double()
            # Load associated config
            with open(self.test_samples_path.with_suffix(".json"), "r") as f:
                self.test_data_config = json.load(f)

        # Initialise system
        self.system = TriatomicInWaterSys(
            solute_pdb_path,
            solute_xml_path,
            solute_inpcrd_path,
            solute_prmtop_path,
            self.cartesian_dim,
            external_constraints,
            internal_constraints,
            rigidwater,
        )

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
            transform_data = torch.tensor(position.reshape(1, self.cartesian_dim)).double()
            del traj_sim
            self.transform_data = transform_data
        else:
            self.transform_data = self.val_data_x.clone()[0].reshape(1, self.cartesian_dim)

        assert self.transform_data.shape[-1] == self.cartesian_dim, (
            f"Data shape ({self.transform_data.shape}) does not match number of "
            f"coordinates in current system ({self.cartesian_dim})."
        )

        self.coordinate_transform = Global3PointSphericalTransform(self.system, self.transform_data.to(device))

        # Transform MD data to internal coordinates (X --> I): these are the coordinates that we feed into the flow on
        #  its output end.
        if self.train_data_x is not None:
            # OH bonds are still ~0.1 nm apart
            self.train_data_i, self.train_logdet_xi = self.coordinate_transform.inverse(
                self.train_data_x.reshape(-1, self.cartesian_dim)  # Transform expects flattened coordinates
            )
        if self.val_data_x is not None:
            self.val_data_i, self.val_logdet_xi = self.coordinate_transform.inverse(
                self.val_data_x.reshape(-1, self.cartesian_dim)
            )
        if self.test_data_x is not None:
            self.test_data_i, self.test_logdet_xi = self.coordinate_transform.inverse(
                self.test_data_x.reshape(-1, self.cartesian_dim)
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
        """
        Load MD samples from file.
        """
        if data_path.suffix == ".h5":
            with h5py.File(str(data_path), "r") as f:
                target_data = torch.from_numpy(f["coordinates"][()])
        elif data_path.suffix == ".pt":
            target_data = torch.load(str(data_path))
            assert len(target_data.shape) == 2, "Data must be of shape (num_frames, dim)."
        elif data_path.suffix == ".pdb":
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

    def log_prob(self, i: Tensor):
        return self.p.log_prob(i)  # I --> X, then unnormalised logprob.

    def log_prob_and_jac(self, i: Tensor):
        return self.p.log_prob_and_jac(i)  # I --> X, then unnormalised logprob and Jacobian.

    def log_prob_x(self, x: Tensor):
        return self.p.log_prob_x(x)  # Direct unnormalised logprob on Cartesian (X) data.

    def performance_metrics(
            self,
            samples: Optional[Tensor] = None,
            log_w: Optional[Tensor] = None,
            log_q_fn: Callable = None,
            batch_size: int = 1000,
            iteration: Optional[int] = None,
            flow: Optional[nn.Module] = None,
    ):
        """
        Compute performance metrics for the target distribution. Used by the main training loop for evaluation.

        Parameters
        ----------
        NOTE: Some parameters are not used in this function, but are still passed for compatibility with the original
        codebase.

        samples: Optional[Tensor], samples from the target distribution. If None, samples are generated using the flow.
        log_w: Optional[Tensor], log weights for AIS samples. If None, AIS samples are not used. Currently, this
            function is not used for AIS evaluation.
        log_q_fn: Callable, log probability function of the flow. Used for computing log probability of MD data under
            the current flow. Split off from the `flow` argument for backwards compatibility.
        batch_size: int, batch size for evaluation. Currently unused.
        iteration: Optional[int], iteration number. Used for logging metrics.
        flow: Optional[nn.Module], flow model. Used for generating samples if none are provided.
        """
        # TODO: batch_size here is currently equal to inner_batch_size in core.get_eval_info(), which
        #  corresponds to the training batch size. This is because eval_batch size is used for determining
        #  how many eval datapoints to use in total in the original code. This all works out when using this
        #  batch_size for AIS evaluation, since AIS samples are generated outside of this function.
        #  Actually, batch size is not used right now.

        # This function is typically called both with Flow (likelihood available for samples) and with Flow+AIS
        # samples (no likelihood available for Flow+AIS samples).
        summary_dict = {}
        # Load MD data for evaluation
        if self.eval_mode == "val":  # TODO: batch this?
            target_data_x = self.val_data_x.reshape(-1, self.cartesian_dim).to(self.device)
            target_data_i = self.val_data_i.reshape(-1, self.internal_dim).to(self.device)
            target_logdet_xi = self.val_logdet_xi.to(self.device)
        elif self.eval_mode == "test":
            target_data_x = self.test_data_x.reshape(-1, self.cartesian_dim).to(self.device)
            target_data_i = self.test_data_i.reshape(-1, self.internal_dim).to(self.device)
            target_logdet_xi = self.test_logdet_xi.to(self.device)
        else:
            raise ValueError("Invalid eval_mode. Must be 'val' or 'test'.")

        # Log_prob of flow given, so use this for evaluating the log probability of MD data.
        if log_q_fn:
            # Log_prob of target data under flow
            with torch.no_grad():
                # log_q_fn is the log_prob function of the flow.
                log_q_test = log_q_fn(target_data_i) + target_logdet_xi
            test_mean_log_prob = torch.mean(log_q_test)
            summary_dict.update({"flow_test_log_prob": test_mean_log_prob.cpu().item()})

        # Use flow samples for computing marginal KL estimates.
        if samples is None:  # No samples provided, so generate using flow.
            assert flow, (
                "Flow model must be provided for generating evaluation samples if none are provided."
            )
            num_flow_samples = len(target_data_i)  # Use same number of Flow and MD samples.
            with torch.no_grad():
                flow_samples, _ = flow.sample_and_log_prob((num_flow_samples,))
        else:  # Samples provided (can be Flow or Flow+AIS samples).
            flow_samples = samples

        # Estimate the KL per dimension using the provided samples; we can normalise this using the samples
        # themselves. Here we're essentially estimating the KL as the average of marginal KLs (per dimension).
        # TODO: Possibly batch this if necessary?
        nbins = 200
        hist_range = [-5, 5]
        target_data_kl = target_data_i.cpu().clone().numpy()
        flow_samples_kl = flow_samples.cpu().clone().numpy()
        hists_test = np.zeros((nbins, self.internal_dim))
        hists_flow = np.zeros((nbins, self.internal_dim))
        for dim in range(self.internal_dim):
            # TODO: KL in I space or in X space? Also, what to do with the Jacobian term then?
            #  Jacobian term is unnecessary in KL estimate, since we use samples + histograms, rather
            #  than the log probability density.
            hist_test, _ = np.histogram(target_data_kl[:, dim], bins=nbins, range=hist_range, density=True)
            hist_flow, _ = np.histogram(flow_samples_kl[:, dim], bins=nbins, range=hist_range, density=True)
            hists_test[:, dim] = hist_test
            hists_flow[:, dim] = hist_flow
        # KL of marginals
        eps = 1e-10
        forward_kl_marginals_unscaled = np.sum(
            hists_test * (np.log(hists_test + eps) - np.log(hists_flow + eps)), axis=0
        )
        forward_kl_marginals = forward_kl_marginals_unscaled * (hist_range[1] - hist_range[0]) / nbins
        reverse_kl_marginals_unscaled = np.sum(
            hists_flow * (np.log(hists_flow + eps) - np.log(hists_test + eps)), axis=0
        )
        reverse_kl_marginals = reverse_kl_marginals_unscaled * (hist_range[1] - hist_range[0]) / nbins
        summary_dict.update({
            "mean_forward_kl_marginals": forward_kl_marginals.mean(),
            "mean_reverse_kl_marginals": reverse_kl_marginals.mean(),
        })

        if summary_dict:
            with open(os.path.join(self.metric_dir, f"metrics_{iteration}.json"), "w") as f:
                json.dump(summary_dict, f)
        else:
            warnings.warn("No summary metrics were computed.")

        return summary_dict