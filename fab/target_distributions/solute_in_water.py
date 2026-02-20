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
import openmm as mm
from openmm import unit
from openmm import app
from openmmtools.testsystems import TestSystem

from fab.utils.logging import Logger
from fab.target_distributions.base import TargetDistribution
from fab.target_distributions.boltzmann import TransformedBoltzmann, TransformedBoltzmannParallel
from fab.transforms.global_3point_spherical_transform import Global3PointSphericalTransform
from fab.target_distributions.pbc_water_preprocess import preprocess_frame_batch, IdentityTransform


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
    rigid_water: bool, whether to use rigid water molecules. If False, the water molecules will be flexible. Should
        False during training, for reasons mentioned in `internal_constraints` docstring.
    constraint_radius: float, radius (in nm) of the spherical constraint for keeping the droplet in place.
        Should be rounded to whole Angstrom.
    constraint_force: float, force constant for the spherical droplet constraint.
    """
    def __init__(
            self,
            solute_pdb_path: str,
            solute_xml_path: str,
            solute_inpcrd_path: str,
            solute_prmtop_path: str,
            num_solvent_molecules: int,
            boundary_condition: str,
            box_length_nm: float,
            nonbonded_cutoff_nm: float,
            rigid_water: bool,
            internal_constraints: str,
            external_constraints: bool,
            constraint_radius: float,
            constraint_force: float,
            **kwargs,
    ):
        TestSystem.__init__(self, **kwargs)
        # http://docs.openmm.org/latest/userguide/application/02_running_sims.html
        self.solute_pdb_path = solute_pdb_path
        self.solute_xml_path = solute_xml_path
        self.solute_inpcrd_path = solute_inpcrd_path
        self.solute_prmtop_path = solute_prmtop_path
        self.num_solvent_molecules = int(num_solvent_molecules)
        self.boundary_condition = boundary_condition
        self.box_length_nm = box_length_nm
        self.internal_constraints = internal_constraints
        self.rigid_water = rigid_water
        self.external_constraints = external_constraints
        self.constraint_radius = constraint_radius
        self.constraint_force = constraint_force
        self.nonbonded_cutoff_nm = nonbonded_cutoff_nm
        self.num_atoms_per_solute = 3  # Triatomic
        self.num_atoms_per_solvent = 3  # Water

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
            modeller = app.Modeller(pdb.topology, pdb.positions)  # In nanometers
            forcefield = app.ForceField("amber14/tip3p.xml")  # tip3pfb
            # forcefield = app.ForceField('amber19-all.xml', 'amber19/tip3pfb.xml')
            # ‘tip3p’, ‘spce’, ‘tip4pew’, ‘tip5p’, ‘swm4ndp’
            if solute_xml_path is not None:
                forcefield.loadFile(solute_xml_path)
            
            # Add solvent based on num_solvent_molecules
            if self.num_solvent_molecules > 0:
                modeller.addSolvent(forcefield, model="tip3p", numAdded=self.num_solvent_molecules)

            if self.boundary_condition == "droplet":
                # Create system
                self.system = forcefield.createSystem(  # Create system from forcefield
                    modeller.topology,
                    nonbondedMethod=app.CutoffNonPeriodic,
                    nonbondedCutoff=self.nonbonded_cutoff_nm * unit.nanometers,
                    constraints=constraints_dict[self.internal_constraints],  # `"none"` for flexible H2O
                    rigidWater=self.rigid_water,  # `False` for flexible H2O
                )
                # External constraints
                if self.external_constraints:
                    self._add_external_constraints()

            elif self.boundary_condition == "pbc":
                if self.box_length_nm is None:
                    raise ValueError("box_length_nm must be set for PBC.")
                if self.nonbonded_cutoff_nm > 0.5 * self.box_length_nm:
                    raise ValueError("For PBC, require nonbonded_cutoff_nm <= box_length_nm/2.")

                # Add solvent based on num_solvent_molecules which is calculated with the density.
                L = self.box_length_nm
                a = mm.Vec3(L, 0, 0)
                b = mm.Vec3(0, L, 0)
                c = mm.Vec3(0, 0, L)
                modeller.topology.setPeriodicBoxVectors((a, b, c) * unit.nanometers)
                
                # Create system
                self.system = forcefield.createSystem(
                    modeller.topology,
                    nonbondedMethod=app.PME,               
                    nonbondedCutoff=self.nonbonded_cutoff_nm * unit.nanometers,
                    constraints=constraints_dict[self.internal_constraints],
                    rigidWater=self.rigid_water,
                )
            else:
                raise ValueError(f"Invalid boundary_condition: {self.boundary_condition}. Must be 'droplet' or 'pbc'.")
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


        self.topology, self.positions = modeller.getTopology(), modeller.getPositions()
        # self.topology.atoms() yields the atom order, which is OHH OHH OHH etc.
        # This is the order in which the coordinates are stored in the positions array.
        self.atoms = [atom.name for atom in self.topology.atoms()]
    
    def _add_external_constraints(self):
        # This keeps the first atom around the origin.
        center = mm.CustomExternalForce('k*r^2; r=sqrt(x*x+y*y+z*z)')
        center.addGlobalParameter("k", 100000.0)
        self.system.addForce(center)
        center.addParticle(0, [])

        # Add spherical restraint to hold the droplet
        force = mm.CustomExternalForce('w*max(0, r-{:.1f})^2; r=sqrt(x*x+y*y+z*z)'.format(self.constraint_radius))
        force.addGlobalParameter("w", self.constraint_force)
        self.system.addForce(force)
        for i in range(self.system.getNumParticles()):
            force.addParticle(i, [])
        


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
    rigid_water: bool, whether to use rigid water molecules. If False, the water molecules will be flexible. Should
        False during training, for reasons mentioned in `internal_constraints` docstring.
    constraint_radius: float, radius (in nm) of the spherical constraint for keeping the droplet in place.
        Should be rounded to whole Angstrom.
    constraint_force: float, force constant for the spherical droplet constraint.
    """
    def __init__(
        self,
        solute_pdb_path: str,
        solute_xml_path: str,
        solute_inpcrd_path: str,
        solute_prmtop_path: str,
        dim: int = 3 * (3 + 3 * 8),
        num_solvent_molecules: int = 5,
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
        boundary_condition: str = "droplet",
        box_length_nm: float = 3.105,
        nonbonded_cutoff_nm: float = 1.0,
        rigid_water: bool = False,
        internal_constraints: str = "none",
        external_constraints: bool = False,
        constraint_radius: float = 1.0,
        constraint_force: float = 10000.0,
        platform_name: str = None,
        platform_properties: Optional[Dict[str, str]] = None,
    ):
        super(SoluteInWater, self).__init__()

        self.num_solvent_molecules = num_solvent_molecules

        self.cartesian_dim = dim
        self.temperature = temperature
        self.energy_cut = energy_cut
        self.energy_max = energy_max
        self.n_threads = n_threads
        self.device = device

        self.plot_MD_energies = plot_MD_energies
        self.plot_marginal_hists = plot_marginal_hists
        self.boundary_condition = boundary_condition
        self.box_length_nm = box_length_nm

        if self.boundary_condition == "pbc":
            self.internal_dim = self.cartesian_dim
        else:
            self.internal_dim = self.cartesian_dim - 6

        self.logger = logger
        self.save_dir = save_dir
        self.metric_dir = os.path.join(self.save_dir, f"metrics")
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)

        # OpenMM platform
        self.platform_name = platform_name
        self.platform_properties = platform_properties

        # Load any MD data
        self.eval_mode = eval_mode
        # self.train_samples_path, self.val_samples_path, self.test_samples_path = None, None, None
        self.train_data_config, self.val_data_config, self.test_data_config = None, None, None
        self.train_data_x, self.val_data_x, self.test_data_x = None, None, None
        self.train_data_i, self.val_data_i, self.test_data_i = None, None, None
        self.train_logdet_xi, self.val_logdet_xi, self.test_logdet_xi = None, None, None
        
        print("Loading MD data...", flush=True)
        if train_samples_path:
            train_samples_path = pathlib.Path(train_samples_path)
            # OH bonds still ~0.1 nm in length for this data.
            self.train_data_x = self.load_target_data(train_samples_path, self.cartesian_dim).double()

        if val_samples_path:
            val_samples_path = pathlib.Path(val_samples_path)
            self.val_data_x = self.load_target_data(val_samples_path, self.cartesian_dim).double()

        if test_samples_path:
            test_samples_path = pathlib.Path(test_samples_path)
            self.test_data_x = self.load_target_data(test_samples_path, self.cartesian_dim).double()

        # Initialise system
        self.system = TriatomicInWaterSys(
            solute_pdb_path,
            solute_xml_path,
            solute_inpcrd_path,
            solute_prmtop_path,
            self.num_solvent_molecules,
            boundary_condition,
            box_length_nm,
            nonbonded_cutoff_nm,
            rigid_water,
            internal_constraints,
            external_constraints,
            constraint_radius,
            constraint_force,
        )


        # Generate trajectory for coordinate transform if no data path is specified
        integrator = mm.LangevinMiddleIntegrator
        print(val_samples_path)
        if not val_samples_path or not use_val_data_for_transform:
            traj_sim = app.Simulation(
                self.system.topology,
                self.system.system,
                integrator(temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                mm.Platform.getPlatformByName(self.platform_name),
                self.platform_properties,
            )
            print("OpenMM platform:", traj_sim.context.getPlatform().getName())
   
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
        if self.boundary_condition == "droplet":
            self.coordinate_transform = Global3PointSphericalTransform(self.system, self.transform_data.to(device))
        elif self.boundary_condition == "pbc":
            print("For pbc we dont use a transform, so we load an identity transform.")
            self.coordinate_transform = IdentityTransform()
        else:
            raise ValueError(f"Invalid boundary_condition: {self.boundary_condition}. Must be 'droplet' or 'periodic'.")
        
        if self.boundary_condition == "droplet":
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
        else:
            # PBC: I == X, logdet == 0
            if self.train_data_x is not None:
                self.train_data_i = self.train_data_x.reshape(-1, self.cartesian_dim).to(self.device)
                self.train_logdet_xi = torch.zeros(self.train_data_i.shape[0], device=self.device, dtype=self.train_data_i.dtype)
            if self.val_data_x is not None:
                self.val_data_i = self.val_data_x.reshape(-1, self.cartesian_dim).to(self.device)
                self.val_logdet_xi = torch.zeros(self.val_data_i.shape[0], device=self.device, dtype=self.val_data_i.dtype)
            if self.test_data_x is not None:
                self.test_data_i = self.test_data_x.reshape(-1, self.cartesian_dim).to(self.device)
                self.test_logdet_xi = torch.zeros(self.test_data_i.shape[0], device=self.device, dtype=self.test_data_i.dtype)

        # Target distribution wrapper
        if n_threads > 1:
            self.p = TransformedBoltzmannParallel(
                self.system,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,            # For PBC this is IdentityTransform()
                platform_name=self.platform_name,
                n_threads=n_threads,
            )
        else:
            # Need to define sim, since the non-parallel version does not take a system as input (parallel builds the
            #  sim from system in exactly this way).
            sim = app.Simulation(
                self.system.topology,
                self.system.system,
                integrator(temperature * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond),
                mm.Platform.getPlatformByName(self.platform_name),
                self.platform_properties,
            )

            print("OpenMM platform:", sim.context.getPlatform().getName())
            
            self.p = TransformedBoltzmann(
                sim.context,
                temperature,
                energy_cut=energy_cut,
                energy_max=energy_max,
                transform=self.coordinate_transform,            # For PBC this is IdentityTransform()
            )

    # @staticmethod
    def load_target_data(self, data_path: pathlib.Path, dim: int):
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
        if self.boundary_condition == "pbc":
            target_data = preprocess_frame_batch(
                target_data.reshape(-1, self.cartesian_dim).to(self.device),
                L=self.box_length_nm,
                n_solute=3,
                n_waters=self.num_solvent_molecules,
                anchor_idx=0, 
                debug=False,
            ).cpu().reshape_as(target_data)
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
            n_eval: int = 500,
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

        # Load MD data for evaluation
        if self.eval_mode == "val":  # TODO: batch this?
            # Note that x (Cartesian) data is NOT centered!
            #  To center X data, run self.coordinate_transform.setup_coordinate_system(X_DATA)[0], or run the inverse
            #   coordinate transform on the I (internal coordinate) data.
            target_data_x = self.val_data_x.reshape(-1, self.cartesian_dim).to(self.device)
            target_data_i = self.val_data_i.reshape(-1, self.internal_dim).to(self.device)
            target_logdet_xi = self.val_logdet_xi.to(self.device)
        elif self.eval_mode == "test":
            target_data_x = self.test_data_x.reshape(-1, self.cartesian_dim).to(self.device)
            target_data_i = self.test_data_i.reshape(-1, self.internal_dim).to(self.device)
            target_logdet_xi = self.test_logdet_xi.to(self.device)
        else:
            raise ValueError("Invalid eval_mode. Must be 'val' or 'test'.")

        # Subsample on CPU
        N = target_data_i.shape[0]
        n_use = min(n_eval, N)

        # deterministic or random
        idx = torch.randperm(N)[:n_use]

        target_data_i_eval = target_data_i[idx]
        target_logdet_xi_eval = target_logdet_xi[idx]


        summary_dict = {}

        # Log_prob of flow given, so use this for evaluating the log probability of MD data.
        if log_q_fn is not None:
            with torch.no_grad():
                s = 0.0
                n = 0
                for start in range(0, target_data_i_eval.shape[0], batch_size):
                    end = start + batch_size
                    v = log_q_fn(target_data_i_eval[start:end]) + target_logdet_xi_eval[start:end]
                    s += v.sum().item()
                    n += v.numel()
    
                summary_dict["flow_test_log_prob"] = s / n
                summary_dict["flow_test_log_prob_per_dim"] = (s / n) / self.internal_dim
            # Log_prob of target data under flow
            # with torch.no_grad():
            #     # log_q_fn is the log_prob function of the flow.
            #     log_q_test = log_q_fn(target_data_i) + target_logdet_xi
            # test_mean_log_prob = torch.mean(log_q_test)
            # summary_dict.update({"flow_test_log_prob": test_mean_log_prob.cpu().item()})

        # Use flow samples for computing marginal KL estimates.
        if samples is None:  # No samples provided, so generate using flow.
            assert flow, (
                "Flow model must be provided for generating evaluation samples if none are provided."
            )
            # num_flow_samples = len(target_data_i)  # Use same number of Flow and MD samples.
            num_flow_samples = target_data_i_eval.shape[0]  # match n_eval
            with torch.no_grad():
                flow_samples, _ = flow.sample_and_log_prob((num_flow_samples,))
        else:  # Samples provided (can be Flow or Flow+AIS samples).
            flow_samples = samples

        # Estimate the KL per dimension using the provided samples; we can normalise this using the samples
        # themselves. Here we're essentially estimating the KL as the average of marginal KLs (per dimension).
        # TODO: Possibly batch this if necessary?
        nbins = 200
        hist_range = [-5, 5]

        target_data_kl = target_data_i_eval.detach().cpu().numpy()
        flow_samples_kl = flow_samples.detach().cpu().numpy()

        # target_data_kl = target_data_i.cpu().clone().numpy()
        # flow_samples_kl = flow_samples.cpu().clone().numpy()
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
    
    def find_bad_frames_bisect(
        self,
        X: torch.Tensor,
        name: str,
        chunk_size: int = 8192,
        msg_substring: str = "Found rotation around axis with x=0 outside of coordinate setup.",
        max_bad: int | None = None,
        ):

        """
        Find indices of frames in X that cause coordinate_transform.inverse() to raise a specific ValueError.
        Uses chunk testing + bisection to avoid O(N) per-frame calls.
        """
        assert X.ndim == 2, f"{name}: expected (n_frames, dim), got {tuple(X.shape)}"
        if X.shape[1] != self.cartesian_dim:
            raise ValueError(
                f"{name}: dim mismatch: X.shape[1]={X.shape[1]} vs cartesian_dim={self.cartesian_dim}"
            )

        X = X.to(self.device)
        n = X.shape[0]
        bad = []

        def fails(i0: int, i1: int) -> bool:
            try:
                with torch.no_grad():
                    _ = self.coordinate_transform.inverse(X[i0:i1])
                return False
            except ValueError as e:
                if msg_substring in str(e):
                    return True
                raise  # other errors should still surface

        print("Finding bad frames via bisection...")
        # 1) coarse scan
        failing_ranges = []
        for i0 in range(0, n, chunk_size):
            i1 = min(n, i0 + chunk_size)
            if fails(i0, i1):
                failing_ranges.append((i0, i1))

        # 2) bisect failing ranges to isolate failing frames
        stack = failing_ranges[:]
        while stack:
            i0, i1 = stack.pop()
            if i1 - i0 == 1:
                bad.append(i0)
                if max_bad is not None and len(bad) >= max_bad:
                    break
                continue
            mid = (i0 + i1) // 2
            if fails(i0, mid):
                stack.append((i0, mid))
            if fails(mid, i1):
                stack.append((mid, i1))

        bad = sorted(set(bad))
        print(f"[{name}] total={n} bad={len(bad)} rate={len(bad)/max(1,n):.3e}")
        if bad:
            print(f"[{name}] first bad indices: {bad[:20]}")
        return bad
    