import json
import os
import pathlib
import warnings
from typing import Optional, Dict, Literal, Callable

import h5py
import numpy as np
import torch
from torch import nn, Tensor

import openmm as mm
from openmm import app, unit
from openmmtools.testsystems import TestSystem

from fab.transforms.transform_LJ import SolventOnlyTransform, FixedSoluteUnitTorusTransform

from fab.target_distributions.base import TargetDistribution
from fab.target_distributions.boltzmann import (
    TransformedBoltzmann,
    TransformedBoltzmannParallel,
)

import numpy as np


def build_fcc_positions(box_length_nm: float, n_cells: int) -> np.ndarray:
    """
    Build FCC lattice positions in a cubic box.

    Args:
        box_length_nm: side length of the cubic box
        n_cells: number of FCC unit cells along each box axis

    Returns:
        positions: (4 * n_cells**3, 3) array in nm, inside [0, L)
    """
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")

    a = box_length_nm / float(n_cells)

    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    positions = []
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                cell_origin = np.array([i, j, k], dtype=np.float64)
                for b in basis:
                    positions.append((cell_origin + b) * a)

    positions = np.array(positions, dtype=np.float64)

    # wrap into [0, L)
    positions = positions % box_length_nm
    return positions

def build_centered_fcc_positions(box_length_nm: float, n_cells: int) -> np.ndarray:
    """
    Build FCC lattice positions in a cubic box and shift them so that
    one FCC site lies exactly at the box center [L/2, L/2, L/2].

    Returns:
        positions: (4 * n_cells**3, 3) array in nm, inside [0, L)
    """
    positions = build_fcc_positions(box_length_nm, n_cells)

    L = float(box_length_nm)
    center = np.array([L / 2.0, L / 2.0, L / 2.0], dtype=np.float64)

    # Find the FCC site closest to the box center
    d2 = np.sum((positions - center[None, :]) ** 2, axis=1)
    idx_center = np.argmin(d2)

    # Shift lattice so that this site is exactly at the center
    shift = center - positions[idx_center]
    positions = (positions + shift) % L

    return positions

def make_grid_positions(box_length_nm: float, spacing_nm: float) -> np.ndarray:
    coords_1d = np.arange(spacing_nm / 2.0, box_length_nm, spacing_nm, dtype=np.float64)
    grid = np.array(np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij"))
    return grid.reshape(3, -1).T


def build_grid_initialized_positions(
    box_length_nm: float,
    spacing_nm: float,
    n_solvent: int,
    solute_positions_nm: np.ndarray | list,
    seed: int = 0,
):
    grid_points = make_grid_positions(box_length_nm, spacing_nm)
    solute_positions_nm = np.asarray(solute_positions_nm, dtype=np.float64)

    used = np.zeros(len(grid_points), dtype=bool)
    final_solute_positions = []

    for xyz in solute_positions_nm:
        d2 = np.sum((grid_points - xyz[None, :]) ** 2, axis=1)
        order = np.argsort(d2)
        idx = next(i for i in order if not used[i])
        used[idx] = True
        final_solute_positions.append(grid_points[idx])

    available = grid_points[~used]
    if len(available) < n_solvent:
        raise ValueError("Not enough remaining grid points for solvent.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(available))
    solvent_positions = available[perm[:n_solvent]]

    return np.array(final_solute_positions), np.array(solvent_positions)


class LJParticlesSys(TestSystem):
    """
    LJ solvent + optional fixed LJ solutes in a cubic periodic box.
    """

    def __init__(
        self,
        n_solvent: int,
        box_length_nm: float,
        solvent_sigma_nm: float,
        solvent_epsilon_kjmol: float,
        solvent_mass_amu: float,
        solute_positions_nm=None,
        solute_sigma_nm=None,
        solute_epsilon_kjmol=None,
        solute_mass_amu: float = 0.0,
        switch_nm: float = 0.5,
        cutoff_nm: float = 0.8,
        constrain_solutes: bool = True,
        solute_solute_interaction: bool = False,
        seed: int = 0,
        grid_spacing_nm: float = 0.36,
        solid: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if cutoff_nm > 0.5 * box_length_nm:
            raise ValueError("Require cutoff_nm <= box_length_nm / 2 for cubic PBC.")

        self.n_solvent = int(n_solvent)
        self.box_length_nm = float(box_length_nm)
        self.solvent_sigma_nm = float(solvent_sigma_nm)
        self.solvent_epsilon_kjmol = float(solvent_epsilon_kjmol)
        self.solvent_mass_amu = float(solvent_mass_amu)
        self.solute_mass_amu = float(solute_mass_amu)
        self.switch_nm = float(switch_nm)
        self.cutoff_nm = float(cutoff_nm)
        self.grid_spacing_nm = float(grid_spacing_nm)
        self.solid = solid

        if solute_positions_nm is None:
            solute_positions_nm = []
        self.solute_positions_nm = np.asarray(solute_positions_nm, dtype=np.float64)
        self.n_solute = len(self.solute_positions_nm)

        if self.n_solute > 0:
            if solute_sigma_nm is None or solute_epsilon_kjmol is None:
                raise ValueError("Provide solute_sigma_nm and solute_epsilon_kjmol for solutes.")

            if np.isscalar(solute_sigma_nm):
                solute_sigma_nm = [solute_sigma_nm] * self.n_solute
            if np.isscalar(solute_epsilon_kjmol):
                solute_epsilon_kjmol = [solute_epsilon_kjmol] * self.n_solute

            self.solute_sigma_nm = np.asarray(solute_sigma_nm, dtype=np.float64)
            self.solute_epsilon_kjmol = np.asarray(solute_epsilon_kjmol, dtype=np.float64)
        else:
            self.solute_sigma_nm = np.zeros(0, dtype=np.float64)
            self.solute_epsilon_kjmol = np.zeros(0, dtype=np.float64)

        system = mm.System()

        L = self.box_length_nm
        a = mm.Vec3(L, 0.0, 0.0)
        b = mm.Vec3(0.0, L, 0.0)
        c = mm.Vec3(0.0, 0.0, L)
        system.setDefaultPeriodicBoxVectors(
            a * unit.nanometer,
            b * unit.nanometer,
            c * unit.nanometer,
        )

        for _ in range(self.n_solute):
            if constrain_solutes:
                system.addParticle(0.0 * unit.amu)
            else:
                system.addParticle(self.solute_mass_amu * unit.amu)

        for _ in range(self.n_solvent):
            system.addParticle(self.solvent_mass_amu * unit.amu)

        lj = mm.CustomNonbondedForce(
            "4*sqrt(epsilon1*epsilon2)*((sigma/r)^12 - (sigma/r)^6) * S;"
            "sigma = 0.5*(sigma1+sigma2);"
            "S = step(rswitch-r) + step(r-rswitch)*step(rcut-r)*("
            "1 - 10*((r-rswitch)/(rcut-rswitch))^3 + "
            "15*((r-rswitch)/(rcut-rswitch))^4 - "
            "6*((r-rswitch)/(rcut-rswitch))^5"
            ");"
        )
        lj.addPerParticleParameter("sigma")
        lj.addPerParticleParameter("epsilon")
        lj.addGlobalParameter("rswitch", self.switch_nm)
        lj.addGlobalParameter("rcut", self.cutoff_nm)
        lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        lj.setCutoffDistance(self.cutoff_nm * unit.nanometer)

        for i in range(self.n_solute):
            lj.addParticle([self.solute_sigma_nm[i], self.solute_epsilon_kjmol[i]])
        for _ in range(self.n_solvent):
            lj.addParticle([self.solvent_sigma_nm, self.solvent_epsilon_kjmol])

        if self.n_solute > 1 and not solute_solute_interaction:
            for i in range(self.n_solute):
                for j in range(i + 1, self.n_solute):
                    lj.addExclusion(i, j)

        system.addForce(lj)
        self.system = system

        topology = app.Topology()
        chain = topology.addChain()

        if self.n_solute > 0:
            solute_res = topology.addResidue("SOLU", chain)
            for i in range(self.n_solute):
                topology.addAtom(f"S{i}", app.Element.getByAtomicNumber(18), solute_res)

        solvent_res = topology.addResidue("SOLV", chain)
        for i in range(self.n_solvent):
            topology.addAtom(f"LJ{i}", app.Element.getByAtomicNumber(10), solvent_res)

        topology.setPeriodicBoxVectors((
            a * unit.nanometer,
            b * unit.nanometer,
            c * unit.nanometer,
        ))
        self.topology = topology

        if self.solid is True:
            n_total = self.n_solute + self.n_solvent

            n_cells = round((n_total / 4) ** (1 / 3))
            if 4 * (n_cells ** 3) != n_total:
                raise ValueError(
                    f"FCC requires total particle count = 4 * n_cells^3, got total={n_total} "
                    f"(n_solute={self.n_solute}, n_solvent={self.n_solvent})"
                )

            fcc = build_centered_fcc_positions(
                box_length_nm=self.box_length_nm,
                n_cells=n_cells,
            )

            L = self.box_length_nm
            center = np.array([L / 2.0, L / 2.0, L / 2.0], dtype=np.float64)

            if self.n_solute > 0:
                # Find the FCC site exactly at the box center
                d2 = np.sum((fcc - center[None, :]) ** 2, axis=1)
                solute_idx = np.argmin(d2)

                if not np.allclose(fcc[solute_idx], center, atol=1e-10):
                    raise RuntimeError("Failed to place an FCC site at the box center.")

                final_solutes = np.array([fcc[solute_idx]], dtype=np.float64)

                mask = np.ones(len(fcc), dtype=bool)
                mask[solute_idx] = False
                solvent_xyz = fcc[mask]

                if solvent_xyz.shape[0] != self.n_solvent:
                    raise RuntimeError(
                        f"Expected {self.n_solvent} solvent sites, got {solvent_xyz.shape[0]}"
                    )
            else:
                final_solutes = np.zeros((0, 3), dtype=np.float64)
                solvent_xyz = fcc[:self.n_solvent].copy()

        else:
            final_solutes, solvent_xyz = build_grid_initialized_positions(
                box_length_nm=self.box_length_nm,
                spacing_nm=self.grid_spacing_nm,
                n_solvent=self.n_solvent,
                solute_positions_nm=self.solute_positions_nm,
                seed=seed,
            )

        self.solute_positions_nm = final_solutes

        positions = []
        for xyz in final_solutes:
            positions.append(mm.Vec3(*xyz) * unit.nanometer)
        for xyz in solvent_xyz:
            positions.append(mm.Vec3(*xyz) * unit.nanometer)

        self.positions = positions



class LJParticles(nn.Module, TargetDistribution):
    """
    LJ target using Cartesian coordinates directly.
    """

    def __init__(
        self,
        dim: int,
        n_solvent: int,
        temperature: float,
        energy_cut: float = 1.0e8,
        energy_max: float = 1.0e20,
        n_threads: int = 1,
        train_samples_path: Optional[str] = None,
        val_samples_path: Optional[str] = None,
        test_samples_path: Optional[str] = None,
        eval_mode: Literal["val", "test"] = "val",
        device: str = "cpu",
        logger=None,
        save_dir: Optional[str] = None,
        box_length_nm: float = 1.69,
        solvent_sigma_nm: float = 0.34,
        solvent_epsilon_kjmol: float = 0.996,
        solvent_mass_amu: float = 39.9,
        solute_positions_nm=None,
        solute_sigma_nm=None,
        solute_epsilon_kjmol=None,
        solute_mass_amu: float = 0.0,
        switch_nm: float = 0.5,
        cutoff_nm: float = 0.8,
        constrain_solutes: bool = True,
        solute_solute_interaction: bool = False,
        seed: int = 0,
        grid_spacing_nm: float = 0.36,
        platform_name: str = "CUDA",
        platform_properties: Optional[Dict[str, str]] = None,
        plot_MD_energies: bool = False,
        plot_marginal_hists: bool = False,
        transform_version: str = "v1",
        curriculum_type: Optional[str] = None,
        curriculum_lambda: float = 1.0,
        curriculum_soft_energy_cut: float = 1.0,
        solid: bool = False,
    ):
        nn.Module.__init__(self)
        TargetDistribution.__init__(self)

        self.n_solvent = int(n_solvent)
        self.temperature = float(temperature)
        self.energy_cut = float(energy_cut)
        self.energy_max = float(energy_max)
        self.n_threads = int(n_threads)
        self.device = device
        self.eval_mode = eval_mode
        self.logger = logger
        self.save_dir = save_dir
        self.plot_MD_energies = plot_MD_energies
        self.plot_marginal_hists = plot_marginal_hists
        self.transform_version = transform_version

        self.box_length_nm = float(box_length_nm)
        self.platform_name = platform_name
        self.platform_properties = platform_properties
        self.curriculum_type = curriculum_type
        self.curriculum_lambda = curriculum_lambda
        self.curriculum_soft_energy_cut = curriculum_soft_energy_cut

        self.metric_dir = None
        if self.save_dir is not None:
            self.metric_dir = os.path.join(self.save_dir, "metrics")
            os.makedirs(self.metric_dir, exist_ok=True)

        self.system = LJParticlesSys(
            n_solvent=n_solvent,
            box_length_nm=box_length_nm,
            solvent_sigma_nm=solvent_sigma_nm,
            solvent_epsilon_kjmol=solvent_epsilon_kjmol,
            solvent_mass_amu=solvent_mass_amu,
            solute_positions_nm=solute_positions_nm,
            solute_sigma_nm=solute_sigma_nm,
            solute_epsilon_kjmol=solute_epsilon_kjmol,
            solute_mass_amu=solute_mass_amu,
            switch_nm=switch_nm,
            cutoff_nm=cutoff_nm,
            constrain_solutes=constrain_solutes,
            solute_solute_interaction=solute_solute_interaction,
            seed=seed,
            grid_spacing_nm=grid_spacing_nm,
            solid=solid,
        )

        self.cartesian_dim = int(dim)
        expected_dim = 3 * self.system.topology.getNumAtoms()
        if self.cartesian_dim != expected_dim:
            raise ValueError(f"dim={dim}, expected {expected_dim}")

        n_atoms = self.system.topology.getNumAtoms()
        self.cartesian_dim = 3 * n_atoms

        self.n_solute = self.system.n_solute
        self.n_solvent = self.system.n_solvent

        if self.transform_version == "v1":
            self.coordinate_transform = SolventOnlyTransform(
                solute_positions_nm=self.system.solute_positions_nm,
                n_solute=self.n_solute,
                n_solvent=self.n_solvent,
            )
        elif self.transform_version == "v4":
            self.coordinate_transform = FixedSoluteUnitTorusTransform(
                solute_positions_nm=self.system.solute_positions_nm,
                n_solute=self.n_solute,
                n_solvent=self.n_solvent,
                box_length_nm=self.box_length_nm,
            )
        else:
            raise ValueError(f"Unknown transform_version={self.transform_version}")
        


        self.internal_dim = self.coordinate_transform.internal_dim

        self.train_data_x = self.load_target_data(train_samples_path, self.cartesian_dim) if train_samples_path else None
        self.val_data_x = self.load_target_data(val_samples_path, self.cartesian_dim) if val_samples_path else None
        self.test_data_x = self.load_target_data(test_samples_path, self.cartesian_dim) if test_samples_path else None

        self.train_data_i, self.train_logdet_xi = self.coordinate_transform.inverse(self.train_data_x.reshape(-1, self.cartesian_dim)) if self.train_data_x is not None else (None, None)   
        self.val_data_i, self.val_logdet_xi = self.coordinate_transform.inverse(self.val_data_x.reshape(-1, self.cartesian_dim)) if self.val_data_x is not None else (None, None) 
        self.test_data_i, self.test_logdet_xi = self.coordinate_transform.inverse(self.test_data_x.reshape(-1, self.cartesian_dim)) if self.test_data_x is not None else (None, None) 

        integrator = mm.LangevinMiddleIntegrator(
            self.temperature * unit.kelvin,
            1.0 / unit.picosecond,
            1.0 * unit.femtosecond,
        )
        sim = app.Simulation(
            self.system.topology,
            self.system.system,
            integrator,
            mm.Platform.getPlatformByName(self.platform_name),
            self.platform_properties,
        )
        sim.context.setPositions(self.system.positions)

        if n_threads > 1:
            self.p = TransformedBoltzmannParallel(
                self.system,
                self.temperature,
                energy_cut=self.energy_cut,
                energy_max=self.energy_max,
                transform=self.coordinate_transform,
                platform_name=self.platform_name,
                n_threads=n_threads,
                
            )
        else:
            force_groups = None
            self.p = TransformedBoltzmann(
                sim.context,
                self.temperature,
                energy_cut=self.energy_cut,
                energy_max=self.energy_max,
                transform=self.coordinate_transform,
                force_groups=force_groups,
                curriculum_type=self.curriculum_type,
                curriculum_lambda=self.curriculum_lambda,
                curriculum_soft_energy_cut=self.curriculum_soft_energy_cut
            )

    def load_target_data(self, data_path: pathlib.Path | str, dim: int):
        data_path = pathlib.Path(data_path)

        if data_path.suffix == ".h5":
            with h5py.File(str(data_path), "r") as f:
                coords = torch.from_numpy(f["coordinates"][()]).reshape(-1, dim)
        elif data_path.suffix == ".pt":
            coords = torch.load(str(data_path))
            if coords.ndim != 2:
                raise ValueError("PT data must have shape (num_frames, dim)")
        elif data_path.suffix == ".pdb":
            warnings.warn("Loading .pdb is slow. Prefer .h5 or .pt.")
            pdb = app.PDBFile(str(data_path))
            frames = []
            for i in range(pdb.getNumFrames()):
                frame = torch.from_numpy(np.array(pdb.getPositions(asNumpy=True, frame=i))).reshape(1, dim)
                frames.append(frame)
            coords = torch.cat(frames, dim=0)
        else:
            raise ValueError(f"Unsupported suffix: {data_path.suffix}")

        if coords.shape[1] != dim:
            raise ValueError(f"Loaded dim {coords.shape[1]} != expected {dim}")
        return coords

    def log_prob(self, i: Tensor):
        return self.p.log_prob(i)

    def log_prob_and_jac(self, i: Tensor):
        return self.p.log_prob_and_jac(i)

    def log_prob_x(self, x: Tensor):
        return self.p.log_prob_x(x)

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
        if self.eval_mode == "val":
            target_data_i = self.val_data_i
            target_logdet_xi = self.val_logdet_xi
        elif self.eval_mode == "test":
            target_data_i = self.test_data_i
            target_logdet_xi = self.test_logdet_xi
        else:
            raise ValueError("eval_mode must be 'val' or 'test'")

        if target_data_i is None:
            return {}

        target_data_i = target_data_i.to(self.device)
        target_logdet_xi = target_logdet_xi.to(self.device)

        N = target_data_i.shape[0]
        n_use = min(n_eval, N)
        idx = torch.randperm(N, device=target_data_i.device)[:n_use]
        target_data_i_eval = target_data_i[idx]
        target_logdet_xi_eval = target_logdet_xi[idx]

        summary_dict = {}

        if log_q_fn is not None:
            vals = []
            with torch.no_grad():
                for start in range(0, target_data_i_eval.shape[0], batch_size):
                    end = start + batch_size
                    v = log_q_fn(target_data_i_eval[start:end]) + target_logdet_xi_eval[start:end]
                    vals.append(v)
                vals = torch.cat(vals, dim=0)
            summary_dict["flow_test_log_prob"] = vals.mean().item()
            summary_dict["flow_test_log_prob_per_dim"] = vals.mean().item() / self.internal_dim

        if samples is None:
            if flow is None:
                raise ValueError("Need either samples or a flow.")
            with torch.no_grad():
                flow_samples, flow_log_q = flow.sample_and_log_prob((n_use,))
                log_p = self.log_prob(flow_samples)
                log_w = log_p - flow_log_q
                w = torch.exp(log_w - torch.max(log_w))
                summary_dict["eval_ess_flow"] = ((w.sum() ** 2) / (w.pow(2).sum())).item()
        else:
            flow_samples = samples

        nbins = 200
        hist_range = [-5, 5]
        eps = 1e-10

        target_np = target_data_i_eval.detach().cpu().numpy()
        flow_np = flow_samples.detach().cpu().numpy()

        hists_test = np.zeros((nbins, self.internal_dim))
        hists_flow = np.zeros((nbins, self.internal_dim))
        for d in range(self.internal_dim):
            h_test, _ = np.histogram(target_np[:, d], bins=nbins, range=hist_range, density=True)
            h_flow, _ = np.histogram(flow_np[:, d], bins=nbins, range=hist_range, density=True)
            hists_test[:, d] = h_test
            hists_flow[:, d] = h_flow

        forward_kl = np.sum(hists_test * (np.log(hists_test + eps) - np.log(hists_flow + eps)), axis=0)
        reverse_kl = np.sum(hists_flow * (np.log(hists_flow + eps) - np.log(hists_test + eps)), axis=0)
        dx = (hist_range[1] - hist_range[0]) / nbins
        summary_dict["mean_forward_kl_marginals"] = float((forward_kl * dx).mean())
        summary_dict["mean_reverse_kl_marginals"] = float((reverse_kl * dx).mean())

        if self.metric_dir is not None and iteration is not None:
            with open(os.path.join(self.metric_dir, f"metrics_{iteration}.json"), "w") as f:
                json.dump(summary_dict, f, indent=2)

        return summary_dict