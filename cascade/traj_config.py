"""Per-trajectory launch configuration, shared by run_cascade_academy.py and
run_reference_dynamics.py.

Lives in the cascade package (rather than in either launch script) because Parsl
workers deserialize task arguments by importing the module a type was defined in -
a loose script isn't importable there, but cascade is a properly installed package.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms, units
import ase.md.md
from ase.md.verlet import VelocityVerlet
from ase.md.npt import NPT
from ase.md.nose_hoover_chain import MTKNPT


@dataclass
class InitialTrajConfig:
    """Initial configuration for a single trajectory"""
    path: str
    """Path to the initial structure, readable by ase.io.read"""
    temperature_K: float | None = None
    """If set, initialize velocities via a Maxwell-Boltzmann distribution at this temperature"""
    dyn_cls: str = 'velocity-verlet'
    """Dynamics integrator to use (see get_dynamics_cls)"""
    dt_fs: float = 1.0
    """Timestep in femtoseconds"""
    dyn_kws: dict = field(default_factory=dict)
    """Additional keyword arguments passed to the dynamics constructor (besides timestep)"""
    run_kws: dict = field(default_factory=dict)
    """Keyword arguments passed to the dynamics run method"""


def load_initial_configs(path: str) -> list[InitialTrajConfig]:
    data = json.loads(pathlib.Path(path).read_text())
    return [InitialTrajConfig(**entry) for entry in data]


def get_dynamics_cls(cls_name: str) -> type[ase.md.md.MolecularDynamics]:
    if cls_name == 'velocity-verlet':
        return VelocityVerlet
    elif cls_name == 'npt':
        return NPT
    elif cls_name == 'mtknpt':
        return MTKNPT
    else:
        raise ValueError(f'Unknown dynamics class: {cls_name}')


@dataclass
class NPTConfig:
    """Friendly-unit settings for ase.md.npt.NPT, converted in to_ase_kwargs()"""
    temperature_K: float
    ttime_fs: float
    externalstress_GPa: float
    pfactor_time_fs: float
    pfactor_pressure_GPa: float
    mask: list[int] | None = None

    def to_ase_kwargs(self) -> dict:
        return dict(
            temperature_K=self.temperature_K,
            ttime=self.ttime_fs * units.fs,
            externalstress=self.externalstress_GPa * units.GPa,
            pfactor=(self.pfactor_time_fs * units.fs) ** 2 * self.pfactor_pressure_GPa * units.GPa,
            mask=self.mask,
        )


@dataclass
class MTKNPTConfig:
    """Friendly-unit settings for ase.md.nose_hoover_chain.MTKNPT, converted in to_ase_kwargs()"""
    temperature_K: float
    pressure_GPa: float
    tdamp_fs: float
    pdamp_fs: float
    tchain: int = 3
    pchain: int = 3
    tloop: int = 1
    ploop: int = 1
    loginterval: int = 1

    def to_ase_kwargs(self) -> dict:
        return dict(
            temperature_K=self.temperature_K,
            pressure_au=self.pressure_GPa * units.GPa,
            tdamp=self.tdamp_fs * units.fs,
            pdamp=self.pdamp_fs * units.fs,
            tchain=self.tchain,
            pchain=self.pchain,
            tloop=self.tloop,
            ploop=self.ploop,
            loginterval=self.loginterval,
        )


def resolve_dyn_kws(cfg: InitialTrajConfig) -> dict:
    """Build the real ASE dynamics-constructor kwargs for a trajectory config"""
    if cfg.dyn_cls == 'npt':
        return {'timestep': cfg.dt_fs * units.fs, **NPTConfig(**cfg.dyn_kws).to_ase_kwargs()}
    elif cfg.dyn_cls == 'mtknpt':
        return {'timestep': cfg.dt_fs * units.fs, **MTKNPTConfig(**cfg.dyn_kws).to_ase_kwargs()}
    return {'timestep': cfg.dt_fs * units.fs, **cfg.dyn_kws}


def extract_dyn_state(dyn: ase.md.md.MolecularDynamics) -> dict | None:
    """Snapshot the extended-system (barostat/thermostat) state of an NPT-family
    integrator so it can be restored on the next chunk.

    NPT-family integrators keep their barostat/thermostat degrees of freedom as
    attributes on the Dynamics instance rather than on Atoms, and ASE zeroes them
    unconditionally in __init__. Without this, reconstructing a fresh integrator
    for every chunk (necessary since chunks run as separate executor tasks) makes
    the barostat/thermostat restart from rest at every chunk boundary.

    Returns None for dynamics classes with no such state (e.g. VelocityVerlet).
    """
    if isinstance(dyn, MTKNPT):
        return {
            'p_g': dyn._p_g.copy(),
            'thermostat_eta': dyn._thermostat._eta.copy(),
            'thermostat_p_eta': dyn._thermostat._p_eta.copy(),
            'barostat_xi': dyn._barostat._xi.copy(),
            'barostat_p_xi': dyn._barostat._p_xi.copy(),
        }
    elif isinstance(dyn, NPT):
        return dyn.get_data()
    return None


def restore_dyn_state(dyn: ase.md.md.MolecularDynamics, state: dict | None) -> None:
    """Inverse of extract_dyn_state: re-inject a prior chunk's extended-system
    state into a freshly constructed integrator, in place.

    For classic NPT this mirrors ASE's own read_from_trajectory restore pattern
    (construct normally, setattr the get_data() fields, leave `initialized`
    unset so the next run() call's own initialize() bootstrap derives h_past/
    q_past consistently from the restored eta/zeta).
    """
    if state is None:
        return
    if isinstance(dyn, MTKNPT):
        dyn._p_g = state['p_g']
        dyn._thermostat._eta = state['thermostat_eta']
        dyn._thermostat._p_eta = state['thermostat_p_eta']
        dyn._barostat._xi = state['barostat_xi']
        dyn._barostat._p_xi = state['barostat_p_xi']
    elif isinstance(dyn, NPT):
        for k, v in state.items():
            setattr(dyn, k, v)


def _upper_triangular_cell(atoms: Atoms) -> Atoms:
    """Rigidly rotate a structure's cell + positions so the cell matrix becomes upper
    triangular, preserving all lengths, angles, and volume.

    ase.md.npt.NPT requires this cell form (an intrinsic requirement of the Parrinello-Rahman
    equations of motion it implements, not an ASE-specific restriction), and checks it with
    exact equality (cell[1,0] == cell[2,0] == cell[2,1] == 0.0), so the tiny floating-point
    residue from the rotation is zeroed out explicitly afterward.
    """
    atoms = atoms.copy()
    cell = np.array(atoms.get_cell())

    # RQ decomposition (cell = R @ Q, R upper triangular, Q orthogonal) via QR of a flipped
    # matrix - numpy has no direct RQ routine. Rotating by Q.T gives cell @ Q.T = R.
    q, r = np.linalg.qr(np.flipud(cell).T)
    r = np.flipud(r.T)
    r = np.fliplr(r)
    q = np.flipud(q.T)
    if np.linalg.det(q) < 0:
        q[-1, :] *= -1
        r[:, -1] *= -1

    new_cell = cell @ q.T
    new_positions = atoms.get_positions() @ q.T
    new_cell[1, 0] = 0.0
    new_cell[2, 0] = 0.0
    new_cell[2, 1] = 0.0

    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.set_positions(new_positions)
    return atoms


def prepare_atoms_for_dynamics(atoms: Atoms, cfg: InitialTrajConfig) -> Atoms:
    """Apply any structure preprocessing a trajectory's dynamics integrator requires"""
    if cfg.dyn_cls == 'npt':
        atoms = _upper_triangular_cell(atoms)
    return atoms
