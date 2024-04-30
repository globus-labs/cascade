# # Run NPT MD 
# Run a structure to see if it equilibrates

from cascade.calculator import make_calculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.npt import NPT
from ase.io import read
from ase import units
from pathlib import Path
import shutil

method: str = 'blyp'
temperature = 298
initial_geometry = 'final-geometries/packmol-CH4-in-H2O=32-seed=0-blyp.vasp'
steps: int = 512


# Derived
name = f'{Path(initial_geometry).name[:-5]}-npt={temperature}'
run_dir = Path('md') / name

run_dir.mkdir(exist_ok=True, parents=True)
# ## Perform the Dynamics
# Run a set number of MD steps
traj_file = run_dir / 'md.traj'
if traj_file.is_file() and traj_file.stat().st_size > 0:
    traj = read(str(traj_file), slice(None))
    start = len(traj)
    atoms = traj[-1]
    print('Loaded last structure')
else:
    atoms = read(initial_geometry)
    start = 0
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature * 2)


# Relax it
atoms.calc = make_calculator(method, 
                             directory='cp2k-run',
                            )

dyn = NPT(atoms,
          timestep=0.5 * units.fs,
          temperature_K=temperature,
          ttime=100 * units.fs,
          pfactor=0.01,
          externalstress=0,
          logfile=str(run_dir / 'md.log'),
          trajectory=str(traj_file),
          append_trajectory=True)
dyn.run(512 - start)

