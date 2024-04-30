
# # Relax the Initial Geometries
# Produce a relaxed structure based on one of the initial geometries

from cascade.calculator import make_calculator
from ase.filters import UnitCellFilter
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from ase.db import connect
from ase.io import read
from ase import units
from pathlib import Path
import shutil


# Configuration

method: str = 'blyp'
initial_geometry = 'initial-geometries/packmol-CH4-in-H2O=32-seed=0.vasp'


# Derived

name = f'{Path(initial_geometry).name[:-5]}-{method}'
run_dir = Path('relax') / name

run_dir.mkdir(exist_ok=True, parents=True)


# ## Perform the Relaxation
# Either from the initial geometry or the latest relaxation step from the trajectory


relax_traj = run_dir / 'relax.traj'
if relax_traj.is_file() and relax_traj.stat().st_size > 75:
    atoms = read(str(relax_traj), index=-1)
    print('Loaded last structure')
else:
    atoms = read(initial_geometry)


# Set the calculator
if Path('cp2k-run').exists():
    Path('cp2k-run/cp2k.out').write_text('')
atoms.calc = make_calculator(method, 
                             directory='cp2k-run',
                            )



init_vol = atoms.get_volume()
ecf = UnitCellFilter(atoms, hydrostatic_strain=True)
with Trajectory(str(relax_traj), mode='a') as traj:
    dyn = LBFGS(ecf, 
                logfile=str(run_dir / 'relax.log'),
                trajectory=traj)
    dyn.run(fmax=0.1)
final_vol = atoms.get_volume()



Path('final-geometries').mkdir(exist_ok=True)
atoms.write(f'final-geometries/{name}.vasp')

