"""
"""
from concurrent.futures import as_completed
from tqdm.auto import tqdm


import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.executors import HighThroughputExecutor


@python_app
def generate_structure(solute: str = 'CH4', 
                       solvent: str = 'H2O', 
                       num_solvent: int = 32,
                       density: float = 1.1,
                       distance_tolerance: float = 2.,
                       seed: int = 0,
                       ):
    """Generate an initial solution structure for downstream simulations.

    Structure contains a single solute molecule and many solvents. Wrapped in 
    PBC and witha minimum distance between atoms.
    
    Arguments: 
        solute (str): SMILES string for a solute molecule
        solvent (str): SMILES string for solvent molecule
        num_solvent (int): how many solvent molecules to include
        distance_tolerance (float): (angstroms) structure will be rescaled with respect to this
    """

    import numpy as np
    from ase.build import molecule
    from ase.io import read
    from ase import units
    import tempfile
    import contextlib
    from pathlib import Path
    from subprocess import run

    # Do all of the scratch work in a temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        with contextlib.chdir(tempdir):

            solvent_atoms = molecule(solvent)
            solute_atoms = molecule(solute)

            # # save the solvent and solute in a file
            # filenames = map(lambda fn: str(Path(tempdir) / (fn + '.pdb')), 
            #                 ['solvent', 'solute', 'cell']) #(cell will be used later)
            # solvent_file, solute_file, cell_file = filenames
            solvent_atoms.write('solvent.pdb')
            solute_atoms.write('solute.pdb')

            # Compute the density based on the solvent
            total_mass = solvent_atoms.get_masses().sum() * num_solvent / units.mol # g
            volume = (total_mass / units.mol) / density * (units.kg / 1000)  # cm^3
            volume /= 1e-24  # A^3
            side_length = np.power(volume, 1./3)
            print(f'Building a box with side lengths of {side_length:.2f} A')


            # Create the packmol input file
            middle = side_length / 2
            packmol_inp = f'''
            seed {seed}
            tolerance {distance_tolerance:.1f}
            output cell.pdb
            structure solvent.pdb
            number {num_solvent}
            inside cube 0. 0. 0. {side_length:.2f}
            end structure
            structure solute.pdb
            number 1
            center 
            fixed {middle:.2f} {middle:.2f} {middle:.2f} 0. 0. 0.
            end structure 
            '''
            Path('packmol.inp').write_text(packmol_inp)
            with Path('packmol.inp').open() as fp:
                run('packmol', stdin=fp, capture_output=True)
            cell = read('cell.pdb')
            cell.pbc = True
            cell.cell = [side_length + distance_tolerance / 2] * 3
    Path('initial-geometries').mkdir(exist_ok=True)
    cell.write(f'initial-geometries/packmol-{solute}-in-{solvent}={num_solvent}-seed={seed}.vasp', sort=True)
    return


if __name__ == "__main__": 

    local_htex = Config(
        executors=[
            HighThroughputExecutor(
                label="htex_Local",
                worker_debug=True,
                cores_per_worker=1,
                provider=LocalProvider(
                    channel=LocalChannel(),
                    init_blocks=1,
                    max_blocks=1,
                ),
            )
        ],
        strategy=None,
    )

    parsl.load(local_htex)

    futures = []
    for seed in range(4): 
        futures.append(generate_structure(seed=seed))

    for future in tqdm(as_completed(futures), total=len(futures)):
        pass