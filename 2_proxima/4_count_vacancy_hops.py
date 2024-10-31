"""A *script* to find the vacancy hops in our diffusion runs
saves the indices of the new states and the transition states to disk
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import ase
from ase.io import read

parser = ArgumentParser()
parser.add_argument('path', 
                    type=str, 
                    help='md run directory containing ASE traj file')
args = parser.parse_args()

path = Path(args.path)

traj = read(path/'md.traj', index=':')


# the 'correct' neighbor distribution (i.e., non-transition)
# for 63 silicon atoms + 1 vacancy
correct_counts = pd.Series({4:59, 3: 4})

def get_n_neighbors(atoms: ase.Atoms, r_cut: float = 3) -> np.ndarray[int]:
    """Get the number of neighbors within r_cut angstroms of each atom"""
    D = atoms.get_all_distances(mic=True)
    nn = (D < r_cut).sum(1) - 1 # subtract 1 for self distances
    return nn

# loop over the trajectory and identify hops, transition states
# todo mt-2024.10.30: this can probably be sped up a lot with vectorizaiton 
# one could identify all of the unique nn configurations (fast/parallelizable)
# check which are correct, and then use a row-wise diff and some indexing

atoms = traj[0]
nn = get_n_neighbors(atoms)
# track whether a timestep has a 'wrong' or new configuration of neighbors
steps_wrong = np.zeros(len(traj))
steps_new = np.zeros(len(traj))
nn_all = [nn]
for t, atoms in enumerate(traj[1:]):
    print(f'Checking timestep {t}/{len(traj)}', end='\r')
    nn_new = get_n_neighbors(atoms)
    counts = pd.Series(nn_new).value_counts()

    # check whether we have the correct neighbor count distribution
    # handling the case where series comparison is invalid
    try: 
        is_correct = (counts == correct_counts).all()
    except: 
        is_correct = False

    # if we do have the correct neighbor count distribution
    # check if the atoms with these counts have changed
    if is_correct: 
        is_new = not (nn == nn_new).all()
        if is_new: 
            # count this as a new state
            steps_new[t] = 1
            # update the reference n_neighbor distribution
            nn = nn_new
    else: 
        steps_wrong[t] = 1

hop_indices = np.where(steps_new == 1)[0]
durations = np.diff(hop_indices)
durations = hop_indices[0] + list(durations)

np.savez(path/'hops.npz', 
    hop_indices=hop_indices,
    durations=durations,
    steps_new=steps_new,
    steps_wrong=steps_wrong,
)

