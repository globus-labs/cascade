"""A postprocessing script for proxima runs

1. saves trajectories as extended xyz format for visualization in ovito
2. calculates diffusivities and saves numbers/plot to disk

"""
from argparse import ArgumentParser
from pathlib import Path
import os

import scipy as sp
from scipy import stats
import numpy as np
from ase.io import read, extxyz
import matplotlib.pyplot as plt

from cascade.utils import unwrap_trajectory, calculate_sqared_disp


parser = ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--start_index', 
                    type=int, 
                    default=5000,
                    help='Which index')

args = parser.parse_args()

# do all work in the target directory
os.chdir(args.path)
traj = read('md.traj', index=':')

# save the extended xyz
extxyz.write_extxyz('traj.xyz', traj)

E0 = traj[0].get_potential_energy()
E = [a.get_potential_energy() - E0 for a in traj]
V = [a.get_volume() for a in traj]
fig, axs = plt.subplots(2, figsize=(18, 10))
plt.sca(axs[0])
plt.plot(E)
plt.axvline(args.start_index, 
            color='k',
            linestyle='dotted',
            label='burn-in cutoff')
plt.xlabel('$t$ (fs)')
plt.ylabel('$U - U_0$ (eV)')

plt.sca(axs[1])
plt.plot(V)
plt.axvline(args.start_index, 
            color='k',
            linestyle='dotted',
            label='burn-in cutoff')
plt.xlabel('$t$ (fs)')
plt.ylabel('$V (\AA^3)$')
plt.legend()
plt.savefig('energy_volume.png')

print('Applying cutoff')
traj = traj[args.start_index:]

print('Unwrapping trajectory')
traj_unwrapped = unwrap_trajectory(traj)

print('Computing mean squared displacement')
msd = calculate_sqared_disp(traj_unwrapped, n_jobs=-1)
t = np.arange(msd.shape[0])


poly_start = int(0.*len(msd))
poly_stop  = int(1.*len(msd))

fig, ax = plt.subplots()

plt.plot(msd)
poly_start = int(0.*len(msd))
poly_stop  = int(1.*len(msd))
m, b = sp.stats.siegelslopes( 
    msd[poly_start:poly_stop],
t[poly_start:poly_stop],)

ax.text(0.05, 0.85, f'{m=:0.3e}\nD={m/6:0.3e}', transform=ax.transAxes, fontsize=14,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white'))
plt.plot(t, m*t + b)
plt.axvline(poly_start, color='k', linestyle='dotted', label='fitting range')
plt.axvline(poly_stop, color='k', linestyle='dotted')
plt.ylabel(r'$\langle \Delta r^2(t) \rangle$')
plt.xlabel('t (fs)')
plt.legend()
plt.savefig('msd.png')

np.savez('msd.npz', 
         msd=msd,
         traj_unwrapped=traj_unwrapped,
         m=m, b=b, d=m/6
         )
