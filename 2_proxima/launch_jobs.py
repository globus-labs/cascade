import subprocess
from itertools import product
import pandas as pd
from pathlib import Path


# replicate variables
start_seed = 0
n_replicates = 1

# dynamics variables
temp = 1573
steps = 50_000
ens='nvt'
vol=1376 # the average equil volume from 2 dft runs in our previous npt sweep

# proxima variables
min_target_frac = 0.1
ferror = 10
max_retrain = 512
retrain_freq = 64
epochs = 16
n_blending_steps=[10]

# I set this up so that we'd get all three conditions for each seed
# together, so we can analyze the data as it comes in
for seed in range(start_seed, n_replicates):
    for blend in n_blending_steps:
        retrain_freqs = [retrain_freq] 
        min_target_fracs = [min_target_frac]
        if blend == 0:
            retrain_freqs.append(steps + 10) # so we never train the surrogate
            min_target_fracs.append(1.)      # so we never use the surrogate 
        for retf, tfrac in zip(retrain_freqs, min_target_fracs):
            cmd = f'qsub -v blend={blend},ferr={ferror:.12f},frac={tfrac},seed={seed},steps={steps},retrain_freq={retf},temp={temp},max_retrain={max_retrain},ens={ens},epochs={epochs},vol={vol} ./improv-proxima.sh'
            # if tfrac == 1: 
            #     continue
            print(f'running: "{cmd}"')
            subprocess.run(cmd.split(' '))