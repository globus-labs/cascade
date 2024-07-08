from glob import glob
from argparse import ArgumentParser
import logging
import sys

import torch
import numpy as np
import pandas as pd
from ase.io import read
from ase import Atoms
import seaborn as sns
import matplotlib.pyplot as plt

# our methods for quickly making ani nets
from cascade.learning.torchani import TorchANI, estimate_atomic_energies
from cascade.learning.torchani.build import make_aev_computer, make_output_nets

def get_traj_contents(files: list[str]) -> np.ndarray[Atoms]: 
    """Read multiple trajectories into a single list of atoms"""
    out = []
    for file in files: 
        out.extend(read(file, index=":"))
    return out

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()

    group = parser.add_argument_group(title='Training Data', description='Setting which control how the training data are generated')
    group.add_argument('--train-files', nargs='+', help='Trajectories used for training')
    group.add_argument('--test-files', nargs='+', help='Trajectories used for testing')
    group.add_argument('--val-frac', default=0.1, help='What fraction of the trajectory to use for validation.')

    group = parser.add_argument_group(title='Model parameters', description='Parameters describing the ANI model')
    group.add_argument('--n_epochs', default=10, help='Number of epochs to train', type=int)
    group.add_argument('--seed', default=0, help='Random seed for bootstrap sampling', type=int)

    args = parser.parse_args()



    logger = logging.getLogger('main')
    handlers = [logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info('Started')

    use_cuda = torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'
    logger.info(f'Using device: {device}')

    # read in train and test data
    logger.info('Reading data')
    train = get_traj_contents(args.train_files)
    if args.test_files:
        test  = get_traj_contents(args.test_files)

    # set random seed 
    rng = np.random.RandomState(args.seed)

    # sample the val set
    n_train = len(train)
    n_val = int(args.val_frac * len(train))
    logger.info(f'{n_train=}, {n_val=}')
    
    val_ix = np.random.choice(n_train, size=(n_val,), replace=False)
    train_ix = np.setdiff1d(np.arange(n_train), val_ix)
    
    valid = [train[i] for i in val_ix]
    train = [train[i] for i in train_ix]

    # construct network
    ref_energies = estimate_atomic_energies(train)
    species = list(ref_energies.keys())
    aev = make_aev_computer(species)
    nn = make_output_nets(species, aev)

    ani = TorchANI()
    orig_e, orig_f = ani.evaluate((aev, nn, ref_energies), 
                              train,
                              device=device)
    
    
    # train network
    model_msg, results = ani.train((aev, nn, ref_energies), 
                               train, 
                               valid, 
                               args.n_epochs,
                               device=device)
    
    
    # save results
    results['seed'] = args.seed
    results.to_csv(f'ani_rep_{args.seed}_results.csv')

    # save model
    with open(f'ensemble/ani_rep_{args.seed}.blob', 'wb') as f: 
        f.write(model_msg)
