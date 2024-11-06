"""ASE-compatible implementations of online learning strategy of
`Zamora et al. <https://dl.acm.org/doi/abs/10.1145/3447818.3460370>`_, Proxima."""
from collections import deque
from typing import List, Optional, Any
from pathlib import Path
from random import random
import logging

import numpy as np
import pandas as pd
from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.db import connect

from cascade.learning.base import BaseLearnableForcefield
from cascade.calculator import EnsembleCalculator
from cascade.utils import to_voigt, canonicalize

logger = logging.getLogger(__name__)


class SerialLearningCalculator(Calculator):
    """A Calculator which switches between a physics-based calculator and one
    being trained to emulate it.

    Determines when to switch between the physics and learnable calculator based
    on an uncertainty metric from the learnable calculator.

    Will run `history_length` steps under physics before considering surrogate

    Switching can be applied smoothly, that is by taking a mixture of physics-
    and surrogate-derived quantities that moves slowly between full surrogate
    and full physics utilization over time. The rate of this smoothing is
    controleld by the parameters n_blending_steps

    Parameters for the calculator are:

    target_calc: BaseCalculator
        A physics-based calculator used to provide training data for learnable
    learner: BaseLearnableForcefield
        A class used to train the forcefield and generate an updated calculator
    models: list of State associated with learner
        One or more set of objects that define the architecture and weights of the
        surrogate model. These weights are used by the learner.
    device: str
        Device used for running the learned surrogates
    train_kwargs: dict
        Dictionary of parameters passed to the train function of the learner
    train_freq: int
        After how many new data points to retrain the surrogate model
    train_max_size: int, optional
        Maximum size of training set to use when updating model. Set to ``None`` to use all data
    train_recency_bias: float
        Bias towards selecting newer points if using only a subset of the available training data.
        Weights will be assigned to each point in a geometric series such that the most recent
        point is ``train_recency_bias`` times more likely to be selected than the least recent.
    target_ferr: float
        Target maximum difference between the forces predicted by the target
        calculator and the learnable surrogate
    history_length: int
        The number of previous observations of the error between target and surrogate
        function to use when establishing a link between uncertainty metric
        and the maximum observed error. Will run exactly this number of target
        calculations before considering using the surrogate
    min_target_fraction: float
        Minimum fraction of timesteps to run the target function.
        This value is used as the probability of running the target function
        even if it need not be used based on the UQ metric.
    n_blending_steps: int
        How many timesteps to smoothly combine target and surrogate forces.
        When the threshold is satisy we apply an increasing mixture of ML and
        target forces.
    db_path: Path or str
        Database in which to store the results of running the target calculator,
        which are used to train the surrogate model
    """

    default_parameters = {
        'target_calc': None,
        'learner': None,
        'models': None,
        'device': 'cpu',
        'train_kwargs': {'num_epochs': 8},
        'train_freq': 1,
        'train_max_size': None,
        'train_recency_bias': 1.,
        'target_ferr': 0.1,  # TODO (wardlt): Make the error metric configurable
        'min_target_fraction': 0.,
        'n_blending_steps': 0,
        'history_length': 8,
        'db_path': None
    }

    train_logs: Optional[List[pd.DataFrame]] = None
    """Logs from the most recent model training"""
    surrogate_calc: Optional[EnsembleCalculator] = None
    """Cache for the surrogate calculator"""
    error_history: Optional[deque[tuple[float, float]]] = None
    """History of pairs of the uncertainty metric and observed error"""
    alpha: Optional[float] = None
    """Coefficient which relates distrust metric and observed error"""
    threshold: Optional[float] = None
    """Current threshold for the uncertainty metric beyond which the target calculator will be used"""
    used_surrogate: Optional[bool] = None
    """Whether the last invocation used the surrogate model"""
    new_points: int = 0
    """How many new points have been acquired since the last model update"""
    total_invocations: int = 0
    """Total number of calls to the calculator"""
    target_invocations: int = 0
    """Total number of calls to the target calculator"""
    blending_step: np.int64 = np.int_(0)
    """Ranges from 0 to n_blending_steps, corresponding to
    full surrogate and full physics, respectively"""
    lambda_target: float = 1.
    """Ranges from 0-1, describing mixture between surrogate and physics"""
    model_version: int = 0
    """How many times the model has been retrained"""

    def set(self, **kwargs):
        # TODO (wardlt): Fix ASE such that it does not try to do a numpy comparison on everything
        self.parameters.update(kwargs)
        self.reset()

    @property
    def implemented_properties(self) -> List[str]:
        return self.parameters['target_calc'].implemented_properties

    @property
    def learner(self) -> BaseLearnableForcefield:
        return self.parameters['learner']

    @staticmethod
    def smoothing_function(x):
        """Smoothing used for blending surrogate with physics"""
        return 0.5*((np.cos(np.pi*x)) + 1)

    def retrain_surrogate(self):
        """Retrain the surrogate models using the currently-available data"""

        # Load in the data from the db
        db_path = self.parameters['db_path']
        if not Path(db_path).is_file():
            logger.debug(f'No data at {db_path} yet')
            return

        # Retrieve the data such that the oldest training entry is first
        with connect(db_path) as db:
            all_atoms = [a for a in db.select('', sort='-age')]
            assert len(all_atoms) < 2 or all_atoms[0].ctime < all_atoms[1].ctime, 'Logan got the sort order backwards'
            all_atoms = [a.toatoms() for a in all_atoms]
        logger.info(f'Loaded {len(all_atoms)} from {db_path} for retraining {len(self.parameters["models"])} models')
        if len(all_atoms) < 10:
            logger.info('Too few entries to retrain. Skipping')
            return

        # Train each model using a different, randomly-selected subset of the data
        model_list = self.parameters['models']  # Edit it in place
        self.train_logs = []
        for i, model_msg in enumerate(self.parameters['models']):
            # Assign splits such that the same entries do not switch between train/validation as test grows
            rng = np.random.RandomState(i)
            is_train = rng.uniform(0, 1, size=(len(all_atoms),)) > 0.1  # TODO (wardlt): Make this configurable
            train_atoms = [all_atoms[i] for i in np.where(is_train)[0]]  # Where preserves sort
            valid_atoms = [all_atoms[i] for i in np.where(np.logical_not(is_train))[0]]

            # Downselect training set if it is larger than the fixed maximum
            train_max_size = self.parameters['train_max_size']
            if train_max_size is not None and len(train_atoms) > train_max_size:
                valid_size = train_max_size * len(valid_atoms) // len(train_atoms)  # Decrease the validation size proportionally

                train_weights = np.geomspace(1, self.parameters['train_recency_bias'], len(train_atoms))
                train_ids = rng.choice(len(train_atoms), size=(train_max_size,), p=train_weights / train_weights.sum(), replace=False)
                train_atoms = [train_atoms[i] for i in train_ids]

                if valid_size > 0:
                    valid_weights = np.geomspace(1, self.parameters['train_recency_bias'], len(valid_atoms))
                    valid_ids = rng.choice(len(valid_atoms), size=(valid_size,), p=valid_weights / valid_weights.sum(), replace=False)
                    valid_atoms = [valid_atoms[i] for i in valid_ids]

            logger.debug(f'Training model {i} on {len(train_atoms)} atoms and validating on {len(valid_atoms)}')
            new_model_msg, log = self.learner.train(model_msg, train_atoms, valid_atoms, **self.parameters['train_kwargs'])
            model_list[i] = new_model_msg
            self.train_logs.append(log)
            logger.debug(f'Finished training model {i}')
        self.model_version += 1

    def calculate(
            self, atoms=None, properties=all_properties, system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)

        # Start by running an ensemble of surrogate models
        if self.surrogate_calc is None:
            self.retrain_surrogate()
            self.surrogate_calc = EnsembleCalculator(
                calculators=[self.learner.make_calculator(m, self.parameters['device']) for m in self.parameters['models']]
            )
        self.surrogate_calc.calculate(atoms, properties + ['forces'], system_changes)  # Make sure forces are computed too

        # Compute an uncertainty metric for the ensemble model
        #  We use, for now, the maximum mean difference in force prediction for over all atoms.
        forces_ens = self.surrogate_calc.results['forces_ens']
        forces_diff = np.linalg.norm(forces_ens - self.surrogate_calc.results['forces'][None, :, :], axis=-1).mean(axis=0)  # Mean diff per atom
        unc_metric = forces_diff.max()
        logger.debug(f'Computed the uncertainty metric for the model to be: {unc_metric:.2e}')

        # Check whether to use the result from the surrogate
        uq_small_enough = self.threshold is not None and unc_metric < self.threshold
        self.used_surrogate = uq_small_enough and (random() > self.parameters['min_target_fraction'])
        self.total_invocations += 1

        # Track blending parameters for surrogate/target
        increment = +1 if self.used_surrogate else -1
        self.blending_step = np.clip(self.blending_step + increment, 0, self.parameters['n_blending_steps'])
        self.lambda_target = self.smoothing_function(self.blending_step / self.parameters['n_blending_steps'])

        # Case: fully use the surrogate
        if self.used_surrogate and self.blending_step == self.parameters['n_blending_steps']:
            logger.debug(f'The uncertainty metric is low enough ({unc_metric:.2e} < {self.threshold:.2e}). Using the surrogate result.')
            self.results = self.surrogate_calc.results.copy()
            return

        # If not, run the target calculator and use that result
        target_calc: Calculator = self.parameters['target_calc']
        target_calc.calculate(atoms, properties, system_changes)
        self.target_invocations += 1

        if self.blending_step > 0:
            # return a blend if appropriate
            results_target = target_calc.results
            results_surrogate = self.surrogate_calc.results
            self.results = {}
            for k in results_surrogate.keys():
                if k in results_target.keys():  # blend on the intersection of keys
                    r_target, r_surrogate = results_target[k], results_surrogate[k]
                    #  handle differences in voigt vs (3,3) stress convention
                    if k == 'stress' and r_target.shape != r_surrogate.shape:
                        r_target, r_surrogate = map(to_voigt, [r_target, r_surrogate])
                    self.results[k] = self.lambda_target*r_target + (1-self.lambda_target)*r_surrogate
                else:
                    # the surrogate may have some extra results which we store
                    self.results[k] = results_surrogate[k]
        else:
            # otherwise just return the target
            self.results = target_calc.results.copy()

        # Increment the training set with this new result
        db_atoms = atoms.copy()
        db_atoms.calc = target_calc
        with connect(self.parameters['db_path']) as db:
            db.write(canonicalize(db_atoms))

        # Reset the model if the training frequency has been reached
        surrogate_forces = self.surrogate_calc.results['forces']
        self.new_points = (self.new_points + 1) % self.parameters['train_freq']
        if self.new_points == 0:
            self.surrogate_calc = None

        # Update the alpha parameter, which relates uncertainty and observed error
        #  See Section 3.2 from https://dl.acm.org/doi/abs/10.1145/3447818.3460370
        #  Main difference: We do not fit an intercept when estimating \alpha
        actual_err = np.linalg.norm(target_calc.results['forces'] - surrogate_forces, axis=-1).max()
        if self.error_history is None:
            self.error_history = deque(maxlen=self.parameters['history_length'])
        self.error_history.append((unc_metric, actual_err))

        if len(self.error_history) < self.parameters['history_length']:
            logger.debug(f'Too few entries in training history. {len(self.error_history)} < {self.parameters["history_length"]}')
            return
        uncert_metrics, obs_errors = zip(*self.error_history)

        # Special case: uncertainty metrics are all zero. Happens when using the same pre-trained weights for whole ensemble.
        all_zero = np.allclose(uncert_metrics, 0.)
        if all_zero:
            logger.debug('All uncertainty metrics are zero. Setting threshold to zero')
            self.threshold = 0.
            return

        many_alphas = np.true_divide(obs_errors, np.clip(uncert_metrics, 1e-6, a_max=np.inf))  # Alpha's units: error / UQ
        self.alpha = np.mean(many_alphas)
        assert self.alpha >= 0

        # Update the threshold used to determine if the surrogate is usable
        if self.threshold is None:
            # Use the initial estimate for alpha to set a conservative threshold
            #  Following Eq. 1 of https://dl.acm.org/doi/abs/10.1145/3447818.3460370,
            self.threshold = self.parameters['target_ferr'] / self.alpha  # Units: error / (error / UQ) -> UQ
            self.threshold /= 2  # Make the threshold even stricter than we estimate TODO (wardlt): Make this adjustable
        else:
            # Update according to Eq. 3 of https://dl.acm.org/doi/abs/10.1145/3447818.3460370
            current_err = np.mean([e for _, e in self.error_history])
            self.threshold -= (current_err - self.parameters['target_ferr']) / self.alpha
            self.threshold = max(self.threshold, 0)  # Keep it at least zero (assuming UQ signals are nonnegative)

    def get_state(self) -> dict[str, Any]:
        """Get the state of the learner in a state that can be saved to disk using pickle

        The state contains the current threshold control parameters, error history, retraining status, and the latest models.

        Returns:
            Dictionary containing the state of the model(s)
        """

        output = {
            'threshold': self.threshold,
            'alpha': self.alpha,
            'blending_step': int(self.blending_step),
            'error_history': list(self.error_history),
            'new_points': self.new_points,
            'train_logs': self.train_logs,
            'total_invocations': self.total_invocations,
            'target_invocations': self.target_invocations,
            'model_version': self.model_version
        }
        if self.surrogate_calc is not None:
            output['models'] = [self.learner.serialize_model(s) for s in self.parameters['models']]
        return output

    def set_state(self, state: dict[str, Any]):
        """Set the state of learner using the state saved by :meth:`get_state`

        Args:
            state: State containing the threshold control system parameters and trained models, if available
        """

        # Set the state of the threshold
        self.alpha = state['alpha']
        self.blending_step = state['blending_step']
        self.threshold = state['threshold']
        self.new_points = state['new_points']
        self.error_history = deque(maxlen=self.parameters['history_length'])
        self.error_history.extend(state['error_history'])
        self.train_logs = state['train_logs']
        self.total_invocations = state['total_invocations']
        self.target_invocations = state['target_invocations']
        self.model_version = state['model_version']

        # Remake the surrogate calculator, if available
        if 'models' in state:
            self.parameters['models'] = state['models']
            self.surrogate_calc = EnsembleCalculator(
                calculators=[self.learner.make_calculator(m, self.parameters['device']) for m in state['models']]
            )

    def todict(self, skip_default=True):
        # Never skip defaults because testing for equality between current and default breaks for our data types
        output = super().todict(False)

        # The models don't json serialize, so let's skip them
        output.pop('models')
        return output
