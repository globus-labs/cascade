"""ASE-compatible implementations of online learning strategy of
`Zamora et al. <https://dl.acm.org/doi/abs/10.1145/3447818.3460370>`_, Proxima."""
from collections import deque
from typing import List, Optional
from pathlib import Path
import logging

import numpy as np
from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.db import connect
from ase.io import read
from sklearn.model_selection import train_test_split
from scipy.stats import linregress

from cascade.learning.base import BaseLearnableForcefield
from cascade.calculator import EnsembleCalculator

logger = logging.getLogger(__name__)


class SerialLearningCalculator(Calculator):
    """A Calculator which switches between a physics-based calculator and one
    being trained to emulate it.

    Determines when to switch between the physics and learnable calculator based
    on a uncertainty metric from the learnable calculator.

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
    target_ferr: float
        Target maximum difference between the forces predicted by the target
        calculator and the learnable surrogate
    history_length: int
        The number of previous observations of the error between target and surrogate
        function to use when establishing a link between uncertainty metric
        and the maximum observed error
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
        'target_ferr': 0.1,  # TODO (wardlt): Make the error metric configurable
        'history_length': 8,
        'db_path': None
    }

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

    def retrain_surrogate(self):
        """Retrain the surrogate models using the currently-available data"""

        # Load in the data from the db
        db_path = self.parameters['db_path']
        if not Path(db_path).is_file():
            logger.debug(f'No data at {db_path} yet')
            return

        all_atoms = read(db_path, ':')
        logger.info(f'Loaded {len(all_atoms)} from {db_path} for retraining {len(self.parameters["models"])} models')
        if len(all_atoms) < 10:
            logger.debug('Too few entries to retrain')
            return

        # Train each model using a different, randomly-selected subset of the data
        model_list = self.parameters['models']  # Edit it in place
        for i, model_msg in enumerate(self.parameters['models']):
            train_atoms, valid_atoms = train_test_split(all_atoms, test_size=0.1)
            new_model_msg, _ = self.learner.train(model_msg, train_atoms, valid_atoms, **self.parameters['train_kwargs'])
            model_list[i] = new_model_msg
            logger.debug(f'Finished training model {i}')

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

        # Use the result from the surrogate
        self.used_surrogate = self.threshold is not None and unc_metric < self.threshold
        if self.used_surrogate:
            logger.debug(f'The uncertainty metric is low enough ({unc_metric:.2e} < {self.threshold:.2e}). Using the surrogate result.')
            self.results = self.surrogate_calc.results.copy()
            return

        # If not, run the target calculator and use that result
        target_calc: Calculator = self.parameters['target_calc']
        target_calc.calculate(atoms, properties, system_changes)
        self.results = target_calc.results.copy()

        # Increment the training set with this new result, then throw out the current model
        db_atoms = atoms.copy()
        db_atoms.calc = target_calc
        with connect(self.parameters['db_path']) as db:
            db.write(db_atoms)
        surrogate_forces = self.surrogate_calc.results['forces']
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
        many_alphas = np.true_divide(obs_errors, uncert_metrics)
        self.alpha = np.percentile(many_alphas, 50)
        bad_alpha = self.alpha < 0
        if bad_alpha:
            logger.warning(f'Alpha parameter was less than zero ({self.alpha:.2e}).'
                           ' Will not adjust the threshold until this condition changes')
            return

        # Update the threshold used to determine if the surrogate is usable
        if self.threshold is None:
            # Use the initial estimate for alpha to set a conservative threshold
            #  Following Eq. 1 of https://dl.acm.org/doi/abs/10.1145/3447818.3460370,
            self.threshold = self.parameters['target_ferr'] / self.alpha
            self.threshold *= 2  # Make the threshold even higher than we initially estimate
        else:
            # Update according to Eq. 3 of https://dl.acm.org/doi/abs/10.1145/3447818.3460370
            current_err = np.mean([e for _, e in self.error_history])
            self.threshold -= (current_err - self.parameters['target_ferr']) / self.alpha
