"""Cascade Runners -- they run cascade"""

from ase.calculators.calculator import Calculator
from ase.io.trajectory import TrajectoryWriter
from ase.io import read
from ase.md.md import MolecularDynamics
from ase import units
from sklearn.model_selection import train_test_split

from cascade.learning.base import BaseLearnableForcefield
from cascade.auditor import BaseAuditor
from cascade.trajectory import CascadeTrajectory
from cascade.utils import apply_calculator


class SerialCascadeRunner:

    def __init__(self,
                 trajectories: list[CascadeTrajectory],
                 total_steps: int,
                 increment_steps: int,
                 uq_threshold: float,
                 auditor: BaseAuditor,
                 calculator: Calculator,
                 learner: BaseLearnableForcefield,
                 model: bytes,
                 training_file: str,
                 dyn_cls: type[MolecularDynamics],  # I wonder if we could be even more generic
                 train_kws: dict = None,
                 val_frac: float = 0.1,
                 max_train: int = None
                 ):
        self.trajectories = trajectories
        self.total_steps = total_steps
        self.increment_steps = increment_steps
        self.uq_threshold = uq_threshold
        self.auditor = auditor
        self.calculator = calculator
        self.learner = learner
        self.model = model
        self.training_file = training_file
        self.dyn_cls = dyn_cls
        self.train_kws = train_kws if train_kws is not None else {}
        self.val_frac = val_frac
        self.max_train = -max_train if max_train is not None else ''  # store as negative for indexing

    @property
    def n_trajectories(self):
        return len(self.trajectories)

    def run(self, max_iter=None):
        i = 0  # track while loop iterations
        done_indices = []
        while True:
            print('*'*10)
            print(f'Starting pass {i+1}/{max_iter} of cascade loop')
            print(f'Currently {len(done_indices)} of {self.n_trajectories} complete')         
            for j, traj in enumerate(self.trajectories):
                print(f'Examining trajectory {j+1} of {self.n_trajectories}')
                if j in done_indices:
                    print('Trajectory is done, continuing')
                    continue
                # if we've advanced past a trusted segment, lets audit it
                if traj.current_timestep > traj.last_trusted_timestep:
                    print('Trajectory has untrusted segment, auditing')
                    self._audit_untrusted_segment(traj)

                    if traj.last_trusted_timestep == self.total_steps:
                        print('Last audit passed; trajectory complete')
                        done_indices.append(j)
                # otherwise we can run the ML-driven dynamics
                else:
                    print('Trajectory is trusted, advancing')
                    self._advance_trajectory(traj)
            i += 1
            # self._update_model()
            if len(done_indices) == self.n_trajectories: 
                print(f'Finished all trajectories in {i} iterations')
                break
            elif i == max_iter:
                print('Hit max iterations, stopping')
                break

    def _update_model(self):
        # this will have to change quite a bit for the parallel version
        print('Updating model')
        train = read(self.training_file, index=f'{self.max_train}:')
        print(f'read {len(train)} frames for training')
        train, val = train_test_split(train, test_size=self.val_frac)
        self.model, perf = self.learner.train(self.model, train, val, **self.train_kws)

    def _advance_trajectory(self, traj):
        """Advance the trajectory under the current ML surrogate"""
        print('Running ML-driven dynamics')
        atoms = traj.trusted.copy()
        atoms.calc = self.learner.make_calculator(self.model, device='cpu')
        dyn = self.dyn_cls(atoms=atoms,
                           timestep=1*units.fs,
                           trajectory=TrajectoryWriter(traj.path, mode='a')
                           )
        dyn.run(self.increment_steps)
        traj.current_timestep += self.increment_steps
        traj.current = atoms

    def _audit_untrusted_segment(self, traj):
        """Audits the untrusted segment of a trajectory

        If the score is above the threshold, the apply the reference calculator
        and update the training set. Else mark the segment as trusted
        """
        print('Auditing trajectory')
        segment = traj.get_untrusted_segment()
        score, audit_frames = self.auditor.audit(segment, n_audits=32)
        if score > self.uq_threshold:
            print(f'score > threshold ({score} > {self.uq_threshold}), running audit calculations and dropping untrusted segment')

            # apply the expensive calculations
            segment = apply_calculator(self.calculator, segment)

            # save calculations to disk
            writer = TrajectoryWriter(self.training_file, mode='a')
            for atoms in segment:
                writer.write(atoms)

            # remove the untrusted calculations
            traj.trim_untrusted_segment()

        else:
            print(f'score < threshold ({score} < {self.uq_threshold}, marking recent segment as trusted')
            traj.last_trusted_timestep = traj.current_timestep
            traj.trusted = traj.current
