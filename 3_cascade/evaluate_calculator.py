"""Evaluate a calculator's energy/forces/stress predictions against ground-truth trajectories
produced by another (reference) calculator, e.g. run_reference_dynamics.py.

Reads frames already written to the DB, and writes the true and predicted values into an extxyz file.
"""
import argparse
import logging
import os

from ase.calculators.calculator import all_changes
from ase.io import write

from cascade.agents.db_orm import TrajectoryDB
from cascade.calculator import get_calc_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-id',
        type=str,
        required=True,
        help='Reference run (written by e.g. run_reference_dynamics.py) to evaluate the calculator against'
    )
    parser.add_argument(
        '--traj-id',
        type=int,
        nargs='+',
        default=None,
        help='Trajectory ids to evaluate (defaults to all trajectories in the run)'
    )
    parser.add_argument(
        '--calc-type',
        type=str,
        choices=['mace', 'fairchem'],
        default='mace',
        help='Which calculator family to evaluate'
    )
    parser.add_argument(
        '--calc-model',
        type=str,
        default='small',
        help='For --calc-type=mace, a MACE-MP model size (e.g. "small") or path to a MACE '
             'checkpoint. For --calc-type=fairchem, the path to a FairChem .pt checkpoint.'
    )
    parser.add_argument(
        '--calc-task',
        type=str,
        default=None,
        help='FairChem task name selecting the model head (e.g. "omol", "omat", "oc20", '
             '"odac", "omc"), ignored for --calc-type=mace. Only needed for --calc-type=fairchem '
             'if the checkpoint supports more than one task'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for the evaluated calculator'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Evaluate every Nth frame of each trajectory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to write paired true/predicted results (extxyz, appended incrementally)'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default=os.environ.get('CASCADE_DB_URL'),
        help='Database URL, e.g. postgresql://ase:pw@<host>:5432/cascade '
             '(defaults to the CASCADE_DB_URL env var)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Logging level'
    )
    return parser.parse_args()


def evaluate_frame(atoms_true, calc, run_id: str, traj_id: int, frame_index: int):
    """Evaluate calc on a copy of atoms_true, returning an Atoms with both the true (already
    attached to atoms_true.calc) and predicted energy/forces/stress stashed side by side."""
    atoms_pred = atoms_true.copy()
    atoms_pred.calc = calc
    calc.calculate(atoms_pred, properties=calc.implemented_properties, system_changes=all_changes)

    results_true = atoms_true.calc.results
    results_pred = atoms_pred.calc.results

    out = atoms_true.copy()
    out.info['run_id'] = run_id
    out.info['traj_id'] = traj_id
    out.info['frame_index'] = frame_index
    out.info['energy_true'] = results_true['energy']
    out.info['energy_pred'] = results_pred['energy']
    out.new_array('forces_true', results_true['forces'])
    out.new_array('forces_pred', results_pred['forces'])
    if 'stress' in results_true and 'stress' in results_pred:
        out.info['stress_true'] = results_true['stress']
        out.info['stress_pred'] = results_pred['stress']
    return out


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger('evaluate_calculator')

    traj_db = TrajectoryDB(args.db_url)
    traj_ids = args.traj_id
    if traj_ids is None:
        traj_ids = [t['traj_id'] for t in traj_db.list_trajectories_in_run(args.run_id)]
        logger.info(f'No --traj-id given, evaluating all {len(traj_ids)} trajectories in run {args.run_id}')

    calc_factory = get_calc_factory(args.calc_type, args.calc_model, args.device, args.calc_task)
    calc = calc_factory()

    n_written = 0
    for traj_id in traj_ids:
        atoms_list = traj_db.get_trajectory_atoms(args.run_id, traj_id)
        logger.info(f'Evaluating traj {traj_id}: {len(atoms_list)} frames, stride={args.stride}')
        for frame_index in range(0, len(atoms_list), args.stride):
            out = evaluate_frame(atoms_list[frame_index], calc, args.run_id, traj_id, frame_index)
            write(args.output, out, format='extxyz', append=True)
            n_written += 1
            if n_written % 100 == 0:
                logger.info(f'Wrote {n_written} evaluated frames so far (traj {traj_id}, frame {frame_index})')

    logger.info(f'Done. Wrote {n_written} evaluated frames to {args.output}')


if __name__ == '__main__':
    main()
