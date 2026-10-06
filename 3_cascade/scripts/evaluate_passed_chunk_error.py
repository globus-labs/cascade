"""Backfill ground-truth force error for chunks that PASSED audit in a completed cascade run.

Only chunks that FAILED audit ever get sampled and labeled by the reference calculator
live (that's what drives retraining), so DBTrainingFrame.calibration_error only exists
for those. This script samples frames from each trajectory's accepted (passed) chunks,
runs the same reference calculator the live run used, and scores them with cascade's
own error metric (max_force_error), so passed-chunk error becomes directly comparable
to the existing failed-chunk calibration_error values.

Results are written to the reference_evaluations table (DBReferenceEvaluation) via
TrajectoryDB.add_reference_evaluation -- a separate table from DBTrainingFrame, which
backs the live Controller's calibration and must never be mixed with this offline data.
The table auto-creates on connection; no migration step is needed.

    python scripts/evaluate_passed_chunk_error.py \\
        --run-id 2026.09.03-22:03:35-e79143 \\
        --calc-type fairchem \\
        --calc-model ../1_ml-potential/uma/uma_omat_ft_mofoff_r2scan.pt \\
        --n-samples-per-chunk 20 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import logging
import os

import numpy as np
from ase.calculators.calculator import all_changes

from cascade.agents.db_orm import TrajectoryDB, DBTrajectoryFrame
from cascade.agents.task import max_force_error
from cascade.calculator import get_calc_factory

logger = logging.getLogger('evaluate_passed_chunk_error')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', type=str, required=True,
                        help='Completed cascade run to backfill error for')
    parser.add_argument('--traj-id', type=int, nargs='+', default=None,
                        help='Trajectory ids to evaluate (defaults to all trajectories in the run)')
    parser.add_argument('--calc-type', type=str, choices=['mace', 'fairchem'], default='fairchem',
                        help='Reference calculator family to run as ground truth against the '
                             "surrogate's stored predictions (NOT a calculator under test)")
    parser.add_argument('--calc-model', type=str, required=True,
                        help='For --calc-type=mace, a MACE-MP model size or checkpoint path. '
                             'For --calc-type=fairchem, the path to a FairChem .pt checkpoint '
                             '(e.g. the same UMA checkpoint the live run used as its reference)')
    parser.add_argument('--calc-task', type=str, default=None,
                        help='FairChem task name selecting the model head, ignored for '
                             '--calc-type=mace. Only needed if the checkpoint supports more '
                             'than one task')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for the reference calculator')
    parser.add_argument('--n-samples-per-chunk', type=int, default=10,
                        help='Number of frames to sample per passed (chunk_id, attempt_index), '
                             'sampled after deduplicating the chunk-boundary overlap frame')
    parser.add_argument('--sample-mode', type=str, choices=['even', 'random'], default='even')
    parser.add_argument('--seed', type=int, default=0,
                        help='Only used for --sample-mode=random')
    parser.add_argument('--csv-export', type=str, default=None,
                        help='Optional: also dump all reference_evaluations rows for this run '
                             'to a CSV at this path once done, for a quick local copy')
    parser.add_argument('--db-url', type=str, default=os.environ.get('CASCADE_DB_URL'),
                        help='Database URL, e.g. postgresql://ase:pw@<host>:5432/cascade '
                             '(defaults to the CASCADE_DB_URL env var)')
    parser.add_argument('--log-level', type=str, default='INFO')
    return parser.parse_args()


def sample_indices(n_pool: int, n_samples: int, mode: str, rng: np.random.Generator) -> list[int]:
    """Sorted, ascending, no-replacement indices into a 0..n_pool-1 pool."""
    n = min(n_samples, n_pool)
    if mode == 'even':
        return sorted(set(np.linspace(0, n_pool - 1, n).round().astype(int).tolist()))
    return sorted(rng.choice(n_pool, size=n, replace=False).tolist())


def deduped_chunk_frames(sess, run_id: str, traj_id: int, chunk: dict, is_first_chunk: bool):
    """Frame rows for a passed (chunk_id, attempt_index), deduplicated the same way
    TrajectoryDB.get_trajectory_atoms does: the first frame of every chunk after the
    first is a duplicate of the previous chunk's last frame."""
    frames = sess.query(DBTrajectoryFrame).filter_by(
        run_id=run_id, traj_id=traj_id,
        chunk_id=chunk['chunk_id'], attempt_index=chunk['attempt_index'],
    ).order_by(DBTrajectoryFrame.frame_index).all()
    if not is_first_chunk:
        frames = frames[1:]
    return frames


def evaluate_row(frame_id: int, frame_index: int, atoms_surrogate, chunk: dict, calc,
                 run_id: str, traj_id: int, db: TrajectoryDB, calc_type: str, calc_model: str) -> dict:
    atoms_reference = atoms_surrogate.copy()
    atoms_reference.calc = calc
    calc.calculate(atoms_reference, properties=calc.implemented_properties, system_changes=all_changes)

    force_error = max_force_error(atoms_surrogate, atoms_reference)

    uq = atoms_surrogate.info.get('uq_force_std_max')
    energy_surrogate = atoms_surrogate.calc.results.get('energy')
    energy_reference = atoms_reference.calc.results.get('energy')

    return db.add_reference_evaluation(
        run_id=run_id,
        traj_id=traj_id,
        chunk_id=chunk['chunk_id'],
        attempt_index=chunk['attempt_index'],
        model_version=chunk['model_version'],
        trajectory_frame_id=frame_id,
        frame_index=frame_index,
        force_error=float(force_error),
        calc_type=calc_type,
        calc_model=calc_model,
        # psycopg2 can't adapt numpy scalars (some ASE calculators return np.float64 for
        # scalar properties); cast to native float before it ever reaches the DB layer.
        uq=None if uq is None else float(uq),
        energy_surrogate=None if energy_surrogate is None else float(energy_surrogate),
        energy_reference=None if energy_reference is None else float(energy_reference),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level)
    rng = np.random.default_rng(args.seed)

    db = TrajectoryDB(args.db_url)
    traj_ids = args.traj_id
    if traj_ids is None:
        traj_ids = [t['traj_id'] for t in db.list_trajectories_in_run(args.run_id)]
        logger.info(f'No --traj-id given, evaluating all {len(traj_ids)} trajectories in run {args.run_id}')

    calc_factory = get_calc_factory(args.calc_type, args.calc_model, args.device, args.calc_task)

    # Resumability: add_reference_evaluation is idempotent on the DB row, but running the
    # (possibly expensive) reference calculator again for an already-evaluated frame just
    # to throw the result away on conflict would defeat the point. Skip those frames here,
    # before the calculator ever touches them.
    already_done = {
        r['trajectory_frame_id']
        for r in db.get_reference_evaluations(args.run_id)
        if r['calc_type'] == args.calc_type and r['calc_model'] == args.calc_model
    }
    if already_done:
        logger.info(f'{len(already_done)} frames already evaluated for this run/calculator; skipping those')

    n_evaluated = 0
    n_skipped = 0
    calc = None
    for traj_id in traj_ids:
        # Rebuild the calculator fresh per trajectory rather than reusing one for the whole
        # run: fairchem/UMA's fallback path for heterogeneous compositions (triggered when a
        # new trajectory's cell/composition differs from what it last compiled for) appears to
        # accumulate memory across distinct compositions without releasing it, which crashed
        # this script via an external OOM-kill partway through the second trajectory in
        # testing. Recreating the calculator bounds that growth to within one trajectory.
        # Drop the old one and force a GC pass before freeing CUDA's cache -- empty_cache()
        # only returns memory Python has already released back to it.
        if calc is not None:
            del calc
            gc.collect()
            if args.device.startswith('cuda'):
                import torch
                torch.cuda.empty_cache()
        calc = calc_factory()

        chunks = db.get_passed_chunks(args.run_id, traj_id)
        logger.info(f'Traj {traj_id}: {len(chunks)} passed chunks')

        for i, chunk in enumerate(chunks):
            with db.session() as sess:
                frames = deduped_chunk_frames(sess, args.run_id, traj_id, chunk, is_first_chunk=(i == 0))

                if len(frames) < args.n_samples_per_chunk:
                    logger.warning(
                        f"traj {traj_id} chunk {chunk['chunk_id']} attempt {chunk['attempt_index']}: "
                        f'only {len(frames)} frames available, requested {args.n_samples_per_chunk}; '
                        'using all of them'
                    )

                indices = sample_indices(len(frames), args.n_samples_per_chunk, args.sample_mode, rng)
                sampled_frames = [frames[idx] for idx in indices if frames[idx].id not in already_done]
                n_skipped += len(indices) - len(sampled_frames)

                # Deserialize while the session is still open (avoids a detached-instance
                # error on `.id`/`.frame_index` once the `with` block exits), then run the
                # (possibly slow) reference calculator on each outside the session.
                sampled = [
                    (f.id, f.frame_index, db._deserialize_atoms(f.atoms_blob))
                    for f in sampled_frames
                ]

            for frame_id, frame_index, atoms_surrogate in sampled:
                evaluate_row(frame_id, frame_index, atoms_surrogate, chunk, calc,
                            args.run_id, traj_id, db, args.calc_type, args.calc_model)
                n_evaluated += 1
                if n_evaluated % 20 == 0:
                    logger.info(f'Evaluated {n_evaluated} frames so far '
                               f"(traj {traj_id}, chunk {chunk['chunk_id']})")

    logger.info(f'Done. Evaluated {n_evaluated} frames ({n_skipped} already done, skipped) '
               f'for run {args.run_id}.')

    if args.csv_export:
        import pandas as pd
        rows = []
        for traj_id in traj_ids:
            rows.extend(db.get_reference_evaluations(args.run_id, traj_id))
        pd.DataFrame(rows).to_csv(args.csv_export, index=False)
        logger.info(f'Exported {len(rows)} rows to {args.csv_export}')


if __name__ == '__main__':
    main()
