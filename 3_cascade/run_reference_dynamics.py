"""Run reference trajectories with a fixed reference calculator - no MLFF, no auditing, no training.

Useful for generating ground-truth trajectories (e.g. to seed a replay dataset or validate
the cascade pipeline's output against). Reuses the same --init-config-json format as
run_cascade_academy.py, and writes frames into the same TrajectoryDB, so existing notebook
tooling (get_trajectory_atoms, etc.) works unchanged on these runs too.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import warnings
from concurrent.futures import as_completed

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.usage_tracking.levels import LEVEL_1
from parsl.concurrent import ParslPoolExecutor

from cascade.traj_config import InitialTrajConfig, load_initial_configs

warnings.filterwarnings("ignore", category=FutureWarning, module="mace.calculators")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--init-config-json',
        type=str,
        required=True,
        help='Same JSON format as run_cascade_academy.py --init-config-json'
    )
    parser.add_argument(
        '--target-length',
        type=int,
        default=10,
        help='Number of steps to run per trajectory'
    )
    parser.add_argument(
        '--calc-type',
        type=str,
        choices=['mace', 'fairchem'],
        default='mace',
        help='Which reference calculator family to use'
    )
    parser.add_argument(
        '--calc-model',
        type=str,
        default='medium',
        help='For --calc-type=mace, a MACE-MP model size (e.g. "medium") or path to a MACE '
             'checkpoint. For --calc-type=fairchem, the path to a FairChem .pt checkpoint.'
    )
    parser.add_argument(
        '--calc-task',
        type=str,
        default=None,
        help='FairChem task name selecting the model head (e.g. "omol", "omat", "oc20", '
             '"odac", "omc"), ignored for --calc-type=mace. Only needed for --calc-type=fairchem '
             'if the checkpoint supports more than one task; single-task checkpoints infer it '
             'automatically.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for the reference calculator'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Number of trajectories to run concurrently (defaults to one worker per trajectory)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='Update the chunk progress/audit-status record every this many steps, '
             'so get_trajectory_atoms reflects live progress rather than only the final result'
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


def run_reference_trajectory(
    traj_id: int,
    cfg: InitialTrajConfig,
    run_id: str,
    db_url: str,
    target_length: int,
    calc_type: str,
    calc_model: str,
    calc_task: str | None,
    device: str,
    log_level: str,
    log_interval: int,
) -> None:
    """Run one trajectory to completion using a fixed reference calculator, writing frames directly to the DB"""
    import csv
    import logging
    import psutil
    import torch
    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from cascade.agents.db_orm import TrajectoryDB
    from cascade.utils import canonicalize
    from cascade.traj_config import get_dynamics_cls, resolve_dyn_kws, prepare_atoms_for_dynamics
    from cascade.model import AuditStatus
    from cascade.calculator import get_calc_factory
    calc_factory = get_calc_factory(calc_type, calc_model, device, calc_task)
    is_cuda = torch.cuda.is_available() and 'cuda' in device

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(f'reference.traj{traj_id}')

    traj_db = TrajectoryDB(db_url)
    atoms = read(cfg.path, index=-1)
    atoms = prepare_atoms_for_dynamics(atoms, cfg)

    if cfg.temperature_K is not None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=cfg.temperature_K)

    traj_db.initialize_trajectory(run_id=run_id, traj_id=traj_id, target_length=target_length, init_atoms=atoms)

    atoms.calc = calc_factory()
    dyn_cls = get_dynamics_cls(cfg.dyn_cls)
    dyn = dyn_cls(atoms, **resolve_dyn_kws(cfg))

    frame_index = 0

    def write_frame():
        nonlocal frame_index
        traj_db.write_frame(
            run_id=run_id,
            traj_id=traj_id,
            chunk_id=0,
            attempt_index=0,
            frame_index=frame_index,
            atoms=canonicalize(atoms),
        )
        frame_index += 1

    def mark_progress():
        # There's no audit step in reference mode - these frames come straight from the
        # reference calculator, so the chunk is trivially "passed" by construction. Without
        # this record, get_trajectory_atoms (which only returns frames from PASSED chunks)
        # would find nothing despite the frames being written by write_frame. Called on its
        # own interval (independent of write_frame) so a caller can watch progress live via
        # get_trajectory_atoms rather than only seeing the result once the trajectory finishes.
        traj_db.add_chunk_attempt(
            run_id=run_id,
            traj_id=traj_id,
            chunk_id=0,
            model_version=0,
            n_frames=frame_index,
            audit_status=AuditStatus.PASSED,
            attempt_index=0,
        )

    # Per-task loggers never surface anywhere without digging through Parsl's runinfo/
    # (confirmed: reference_run.*.log has always been empty of this output), so this
    # writes a plain CSV next to gpu_mon.*.csv/vmstat.*.log instead - visible with a
    # normal `ls`/`cat` in the run directory, no runinfo spelunking required.
    mem_csv = open(f'gpu_mem.{run_id}.traj{traj_id}.csv', 'w', newline='')
    mem_writer = csv.writer(mem_csv)
    mem_writer.writerow(['step', 'host_rss_mb', 'host_available_mb', 'gpu_allocated_mb', 'gpu_reserved_mb'])

    def manage_and_log_gpu_memory():
        # Periodically release PyTorch's CUDA caching allocator back to the driver. Without
        # this, long NPT/MTKNPT runs (where the cell - and therefore the neighbor-list edge
        # count fed to MACE - keeps changing shape) grow the allocator's reserved pool
        # without bound, and on unified-memory hardware (e.g. GB10) that pool is host RAM,
        # so it eventually starves the node.
        if is_cuda:
            torch.cuda.empty_cache()
        gpu_alloc = torch.cuda.memory_allocated(device) / 1e6 if is_cuda else 0.0
        gpu_reserved = torch.cuda.memory_reserved(device) / 1e6 if is_cuda else 0.0
        mem_writer.writerow([
            dyn.nsteps,
            round(psutil.Process().memory_info().rss / 1e6, 1),
            round(psutil.virtual_memory().available / 1e6, 1),
            round(gpu_alloc, 1),
            round(gpu_reserved, 1),
        ])
        mem_csv.flush()

    dyn.attach(write_frame)
    dyn.attach(mark_progress, interval=log_interval)
    dyn.attach(manage_and_log_gpu_memory, interval=log_interval)
    logger.info(f'Starting reference dynamics for traj {traj_id}, {target_length} steps')
    dyn.run(target_length, **cfg.run_kws)

    mark_progress()  # make sure the final frame count is recorded even if target_length
                      # isn't a multiple of log_interval
    manage_and_log_gpu_memory()
    mem_csv.close()
    traj_db.mark_trajectory_completed(run_id=run_id, traj_id=traj_id)
    logger.info(f'Finished reference dynamics for traj {traj_id}')


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger('reference')

    params = args.__dict__.copy()
    start_time = datetime.datetime.utcnow().strftime("%Y.%m.%d-%H:%M:%S")
    params_hash = hashlib.sha256(json.dumps(params).encode()).hexdigest()[:6]
    run_id = f"reference-{start_time}-{params_hash}"

    init_configs = load_initial_configs(args.init_config_json)

    max_workers = args.max_workers or len(init_configs)
    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex_local",
                max_workers_per_node=max_workers,
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                ),
            )
        ],
        usage_tracking=LEVEL_1,
    )

    logger.info(f'Running {len(init_configs)} reference trajectories under run_id={run_id}')
    with ParslPoolExecutor(config=config) as pool:
        futures = [
            pool.submit(
                run_reference_trajectory,
                traj_id, cfg, run_id, args.db_url, args.target_length,
                args.calc_type, args.calc_model, args.calc_task, args.device, args.log_level, args.log_interval,
            )
            for traj_id, cfg in enumerate(init_configs)
        ]
        for future in as_completed(futures):
            future.result()  # re-raise any exception from the worker

    logger.info(f'Done. run_id={run_id}')


if __name__ == '__main__':
    main()
