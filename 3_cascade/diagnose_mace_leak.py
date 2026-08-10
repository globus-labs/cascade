"""Standalone repro for the host-memory leak observed in run_reference_dynamics.py.

Deliberately excludes everything cascade-specific *except* the pure dynamics-config
helpers in cascade.traj_config (dataclasses, no I/O/DB/Parsl): no TrajectoryDB, no
canonicalize/write_frame, no Parsl/HTEX. If the leak still shows up here, it lives in
ASE/MACE/torch, not in cascade's DB or executor code.

Logs host RSS, available host memory, GPU allocated/reserved memory, and the current
neighbor-list edge count (a proxy for how much the batch shape is changing step to
step - relevant for NPT/MTKNPT runs where the cell, and therefore the neighbor list,
changes continuously). Writes one CSV row per log interval, flushed immediately, so a
crash/OOM still leaves usable data.

Usage (mirrors the settings in run_reference_dynamics.sh / init_config_mof_crystalline_200_300K.json):
    python diagnose_mace_leak.py --structure ../MOFs/data/.../POSCAR_MOF5_crystalline \\
        --dyn-cls mtknpt --temperature-K 200 --pressure-GPa 1.0 --tdamp-fs 100 --pdamp-fs 1000 \\
        --steps 10000 --log-interval 50 --device cuda:0 --out-csv mtknpt_mem.csv

To A/B against a fixed-cell run (no neighbor-list churn) on the same structure:
    python diagnose_mace_leak.py --structure <same file> --dyn-cls velocity-verlet \\
        --temperature-K 200 --steps 10000 --log-interval 50 --device cuda:0 --out-csv vv_mem.csv
"""
import argparse
import csv
import time

import psutil
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import neighbor_list
from mace.calculators import mace_mp

from cascade.traj_config import InitialTrajConfig, get_dynamics_cls, resolve_dyn_kws


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure', type=str, required=True, help='Path to initial structure, readable by ase.io.read')
    parser.add_argument('--dyn-cls', type=str, default='velocity-verlet', choices=['velocity-verlet', 'npt', 'mtknpt'])
    parser.add_argument('--dt-fs', type=float, default=1.0)
    parser.add_argument('--temperature-K', type=float, default=None, help='Set to seed a Maxwell-Boltzmann velocity distribution')
    parser.add_argument('--pressure-GPa', type=float, default=1.0, help='Only used for npt/mtknpt')
    parser.add_argument('--tdamp-fs', type=float, default=100.0, help='Only used for mtknpt')
    parser.add_argument('--pdamp-fs', type=float, default=1000.0, help='Only used for mtknpt')
    parser.add_argument('--ttime-fs', type=float, default=100.0, help='Only used for npt')
    parser.add_argument('--pfactor-time-fs', type=float, default=1000.0, help='Only used for npt')
    parser.add_argument('--calc-model', type=str, default='medium')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--log-interval', type=int, default=50, help='How often (in steps) to record a memory sample')
    parser.add_argument('--out-csv', type=str, default='diagnose_mace_leak.csv')
    parser.add_argument(
        '--torch-empty-cache', action='store_true',
        help='Call torch.cuda.empty_cache() every log-interval steps, to test whether periodically '
             'releasing the CUDA caching allocator back to the driver mitigates the reserved-memory growth'
    )
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> InitialTrajConfig:
    if args.dyn_cls == 'mtknpt':
        dyn_kws = dict(
            temperature_K=args.temperature_K,
            pressure_GPa=args.pressure_GPa,
            tdamp_fs=args.tdamp_fs,
            pdamp_fs=args.pdamp_fs,
        )
    elif args.dyn_cls == 'npt':
        dyn_kws = dict(
            temperature_K=args.temperature_K,
            ttime_fs=args.ttime_fs,
            externalstress_GPa=args.pressure_GPa,
            pfactor_time_fs=args.pfactor_time_fs,
            pfactor_pressure_GPa=args.pressure_GPa,
        )
    else:
        dyn_kws = {}
    return InitialTrajConfig(
        path=args.structure,
        temperature_K=args.temperature_K,
        dyn_cls=args.dyn_cls,
        dt_fs=args.dt_fs,
        dyn_kws=dyn_kws,
    )


def main():
    args = parse_args()
    cfg = build_cfg(args)

    atoms = read(cfg.path, index=-1)
    if cfg.temperature_K is not None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=cfg.temperature_K)

    calc = mace_mp(model=args.calc_model, device=args.device, default_dtype="float32")
    atoms.calc = calc

    try:
        import torch
        has_cuda = torch.cuda.is_available() and 'cuda' in args.device
    except ImportError:
        torch = None
        has_cuda = False

    proc = psutil.Process()

    dyn_cls = get_dynamics_cls(cfg.dyn_cls)
    dyn = dyn_cls(atoms, **resolve_dyn_kws(cfg))

    csv_file = open(args.out_csv, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        'step', 'wall_time_s', 'host_rss_mb', 'host_available_mb',
        'gpu_allocated_mb', 'gpu_reserved_mb', 'n_atoms', 'n_edges', 'volume_A3',
    ])

    start_time = time.time()

    def log_mem():
        step = dyn.nsteps
        n_edges = len(neighbor_list('i', atoms, cutoff=calc.r_max))
        gpu_alloc = gpu_reserved = 0.0
        if has_cuda:
            if args.torch_empty_cache:
                torch.cuda.empty_cache()
            gpu_alloc = torch.cuda.memory_allocated(args.device) / 1e6
            gpu_reserved = torch.cuda.memory_reserved(args.device) / 1e6
        row = [
            step,
            round(time.time() - start_time, 1),
            round(proc.memory_info().rss / 1e6, 1),
            round(psutil.virtual_memory().available / 1e6, 1),
            round(gpu_alloc, 1),
            round(gpu_reserved, 1),
            len(atoms),
            n_edges,
            round(atoms.get_volume(), 2) if atoms.cell.rank == 3 else None,
        ]
        writer.writerow(row)
        csv_file.flush()
        print(f'step={step} rss_mb={row[2]} avail_mb={row[3]} gpu_alloc_mb={row[4]} gpu_reserved_mb={row[5]} n_edges={n_edges}', flush=True)

    dyn.attach(log_mem, interval=args.log_interval)
    log_mem()  # baseline before stepping
    dyn.run(args.steps)
    log_mem()  # final sample

    csv_file.close()
    print(f'Done. Wrote {args.out_csv}')


if __name__ == '__main__':
    main()
