"""
Grid search launcher for train.py

Usage:
  python grid_search.py --grid grid.json --kfolds 3 --workers 1 --output-dir outputs

grid.json example:
{
  "dropout": [0.0, 0.2, 0.3],
  "norm": ["batch","group"],
  "lr": [0.01, 0.001],
  "weight_decay": [0.0005],
  "label_smoothing": [0.0, 0.1],
  "base_channels": [32]
}

The script will generate combinations, run `train.py` for each, and write a summary CSV.
"""
import argparse
import json
import os
import itertools
import subprocess
import datetime
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_grid(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def dict_to_runname(d):
    parts = []
    for k in sorted(d.keys()):
        parts.append(f"{k}{str(d[k]).replace(' ', '')}")
    return '_'.join(parts)


def build_cmd(params, common_args):
    cmd = [common_args.python_exe, 'train.py']
    # add k-folds if specified
    if common_args.kfolds and common_args.kfolds > 1:
        cmd += ['--k-folds', str(common_args.kfolds)]
    for k, v in params.items():
        # translate pythonic key to CLI arg
        arg = f'--{k.replace("_","-")}'
        cmd += [arg, str(v)]
    # output dir and run name
    run_name = common_args.run_prefix + '_' + dict_to_runname(params)
    cmd += ['--output-dir', common_args.output_dir, '--run-name', run_name]
    # extra static args
    if common_args.augment:
        cmd.append('--augment')
    if common_args.epochs:
        cmd += ['--epochs', str(common_args.epochs)]
    if common_args.batch_size:
        cmd += ['--batch-size', str(common_args.batch_size)]
    if common_args.device:
        cmd += ['--device', common_args.device]
    return cmd, run_name


def run_job(cmd, workdir):
    started = datetime.datetime.now().isoformat()
    proc = subprocess.run(cmd, cwd=workdir)
    finished = datetime.datetime.now().isoformat()
    return proc.returncode, started, finished


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=str, required=True, help='path to grid json')
    parser.add_argument('--kfolds', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1, help='number of parallel workers')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--run-prefix', dest='run_prefix', type=str, default='grid')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--python-exe', dest='python_exe', type=str, default='python')
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()

    grid = load_grid(args.grid)
    keys = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f'Launching {len(combos)} runs (workers={args.workers})')

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f'grid_summary_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

    jobs = []
    for combo in combos:
        params = {k: v for k, v in zip(keys, combo)}
        cmd, run_name = build_cmd(params, args)
        jobs.append((run_name, cmd))

    # run jobs
    results = []
    workdir = os.getcwd()
    if args.workers <= 1:
        for run_name, cmd in jobs:
            print('Running', run_name)
            rc, started, finished = run_job(cmd, workdir)
            results.append((run_name, ' '.join(cmd), rc, started, finished))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_map = {}
            for run_name, cmd in jobs:
                print('Submitting', run_name)
                future = ex.submit(run_job, cmd, workdir)
                future_map[future] = (run_name, cmd)
            for fut in as_completed(future_map):
                run_name, cmd = future_map[fut]
                try:
                    rc, started, finished = fut.result()
                except Exception as e:
                    rc = -1
                    started = ''
                    finished = ''
                results.append((run_name, ' '.join(cmd), rc, started, finished))

    # write summary CSV
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['run_name', 'cmd', 'returncode', 'started', 'finished'])
        for row in results:
            writer.writerow(row)

    print('Grid search finished. Summary saved to', summary_path)


if __name__ == '__main__':
    main()
