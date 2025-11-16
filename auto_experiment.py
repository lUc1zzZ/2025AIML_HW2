"""
Auto experiment runner

Usage:
  python auto_experiment.py --grid grid.json --kfolds 3 --workers 1 --output-dir outputs \
      --epochs-grid 20 --epochs-final 40 --batch-size 128 --augment

What it does:
 1. Calls `grid_search.py` to run k-fold CV over hyperparameter grid.
 2. Calls `summarize_cv.py` to build a CSV summary.
 3. Finds the run with best CV mean validation accuracy (from `cv_results.json`).
 4. Loads that run's `args.json` and retrains `train.py` on the full training set with those hyperparameters.

Notes:
- Runs are executed in the current working directory; ensure scripts `grid_search.py`, `summarize_cv.py`, and `train.py` are present.
- This script does not attempt to parallelize across GPUs; if you need GPU assignment, run grid_search with proper GPU-aware settings or ask me to add GPU allocation support.
"""
import argparse
import subprocess
import os
import json
import sys
import datetime


def run(cmd, cwd=None):
    print('>>>', ' '.join(cmd))
    rc = subprocess.run(cmd, cwd=cwd)
    if rc.returncode != 0:
        raise RuntimeError(f'Command failed (rc={rc.returncode}): {cmd}')


def find_best_run(outputs_dir):
    best = None
    for name in os.listdir(outputs_dir):
        run_dir = os.path.join(outputs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        cvf = os.path.join(run_dir, 'cv_results.json')
        if not os.path.isfile(cvf):
            continue
        try:
            with open(cvf, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        mean = data.get('mean')
        if mean is None:
            continue
        if best is None or mean > best[0]:
            best = (mean, name, data)
    return best


def load_args_json(run_dir):
    p = os.path.join(run_dir, 'args.json')
    if not os.path.isfile(p):
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_train_cmd(python_exe, args_dict, epochs, out_dir, run_name, augment_override=None):
    cmd = [python_exe, 'train.py']
    skip = {'output_dir','run_name','k_folds','device'}
    for k, v in args_dict.items():
        if k in skip:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f'--{k.replace("_","-")}')
            continue
        cmd += [f'--{k.replace("_","-")}', str(v)]
    cmd += ['--epochs', str(epochs), '--output-dir', out_dir, '--run-name', run_name]
    if augment_override:
        cmd.append('--augment')
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', required=True, help='path to grid json')
    parser.add_argument('--kfolds', type=int, default=3)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--epochs-grid', type=int, default=20)
    parser.add_argument('--epochs-final', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--python-exe', type=str, default='python')
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()

    # 1) run grid_search
    gs_cmd = [args.python_exe, 'grid_search.py', '--grid', args.grid, '--kfolds', str(args.kfolds), '--workers', str(args.workers), '--output-dir', args.output_dir, '--run-prefix', 'auto']
    gs_cmd += ['--epochs', str(args.epochs_grid), '--batch-size', str(args.batch_size)]
    if args.augment:
        gs_cmd.append('--augment')
    run(gs_cmd)

    # 2) summarize CV
    sv_cmd = [args.python_exe, 'summarize_cv.py', '--outputs', args.output_dir, '--out-csv', os.path.join(args.output_dir, 'cv_summary.csv')]
    run(sv_cmd)

    # 3) find best run
    best = find_best_run(args.output_dir)
    if best is None:
        print('No CV results found in', args.output_dir)
        sys.exit(1)
    mean, run_name, data = best
    print(f'Best run: {run_name} mean={mean}')
    run_dir = os.path.join(args.output_dir, run_name)

    # 4) load args.json from best run and retrain on full training set
    args_json = load_args_json(run_dir)
    if args_json is None:
        print('args.json not found in', run_dir)
        sys.exit(1)

    final_run_name = f'final_{run_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    train_cmd = build_train_cmd(args.python_exe, args_json, args.epochs_final, args.output_dir, final_run_name, augment_override=args.augment)
    # ensure k-folds removed
    if '--k-folds' in train_cmd:
        i = train_cmd.index('--k-folds')
        del train_cmd[i:i+2]
    run(train_cmd)
    print('Done. Final run:', final_run_name)


if __name__ == '__main__':
    main()
