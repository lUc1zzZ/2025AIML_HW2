"""
Auto ensemble experiment

This script trains N models (by varying random seed or suffix) using `train.py`
and then runs `ensemble_eval.py` to evaluate their averaged softmax predictions on the test set.

Example usage:
  python auto_ensemble.py --n-models 3 --base-run-name ensemble_m --epochs 20 --batch-size 128 --lr 0.01 --dropout 0.3 --norm batch --output-dir outputs --device cpu

It will run three trainings with seeds [1,2,3], saving runs as `ensemble_m_seed1_*`, etc.,
then locate their `best.pt` files and call `ensemble_eval.py` with those checkpoints.

Notes:
- The script assumes you train with `--k-folds 1` (non-cross-validation) so `train.py` saves a `best.pt` in the run directory.
- If you use k-fold in trainings, the script can be adapted to pick fold-best checkpoints instead.
"""
import argparse
import subprocess
import os
import glob
import time
import json
import datetime


def run_cmd(cmd):
    print('>>>', ' '.join(cmd))
    rc = subprocess.run(cmd)
    if rc.returncode != 0:
        raise RuntimeError(f'Command failed: {cmd}')


def latest_run_dir(output_dir, run_name_prefix):
    # find directories under output_dir that start with run_name_prefix
    candidates = []
    for name in os.listdir(output_dir):
        if name.startswith(run_name_prefix):
            candidates.append((os.path.getmtime(os.path.join(output_dir, name)), os.path.join(output_dir, name)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def find_best_pt(run_dir):
    # try directly best.pt
    p = os.path.join(run_dir, 'best.pt')
    if os.path.isfile(p):
        return p
    # else look for fold subdirs with best_*.pt
    pts = glob.glob(os.path.join(run_dir, '**', 'best*.pt'), recursive=True)
    if pts:
        # choose newest
        pts.sort(key=os.path.getmtime, reverse=True)
        return pts[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-models', type=int, default=3)
    parser.add_argument('--seeds', type=int, nargs='*', default=None, help='explicit seeds to use; overrides n-models if provided')
    parser.add_argument('--base-run-name', type=str, default='ensemble_m')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--python-exe', type=str, default='python')
    parser.add_argument('--device', type=str, default='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
    # pass-through args to train.py
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--norm', type=str, default='batch')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--k-folds', type=int, default=1, help='recommend using 1 for ensemble training')
    args = parser.parse_args()

    # determine seeds
    if args.seeds and len(args.seeds) > 0:
        seeds = args.seeds
    else:
        seeds = list(range(1, args.n_models+1))

    trained_paths = []

    for s in seeds:
        run_name = f"{args.base_run_name}_seed{s}"
        cmd = [args.python_exe, 'train.py', '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--lr', str(args.lr), '--dropout', str(args.dropout), '--norm', args.norm, '--output-dir', args.output_dir, '--run-name', run_name, '--seed', str(s), '--base-channels', str(args.base_channels), '--weight-decay', str(args.weight_decay), '--label-smoothing', str(args.label_smoothing), '--k-folds', str(args.k_folds)]
        if args.augment:
            cmd.append('--augment')
        # run training
        run_cmd(cmd)
        # wait briefly to ensure run dir created
        time.sleep(1)
        run_dir = latest_run_dir(args.output_dir, run_name)
        if run_dir is None:
            raise RuntimeError(f'Could not find run dir for {run_name} in {args.output_dir}')
        best_pt = find_best_pt(run_dir)
        if best_pt is None:
            raise RuntimeError(f'Could not find best.pt in run dir {run_dir}')
        print('Found checkpoint:', best_pt)
        trained_paths.append(best_pt)

    # perform ensemble evaluation
    ensemble_name = f"{args.base_run_name}_ensemble_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_cmd = [args.python_exe, 'ensemble_eval.py', '--model-paths'] + trained_paths + ['--norm', args.norm, '--dropout', str(args.dropout), '--base-channels', str(args.base_channels), '--output-dir', args.output_dir, '--run-name', ensemble_name]
    if args.device:
        eval_cmd += ['--device', args.device]
    run_cmd(eval_cmd)

    print('Ensemble evaluation complete. Results saved under', args.output_dir)


if __name__ == '__main__':
    main()
