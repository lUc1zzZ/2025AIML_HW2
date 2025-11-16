"""
Summarize cross-validation runs under `outputs/`.

It finds run directories containing `cv_results.json` and `args.json`,
and writes `cv_summary.csv` with one row per run, including flattened args.

Usage:
  python summarize_cv.py --outputs outputs --out-csv outputs/cv_summary.csv
"""
import os
import json
import csv
import argparse


def flatten_args(d):
    # flatten nested args dict to string values for CSV
    flat = {}
    for k, v in d.items():
        try:
            flat[k] = json.dumps(v, ensure_ascii=False)
        except Exception:
            flat[k] = str(v)
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=str, default='outputs')
    parser.add_argument('--out-csv', type=str, default='outputs/cv_summary.csv')
    args = parser.parse_args()

    rows = []
    header_keys = set(['run_dir','mean_val_acc','fold_results'])

    for d in sorted(os.listdir(args.outputs)):
        run_dir = os.path.join(args.outputs, d)
        if not os.path.isdir(run_dir):
            continue
        cv_file = os.path.join(run_dir, 'cv_results.json')
        args_file = os.path.join(run_dir, 'args.json')
        if not os.path.isfile(cv_file):
            # skip runs without cv results
            continue
        with open(cv_file, 'r', encoding='utf-8') as f:
            cv = json.load(f)
        args_dict = {}
        if os.path.isfile(args_file):
            with open(args_file, 'r', encoding='utf-8') as f:
                args_dict = json.load(f)
        flat = flatten_args(args_dict)
        row = {'run_dir': d, 'mean_val_acc': cv.get('mean', None), 'fold_results': json.dumps(cv.get('fold_results', []), ensure_ascii=False)}
        row.update(flat)
        header_keys.update(row.keys())
        rows.append(row)

    header = ['run_dir','mean_val_acc','fold_results'] + sorted([k for k in header_keys if k not in ('run_dir','mean_val_acc','fold_results')])

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (x.get('mean_val_acc') is None, -(x.get('mean_val_acc') or 0))):
            writer.writerow({k: r.get(k, '') for k in header})

    print('Wrote summary to', args.out_csv)


if __name__ == '__main__':
    main()
