import os
import glob
import json
import csv
import math
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)
plots_dir = os.path.join(OUT, 'plots')
os.makedirs(plots_dir, exist_ok=True)

runs = []
for entry in sorted(os.listdir(OUT)):
    run_dir = os.path.join(OUT, entry)
    if not os.path.isdir(run_dir):
        continue
    # skip plots folder
    if entry == 'plots':
        continue
    info = {'run_name': entry, 'run_dir': run_dir}
    # args.json
    argsf = os.path.join(run_dir, 'args.json')
    if os.path.exists(argsf):
        try:
            args = json.load(open(argsf, 'r', encoding='utf-8'))
            # copy some commonly used fields
            for k in ['dropout','norm','lr','weight_decay','label_smoothing','base_channels','seed','run_name']:
                if k in args:
                    info[k] = args[k]
        except Exception as e:
            info['args_error'] = str(e)
    # history.csv
    histf = os.path.join(run_dir, 'history.csv')
    history = None
    if os.path.exists(histf):
        try:
            with open(histf, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                # try to extract numeric columns if exist
                epochs = [int(r.get('epoch', i+1)) for i,r in enumerate(rows)]
                def colval(name):
                    vals = []
                    for r in rows:
                        v = r.get(name)
                        if v is None or v == '':
                            vals.append(None)
                        else:
                            try:
                                vals.append(float(v))
                            except:
                                vals.append(None)
                    return vals
                train_acc = colval('train_acc')
                val_acc = colval('val_acc')
                train_loss = colval('train_loss')
                val_loss = colval('val_loss')
                history = {
                    'epoch': epochs,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                # best val acc
                best_val = None
                best_epoch = None
                for i,va in enumerate(val_acc):
                    if va is None:
                        continue
                    if best_val is None or va > best_val:
                        best_val = va
                        best_epoch = epochs[i]
                info['best_val_acc'] = best_val
                info['epoch_best'] = best_epoch
                info['final_train_acc'] = train_acc[-1] if train_acc else None
                info['final_val_acc'] = val_acc[-1] if val_acc else None
                # save plots
                try:
                    plt.figure(figsize=(8,4))
                    if any(v is not None for v in train_acc):
                        plt.plot(epochs, [v if v is not None else float('nan') for v in train_acc], label='train_acc')
                    if any(v is not None for v in val_acc):
                        plt.plot(epochs, [v if v is not None else float('nan') for v in val_acc], label='val_acc')
                    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.title(entry + ' acc')
                    plt.legend(); plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"{entry}_acc.png"))
                    plt.close()
                except Exception:
                    pass
                try:
                    plt.figure(figsize=(8,4))
                    if any(v is not None for v in train_loss):
                        plt.plot(epochs, [v if v is not None else float('nan') for v in train_loss], label='train_loss')
                    if any(v is not None for v in val_loss):
                        plt.plot(epochs, [v if v is not None else float('nan') for v in val_loss], label='val_loss')
                    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title(entry + ' loss')
                    plt.legend(); plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f"{entry}_loss.png"))
                    plt.close()
                except Exception:
                    pass
        except Exception as e:
            info['history_error'] = str(e)
    # try find any json with test_acc
    info['test_acc'] = None
    for jfile in glob.glob(os.path.join(run_dir, '*.json')):
        try:
            j = json.load(open(jfile, 'r', encoding='utf-8'))
            if isinstance(j, dict) and 'test_acc' in j:
                info['test_acc'] = j['test_acc']
                # capture more fields
                for k in ['timestamp','ensemble_size']:
                    if k in j:
                        info[k] = j[k]
        except:
            continue
    runs.append(info)

# write CSV
keys = ['run_name','dropout','norm','lr','weight_decay','label_smoothing','base_channels','seed','best_val_acc','epoch_best','final_train_acc','final_val_acc','test_acc','timestamp']
csvf = os.path.join(OUT, 'experiment_summary.csv')
with open(csvf, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for r in runs:
        row = {k: r.get(k) for k in keys}
        writer.writerow(row)

print(f"Wrote {csvf} with {len(runs)} runs. Plots saved to {plots_dir}.")
