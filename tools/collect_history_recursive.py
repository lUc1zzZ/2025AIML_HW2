import os, glob, csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, 'outputs')
plots = os.path.join(OUT, 'plots')
os.makedirs(plots, exist_ok=True)

runs = []
# find all run_history.csv or history.csv files recursively
hist_files = glob.glob(os.path.join(OUT, '**', 'run_history.csv'), recursive=True) + glob.glob(os.path.join(OUT, '**', 'history.csv'), recursive=True)
seen_dirs = set()
for hf in sorted(hist_files):
    run_dir = os.path.dirname(hf)
    if run_dir in seen_dirs:
        continue
    seen_dirs.add(run_dir)
    run_name = os.path.basename(run_dir)
    info = {'run_name': run_name, 'run_dir': run_dir}
    # try args.json
    argsf = os.path.join(run_dir, 'args.json')
    if os.path.exists(argsf):
        try:
            args = json.load(open(argsf,'r',encoding='utf-8'))
            for k in ['dropout','norm','lr','weight_decay','label_smoothing','base_channels','seed','run_name']:
                if k in args:
                    info[k]=args[k]
        except:
            pass
    # read history csv
    try:
        with open(hf,newline='',encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        epochs = [int(r.get('epoch', i+1)) for i,r in enumerate(rows)]
        def col(name):
            vals=[]
            for r in rows:
                v = r.get(name,'')
                try:
                    vals.append(float(v))
                except:
                    vals.append(None)
            return vals
        train_acc = col('train_acc')
        val_acc = col('val_acc')
        train_loss = col('train_loss')
        val_loss = col('val_loss')
        info['best_val_acc'] = None
        info['epoch_best'] = None
        for i,v in enumerate(val_acc):
            if v is None: continue
            if info['best_val_acc'] is None or v>info['best_val_acc']:
                info['best_val_acc']=v
                info['epoch_best']=epochs[i]
        info['final_train_acc']=train_acc[-1] if train_acc else None
        info['final_val_acc']=val_acc[-1] if val_acc else None
        # save plots
        try:
            plt.figure(figsize=(8,4))
            if any(t is not None for t in train_acc):
                plt.plot(epochs, [t if t is not None else float('nan') for t in train_acc], label='train_acc')
            if any(v is not None for v in val_acc):
                plt.plot(epochs, [v if v is not None else float('nan') for v in val_acc], label='val_acc')
            plt.xlabel('epoch'); plt.ylabel('acc'); plt.title(run_name)
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(plots, f"{run_name}_acc.png"))
            plt.close()
        except:
            pass
    except Exception as e:
        info['history_error']=str(e)
    # find test acc in any jsons under run_dir
    info['test_acc']=None
    for jf in glob.glob(os.path.join(run_dir,'*.json')):
        try:
            j=json.load(open(jf,'r',encoding='utf-8'))
            if isinstance(j,dict) and 'test_acc' in j:
                info['test_acc']=j['test_acc']
        except:
            continue
    runs.append(info)

# also include runs that have json cv_results but no run_history
for jf in glob.glob(os.path.join(OUT,'**','cv_results.json'), recursive=True):
    rd = os.path.dirname(jf)
    rn = os.path.basename(rd)
    if any(r['run_dir']==rd for r in runs):
        continue
    try:
        j=json.load(open(jf,'r',encoding='utf-8'))
        info={'run_name':rn,'run_dir':rd}
        if isinstance(j,dict):
            # try to extract mean_val_acc
            if 'mean_val_acc' in j:
                info['best_val_acc']=j['mean_val_acc']
            # copy args if present
            if 'args' in j and isinstance(j['args'],dict):
                for k in ['dropout','norm','lr','weight_decay','label_smoothing','base_channels','seed']:
                    if k in j['args']:
                        info[k]=j['args'][k]
        runs.append(info)
    except:
        pass

# write summary
keys = ['run_name','dropout','norm','lr','weight_decay','label_smoothing','base_channels','seed','best_val_acc','epoch_best','final_train_acc','final_val_acc','test_acc']
outf = os.path.join(OUT,'experiment_summary_full.csv')
with open(outf,'w',newline='',encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for r in runs:
        writer.writerow({k: r.get(k) for k in keys})
print(f'Wrote {outf} with {len(runs)} entries. Plots in {plots}')
