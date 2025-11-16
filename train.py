import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import json
import datetime
from sklearn.model_selection import StratifiedKFold
from model import build_model
from utils import train_one_epoch, evaluate, plot_history, save_history_csv
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(augment=False):
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])


def run_training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    # prepare separate datasets: one with augmentation for training, one without for validation
    train_transform_aug = get_transforms(augment=args.augment)
    train_transform_noaug = get_transforms(False)
    # download once (first call), the second uses download=False to avoid re-downloading
    train_dataset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform_aug)
    train_dataset_noaug = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform_noaug)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=get_transforms(False))

    X = np.arange(len(train_dataset_aug))
    y = np.array(train_dataset_aug.targets)

    # create a run-specific directory to store outputs and configs
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name if hasattr(args, 'run_name') and args.run_name else f'run_{ts}'
    out_dir = os.path.join(args.output_dir, run_name + '_' + ts)
    os.makedirs(out_dir, exist_ok=True)
    # save args
    with open(os.path.join(out_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.k_folds > 1:
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
        fold = 0
        fold_results = []
        for train_idx, val_idx in skf.split(X, y):
            fold += 1
            print(f'Fold {fold}/{args.k_folds}')
            # use augmented dataset for training subset, non-augmented for validation subset
            train_subset = Subset(train_dataset_aug, train_idx)
            val_subset = Subset(train_dataset_noaug, val_idx)
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

            model = build_model(num_classes=10, dropout=args.dropout, norm=args.norm, device=device, base_channels=args.base_channels)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

            history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
            best_val_acc = 0.0
            fold_dir = os.path.join(out_dir, f'fold{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            best_path = os.path.join(fold_dir, f'best_fold{fold}.pt')
            for epoch in range(1, args.epochs+1):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, label_smoothing=args.label_smoothing)
                val_loss, val_acc = evaluate(model, val_loader, device)
                scheduler.step()
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                print(f'Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_path)
            print(f'Fold {fold} best val acc: {best_val_acc:.4f}')
            plot_history(history, fold_dir, prefix=f'fold{fold}')
            save_history_csv(history, fold_dir, prefix=f'fold{fold}')
            fold_results.append(best_val_acc)

        print('Cross-validation results:', fold_results, 'mean:', np.mean(fold_results))
        # save summary
        with open(os.path.join(out_dir, 'cv_results.json'), 'w') as f:
            json.dump({'fold_results': fold_results, 'mean': float(np.mean(fold_results))}, f, indent=2)

    else:
        # standard train/val split
        # manual split to ensure train uses augmented dataset and val uses non-augmented
        dataset_size = len(train_dataset_aug)
        val_size = int(dataset_size * args.val_split)
        train_size = dataset_size - val_size
        # reproducible shuffle
        idxs = np.arange(dataset_size)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(idxs)
        train_idx = idxs[:train_size].tolist()
        val_idx = idxs[train_size:].tolist()
        train_subset = Subset(train_dataset_aug, train_idx)
        val_subset = Subset(train_dataset_noaug, val_idx)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = build_model(num_classes=10, dropout=args.dropout, norm=args.norm, device=device, base_channels=args.base_channels)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
        best_val_acc = 0.0
        best_path = os.path.join(out_dir, f'best.pt')
        for epoch in range(1, args.epochs+1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, label_smoothing=args.label_smoothing)
            val_loss, val_acc = evaluate(model, val_loader, device)
            scheduler.step()
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            print(f'Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_path)

        print(f'Best val acc: {best_val_acc:.4f}')
        plot_history(history, out_dir, prefix='run')
        save_history_csv(history, out_dir, prefix='run')

        # evaluate on test set
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f'Test loss: {test_loss:.4f} test acc: {test_acc:.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--norm', type=str, default='batch', choices=['batch','group','instance','none'])
    parser.add_argument('--k-folds', type=int, default=1)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--run-name', type=str, default='')
    parser.add_argument('--base-channels', type=int, default=32, help='base number of channels (wider/deeper model)')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing value for cross entropy')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_training(args)
