import argparse
import os
import json
import datetime
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from model import build_model


def get_transforms():
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])


def load_testset(batch_size=256):
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=get_transforms())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader, test_set


def ensemble_eval(model_paths, norm, dropout, base_channels, batch_size, device, out_dir, run_name):
    device = torch.device(device)
    test_loader, test_set = load_testset(batch_size=batch_size)
    n_models = len(model_paths)
    # collect probabilities per model
    probs_list = []
    for i, path in enumerate(model_paths):
        print(f'Loading model {i+1}/{n_models}: {path}')
        model = build_model(num_classes=10, dropout=dropout, norm=norm, device=device, base_channels=base_channels)
        state = torch.load(path, map_location=device)
        # if state is nested (e.g., full checkpoint), try to find 'state_dict'
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
        all_probs = torch.cat(all_probs, dim=0)  # [N, C]
        probs_list.append(all_probs.numpy())

    # average probabilities
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)  # [N, C]
    preds = np.argmax(avg_probs, axis=1)
    targets = np.array(test_set.targets)
    acc = (preds == targets).mean()
    print(f'Ensembled test accuracy: {acc:.4f}')

    os.makedirs(out_dir, exist_ok=True)
    result = {
        'run_name': run_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'model_paths': model_paths,
        'norm': norm,
        'dropout': dropout,
        'base_channels': base_channels,
        'ensemble_size': n_models,
        'test_acc': float(acc),
    }
    out_path = os.path.join(out_dir, f'ensemble_{run_name}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print('Saved results to', out_path)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-paths', nargs='+', required=True, help='paths to model .pt files to ensemble')
    parser.add_argument('--norm', type=str, default='batch', choices=['batch','group','instance','none'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--run-name', type=str, default='ensemble')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ensemble_eval(args.model_paths, args.norm, args.dropout, args.base_channels, args.batch_size, args.device, args.output_dir, args.run_name)
