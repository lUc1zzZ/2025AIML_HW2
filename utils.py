import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np


def train_one_epoch(model, dataloader, optimizer, device, label_smoothing=0.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc='train', leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        # support label smoothing
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        pbar.set_postfix({'loss': total_loss / total, 'acc': correct / total})
    return total_loss / total, correct / total


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def plot_history(history, out_dir, prefix='run'):
    os.makedirs(out_dir, exist_ok=True)
    epochs = len(history['train_loss'])
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), history['train_loss'], label='train')
    plt.plot(range(1, epochs+1), history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(range(1, epochs+1), history['train_acc'], label='train')
    plt.plot(range(1, epochs+1), history['val_acc'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    out_path = os.path.join(out_dir, f'{prefix}_learning_curve.png')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_history_csv(history, out_dir, prefix='run'):
    os.makedirs(out_dir, exist_ok=True)
    import csv
    path = os.path.join(out_dir, f'{prefix}_history.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','val_loss','train_acc','val_acc'])
        for i in range(len(history['train_loss'])):
            writer.writerow([i+1, history['train_loss'][i], history['val_loss'][i], history['train_acc'][i], history['val_acc'][i]])
