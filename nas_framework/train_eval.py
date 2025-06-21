import time
import torch
import torch.nn as nn
from .model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    return correct / len(loader.dataset)


def train_and_eval(block_ids, train_loader, val_loader, device, epochs=5, patience=2):
    model = build_model(block_ids).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    best_acc = 0
    epochs_no_improve = 0
    start = time.time()
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    runtime = time.time() - start
    return best_acc, runtime
