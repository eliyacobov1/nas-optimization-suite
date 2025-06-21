import torch
import torch.nn as nn
from ..model import ConvBlock, TransformerBlock, PoolBlock
from ..train_eval import train_one_epoch, evaluate

class DARTSCell(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ops = nn.ModuleList([
            ConvBlock(in_channels),
            TransformerBlock(in_channels),
            PoolBlock()
        ])
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))

    def forward(self, x):
        weights = torch.softmax(self.alpha, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class DARTSNetwork(nn.Module):
    def __init__(self, num_blocks=3, in_channels=3, num_classes=100):
        super().__init__()
        self.cells = nn.ModuleList([DARTSCell(in_channels if i==0 else 64) for i in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        for cell in self.cells:
            x = cell(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def darts_search(train_loader, val_loader, device, num_blocks=3, epochs=5):
    model = DARTSNetwork(num_blocks=num_blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
    arch = [cell.alpha.argmax().item() for cell in model.cells]
    return arch, best_acc
