import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    try:
        dataset = torchvision.datasets.ImageFolder('data/imagenet16-120/train', transform=transform)
        val_dataset = torchvision.datasets.ImageFolder('data/imagenet16-120/val', transform=transform)
    except Exception:
        # Fallback to CIFAR100 if ImageNet subset is unavailable
        dataset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=transform)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader
