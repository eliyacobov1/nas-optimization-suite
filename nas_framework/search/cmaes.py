import numpy as np
from ..utils import BLOCK_CHOICES
from ..train_eval import train_and_eval


def cma_es_search(train_loader, val_loader, device, num_blocks=3, iterations=3):
    mean = np.zeros(num_blocks)
    sigma = 1.0
    best_arch = None
    best_score = 0
    for _ in range(iterations):
        samples = np.random.randn(4, num_blocks) * sigma + mean
        samples = np.clip(samples, 0, len(BLOCK_CHOICES)-1)
        for s in samples:
            arch = [BLOCK_CHOICES[int(round(v))] for v in s]
            score, _ = train_and_eval(arch, train_loader, val_loader, device)
            if score > best_score:
                best_score = score
                best_arch = arch
        mean = samples.mean(axis=0)
        sigma *= 0.9
    return best_arch, best_score
