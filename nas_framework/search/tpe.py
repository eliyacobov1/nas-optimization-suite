import random
from collections import defaultdict
from ..utils import random_arch, BLOCK_CHOICES
from ..train_eval import train_and_eval


def tpe_search(train_loader, val_loader, device, trials=5):
    observations = []
    for _ in range(trials):
        if len(observations) < 2:
            arch = random_arch()
        else:
            good = [a for a, s in observations if s > 0.5 * max([sc for _, sc in observations])]
            probs = defaultdict(int)
            for a in good:
                for i, b in enumerate(a):
                    probs[(i, b)] += 1
            arch = []
            for i in range(len(good[0])):
                choices = [c for c in BLOCK_CHOICES]
                weights = [probs.get((i, c), 1) for c in choices]
                total = sum(weights)
                r = random.random() * total
                cum = 0
                for c, w in zip(choices, weights):
                    cum += w
                    if r <= cum:
                        arch.append(c)
                        break
        score, _ = train_and_eval(arch, train_loader, val_loader, device)
        observations.append((arch, score))
    best_arch, best_score = max(observations, key=lambda x: x[1])
    return best_arch, best_score
