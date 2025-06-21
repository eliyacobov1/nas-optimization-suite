import random
from typing import List

from ..train_eval import train_and_eval
from ..utils import random_arch, BLOCK_CHOICES

class Particle:
    def __init__(self, position: List[int]):
        self.position = position
        self.best_position = position[:]
        self.velocity = [0] * len(position)
        self.best_score = 0


def pso_search(train_loader, val_loader, device, num_particles=3, iterations=3):
    particles = [Particle(random_arch()) for _ in range(num_particles)]
    global_best = particles[0].position
    global_best_score = 0

    for _ in range(iterations):
        for p in particles:
            score, _ = train_and_eval(p.position, train_loader, val_loader, device)
            if score > p.best_score:
                p.best_score = score
                p.best_position = p.position[:]
            if score > global_best_score:
                global_best_score = score
                global_best = p.position[:]
        for p in particles:
            for i in range(len(p.position)):
                vel = random.random() * (p.best_position[i] - p.position[i]) + \
                      random.random() * (BLOCK_CHOICES.index(global_best[i]) - BLOCK_CHOICES.index(p.position[i]))
                p.velocity[i] = vel
                new_val = BLOCK_CHOICES[(BLOCK_CHOICES.index(p.position[i]) + int(round(vel))) % len(BLOCK_CHOICES)]
                p.position[i] = new_val
    return global_best, global_best_score
