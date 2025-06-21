import random

BLOCK_CHOICES = [0, 1, 2]  # Conv, Transformer, Pool


def random_arch(num_blocks=3):
    return [random.choice(BLOCK_CHOICES) for _ in range(num_blocks)]


def mutate_arch(arch, mutation_rate=0.1):
    new_arch = arch[:]
    for i in range(len(new_arch)):
        if random.random() < mutation_rate:
            new_arch[i] = random.choice(BLOCK_CHOICES)
    return new_arch
