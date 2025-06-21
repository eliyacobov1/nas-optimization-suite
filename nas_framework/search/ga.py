import random
from ..utils import random_arch, mutate_arch
from ..train_eval import train_and_eval


def crossover(a, b):
    point = random.randint(1, len(a)-1)
    return a[:point] + b[point:]


def ga_search(train_loader, val_loader, device, pop_size=4, generations=3):
    population = [random_arch() for _ in range(pop_size)]
    scores = [0] * pop_size
    for g in range(generations):
        for i, arch in enumerate(population):
            scores[i], _ = train_and_eval(arch, train_loader, val_loader, device)
        ranked = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        population = [ranked[0][1], ranked[1][1]]
        while len(population) < pop_size:
            parent1, parent2 = random.sample(ranked[:2], 2)
            child = crossover(parent1[1], parent2[1])
            child = mutate_arch(child)
            population.append(child)
    best_score, best_arch = max(zip(scores, population), key=lambda x: x[0])
    return best_arch, best_score
