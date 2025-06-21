import torch
from nas_framework.dataset import get_dataloaders
from nas_framework.search import pso, ga, tpe, cmaes, darts, reinforce


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_dataloaders()
    results = {}

    arch, score = pso.pso_search(train_loader, val_loader, device)
    results['PSO'] = (arch, score)

    arch, score = ga.ga_search(train_loader, val_loader, device)
    results['GA'] = (arch, score)

    arch, score = tpe.tpe_search(train_loader, val_loader, device)
    results['TPE'] = (arch, score)

    arch, score = cmaes.cma_es_search(train_loader, val_loader, device)
    results['CMA-ES'] = (arch, score)

    arch, score = darts.darts_search(train_loader, val_loader, device)
    results['DARTS'] = (arch, score)

    arch, score = reinforce.reinforce_search(train_loader, val_loader, device)
    results['REINFORCE'] = (arch, score)

    for method, (arch, score) in results.items():
        print(f'{method}: best_arch={arch} score={score:.4f}')


if __name__ == '__main__':
    main()
