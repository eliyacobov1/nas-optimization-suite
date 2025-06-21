# nas-optimization-suite

This repository provides a simple neural architecture search (NAS) framework
that explores CNN and Transformer blocks for image classification. It includes
implementations of several optimization strategies from scratch:

- Particle Swarm Optimization (PSO)
- Genetic Algorithm (GA)
- Tree-structured Parzen Estimator (TPE)
- CMA-ES
- DARTS
- REINFORCE

The search space is defined as a sequence of block IDs where each block can be a
convolutional block, transformer block, or pooling block. Models are built using
PyTorch and trained on an image classification dataset for a few epochs with
simple early stopping.

A small fallback dataset (CIFAR-100) is used when the ImageNet16-120 subset is
not available. This keeps the example lightweight so the search procedures can
run in constrained environments. The training routine is short (5 epochs) and
intended for demonstration rather than high accuracy.

Run the search with:

```bash
python main.py
```

Each algorithm prints the best architecture it discovered along with the
validation accuracy obtained during the search.
