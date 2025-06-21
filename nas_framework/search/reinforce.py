import torch
import torch.nn as nn
from ..utils import BLOCK_CHOICES
from ..train_eval import train_and_eval

class Policy(nn.Module):
    def __init__(self, num_blocks=3, num_ops=3):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_blocks, num_ops))

    def sample(self):
        probs = torch.softmax(self.logits, dim=1)
        m = torch.distributions.Categorical(probs)
        actions = m.sample()
        log_probs = m.log_prob(actions)
        return actions.tolist(), log_probs.sum()


def reinforce_search(train_loader, val_loader, device, num_blocks=3, episodes=5, lr=0.1):
    policy = Policy(num_blocks, len(BLOCK_CHOICES)).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    best_arch = None
    best_score = 0
    for _ in range(episodes):
        arch, log_prob = policy.sample()
        arch = [BLOCK_CHOICES[a] for a in arch]
        score, _ = train_and_eval(arch, train_loader, val_loader, device)
        loss = -log_prob * score
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if score > best_score:
            best_score = score
            best_arch = arch
    return best_arch, best_score
