import torch
from torch import nn

class SpikeDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        else:
            mask = torch.empty(x.shape, device=x.device).bernoulli_(1 - self.p)
            return x * mask