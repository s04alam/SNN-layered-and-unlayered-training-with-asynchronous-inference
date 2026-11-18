from torch import nn
import torch
import time
import numpy as np

class PositiveSkewCurrentDropout(nn.Module):
    def __init__(self, p=0.5, threshold=-2, slope=3):
        """
        Scales the dropout probability based on the current value of the tensor.
        Probability is skewed towards higher values.

        Args:
            p: dropout probability
            threshold: threshold for the sigmoid function (where its output is 0.5)
            slope: slope of the sigmoid function (will be negated)
        """
        super().__init__()
        self.p = p
        self.threshold = threshold
        if threshold > 0:
            print("WARNING: threshold should be negative for positive skew dropout")
        self.slope = slope

    def forward(self, x: torch.Tensor):
        normalized_probs = torch.sigmoid(-self.slope*(x-self.threshold))
        scaled_probs = normalized_probs * self.p / (normalized_probs.mean() + 1e-6)
        random_mask = torch.rand_like(x) > scaled_probs
        return x * random_mask.float() * 1/(1-self.p)


class NegativeSkewCurrentDropout(nn.Module):
    def __init__(self, p=0.5, threshold=0.3, slope=3):
        """
        Scales the dropout probability based on the current value of the tensor.
        Probability is skewed towards lower values.

        Args:
            p: dropout probability
            threshold: threshold for the sigmoid function (where its output is 0.5)
            slope: slope of the sigmoid function
        """
        super().__init__()
        self.p = p
        self.threshold = threshold
        if threshold < 0:
            print("WARNING: threshold should be positive for negative skew dropout")
        self.slope = slope

    def forward(self, x: torch.Tensor):
        normalized_probs = torch.sigmoid(self.slope*(x-self.threshold))
        scaled_probs = normalized_probs * self.p / (normalized_probs.mean() + 1e-6)
        random_mask = torch.rand_like(x) > scaled_probs
        return x * random_mask.float() * 1/(1-self.p)
    

if __name__ == "__main__":
    torch.random.manual_seed(0)
    x = torch.randn(3, 4)
    dropout = NegativeSkewCurrentDropout(p=0.1)
    start = time.time()
    output = dropout(x)
    
    print((time.time() - start) * 1e3)

    ordinary_dropout = nn.Dropout(p=0.1)
    start = time.time()
    ordinary_dropout(x)
    print((time.time() - start) * 1e3)

    print(x)
    print(output)

    