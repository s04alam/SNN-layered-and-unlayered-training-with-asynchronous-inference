
from aiexperiments.networks.components.matrix.leaky import Leaky
import torch


def test_leaky():
    leaky = Leaky(beta=1, threshold=1, output=True)
    spk, mem = leaky(torch.tensor([10.0, 10.0, 0.5]))
    assert spk.sum() == 2
    assert mem.sum() == 0.5