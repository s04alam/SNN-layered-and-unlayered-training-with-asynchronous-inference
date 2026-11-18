import torch
import torch.nn as nn
from aiexperiments.networks.components.matrix.early_accumulate_linear import EarlyAccumulateLinear

def test_EarlyAccumulateLinear_threshold():
    torch.random.manual_seed(0)
    for i in range(100):
        inp_spikes = torch.ones(1, 3, dtype=torch.float32)
        standard_linear = nn.Linear(3, 4, bias=False)
        early_spiking_linear = EarlyAccumulateLinear(3, 4, threshold=0.1)
        standard_linear.weight = early_spiking_linear.weight

        out_standard = standard_linear(inp_spikes)
        out = early_spiking_linear(inp_spikes, torch.zeros(1, 4))
        below_threshold_standard = out_standard[out_standard < 0.1]
        below_threshold = out[out < 0.1]
        assert below_threshold_standard.shape[0] >= below_threshold.shape[0]

def test_EarlyAccumulateLinear_no_threshold():
    torch.random.manual_seed(0)
    for i in range(100):
        inp_spikes = torch.ones(1, 3, dtype=torch.float32)
        standard_linear = nn.Linear(3, 4, bias=False)
        early_spiking_linear = EarlyAccumulateLinear(3, 4, threshold=1000)
        standard_linear.weight = early_spiking_linear.weight

        out_standard = standard_linear(inp_spikes)
        out = early_spiking_linear(inp_spikes, torch.zeros(1, 4))
        assert torch.allclose(out_standard, out)

def test_EarlyAccumulateLinear_stats_Kmin():
    inp_spikes = torch.asarray([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32) # batch_size, num_neurons_in
    linear = EarlyAccumulateLinear(2, 1, threshold=0.1, K=1) # num_neurons_in, num_neurons_out
    linear.weight = nn.Parameter(torch.tensor([[-1.0, 1.0]], dtype=torch.float32))
    outs = []
    for i in range(1000):
        out = linear(inp_spikes, torch.zeros(4, 1))
        outs.append(out)
    outs = torch.cat(outs, dim=1)
    outs_stats = torch.mean(outs, dim=1)
    assert torch.allclose(outs_stats, torch.tensor([0.0, 1.0, -1.0, 0.5]), atol=0.2)

def test_EarlyAccumulateLinear_stats_Kfull():
    inp_spikes = torch.asarray([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32) # batch_size, num_neurons_in
    linear = EarlyAccumulateLinear(2, 1, threshold=0.1, K=2) # num_neurons_in, num_neurons_out
    linear.weight = nn.Parameter(torch.tensor([[-1.0, 1.0]], dtype=torch.float32))
    outs = []
    for i in range(1000):
        out = linear(inp_spikes, torch.zeros(4, 1))
        outs.append(out)
    outs = torch.cat(outs, dim=1)
    outs_stats = torch.mean(outs, dim=1)
    assert torch.allclose(outs_stats, torch.tensor([0.0, 1.0, -1.0, 0.0]), atol=0.2)