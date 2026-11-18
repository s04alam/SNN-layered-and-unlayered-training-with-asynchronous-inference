import torch
from aiexperiments.networks.components.matrix.variable_leaky import VariableStepLeaky


def test_time_steps():
    neuron = VariableStepLeaky(tau_m=20, output=True)
    # setup a test
    batch_size = 16
    n_events = 100
    x = torch.randint(0, 1, (n_events, batch_size, 34 * 34), dtype=float)
    dt = torch.arange(0, batch_size, 1)
    # run the test
    for i in range(n_events):
        x_i, mem = neuron(x[i], dt)
        assert mem.shape == (batch_size, 34 * 34)
        assert x_i.shape == (batch_size, 34 * 34)

def test_neuron_dynamics():
    neuron = VariableStepLeaky(tau_m=20, threshold=10, output=True)
    x = torch.full((2, 2, 1), 0.75)
    dt = torch.asarray([[0.5, 0.75], [0.3, 0.55]])
    x_i, mem = neuron(x[0], dt[0])
    assert torch.allclose(x_i, torch.zeros((2, 1)))
    assert torch.allclose(mem, torch.full((2, 2, 1), 0.75))
    x_i, mem = neuron(x[1], dt[1])
    assert torch.allclose(x_i, torch.zeros((2, 1)))
    expected = 0.75 * torch.exp(-dt[1] / 20) + 0.75
    expected = expected.t()
    assert torch.allclose(mem.flatten(), expected)
    x_i, mem = neuron(torch.full((2, 1), 10.0), torch.asarray([10.0, 10.0]))
    assert torch.allclose(x_i, torch.ones((2, 1)))
    assert torch.allclose(mem, torch.zeros((2, 2, 1)))
