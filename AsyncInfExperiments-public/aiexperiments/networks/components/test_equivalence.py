
import numpy as np
from aiexperiments.networks.components.eventnet.lif import BarrierLIF, BarrierLIFParameters
from aiexperiments.networks.components.matrix.leaky import Leaky
import torch


def test_matrix_leaky_equiv_to_event_barrier_lif():
    lif_parameters = BarrierLIFParameters(
        tau_membrane=10,
        timestep_size=5,
        y_thresh=1.0,
        min_barrier_events_for_spike=3
    )
    barrier_lif = BarrierLIF(lif_parameters)

    beta = np.exp(-lif_parameters.timestep_size/lif_parameters.tau_membrane)
    leaky = Leaky(beta=beta, threshold=lif_parameters.y_thresh, reset_mechanism='zero', output=True)

    barrier_lif.external_input(0, 0.5, 0)
    spk, mem = leaky(torch.tensor([0.5]))
    assert np.isclose(mem.item(), barrier_lif.ys[-1])
    assert np.isclose(mem.item(), 0.5)

    barrier_lif.external_input(5, 0, 0)
    spk, mem = leaky(torch.tensor([0]))
    assert np.isclose(mem.item(), barrier_lif.ys[-1])
    mem_t5 = 0.5 * beta
    assert np.isclose(mem.item(), mem_t5)

    barrier_lif.external_input(10, 0.17, 0)
    barrier_lif.external_input(10, 0.17, 0)
    barrier_lif.external_input(10, 0.17, 0)
    spk, mem = leaky(torch.tensor([0.51]))
    mem_t10 = mem_t5 * beta + 0.51
    assert np.isclose(mem.item(), mem_t10)
    assert np.isclose(mem.item(), barrier_lif.ys[-1])
    assert spk.sum() == 0

    barrier_lif.external_input(15, 10, 0)
    barrier_lif.external_input(15, -10, 0)
    barrier_lif.external_input(15, -0.1, 0)
    spk, mem = leaky(torch.tensor([-0.1]))
    mem_t15 = mem_t10 * beta - 0.1
    assert np.isclose(mem.item(), mem_t15)
    assert np.isclose(mem.item(), barrier_lif.ys[-1])
    assert spk.sum() == 0

    barrier_lif.external_input(20, 0.4, 0)
    barrier_lif.external_input(20, 0.4, 1)
    spk_out = barrier_lif.external_input(20, 0.4, 2)
    spk, mem = leaky(torch.tensor([1.2]))
    mem_t20 = 0
    assert np.isclose(mem.item(), mem_t20)
    assert np.isclose(mem.item(), barrier_lif.ys[-1])
    assert spk.sum() == len(spk_out)
    assert spk.sum() == 1