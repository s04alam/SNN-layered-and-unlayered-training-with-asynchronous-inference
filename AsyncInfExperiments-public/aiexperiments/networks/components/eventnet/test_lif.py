from aiexperiments.networks.components.eventnet.lif import LIF, LIFParameters
import numpy as np

def test_lif_double_decay():
    lif_params = LIFParameters(tau_membrane=10, timestep_size=5, y_thresh=1.0)
    beta = np.exp(-5/10)
    lif = LIF(lif_params)
    lif.external_input(5, 0.5, 0)
    assert lif.ys[-1] == 0.5
    lif.external_input(15, 0.5, 0)
    assert lif.ys[-1] == 0.5*beta*beta + 0.5

def test_lif_double_decay_analytical():
    lif_params = LIFParameters(tau_membrane=10, timestep_size=0, y_thresh=1.0)
    lif = LIF(lif_params)
    lif.external_input(5, 0.5, 0)
    assert lif.ys[-1] == 0.5
    lif.external_input(15, 0.5, 0)
    assert lif.ys[-1] == 0.5*np.exp(-10/lif_params.tau_membrane) + 0.5
