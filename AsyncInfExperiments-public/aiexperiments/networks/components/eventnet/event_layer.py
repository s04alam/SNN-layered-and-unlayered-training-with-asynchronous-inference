from abc import ABC, abstractmethod
from typing import List

import numpy as np
from aiexperiments.networks.components.eventnet.event_module import EventModule

from aiexperiments.networks.components.eventnet.lif import LIF, BarrierLIF, BarrierLIFParameters, LIFParameters
from aiexperiments.inference.event_utils import EventWithValue
import random
import torch


class EventLayer(ABC):
    neurons: List[EventModule]

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def forward(self, input_spikes: List[EventWithValue]) -> List[EventWithValue]:
        pass


class DenseLIFLayer(EventLayer):
    def __init__(self, n_in, n_out, lif_params: LIFParameters, lif_type = LIF, weights="random", inhibition=False):
        assert issubclass(lif_type, LIF)
        if weights is not None and not isinstance(weights, str):
            assert weights.shape == (n_out, n_in)
            if isinstance(weights, torch.Tensor):
                weights = weights.cpu().numpy()
            self.weights = weights
        elif weights == "random":
            self.weights = np.random.uniform(0, 0.3, (n_out, n_in))
        else:
            self.weights = np.zeros((n_out, n_in))
        self.bias = np.zeros(n_out)  # bias is not used in this implementation
        self.neurons = [lif_type(lif_params) for _ in range(n_out)]
        self.inhibition = inhibition
        self.inhibited_for_timestamp = None # timestamp when inhibition was triggered

    def reset_state(self):
        self.inhibited_for_timestamp = None
        for neuron in self.neurons:
            neuron.reset_state()

    def forward(self, input_spikes: List[EventWithValue]) -> List[EventWithValue]:
        output_spikes = []
        for input_spike in input_spikes:
            output_weights = self.weights[:, input_spike.index] * input_spike.value
            indices = list(enumerate(output_weights))
            random.shuffle(indices)
            
            for neuron_i, weight_to_neuron_i in indices:
                if isinstance(weight_to_neuron_i, torch.Tensor):
                    weight_to_neuron_i = weight_to_neuron_i.item()

                new_spikes = self.neurons[neuron_i].external_input(
                    input_spike.time, weight_to_neuron_i, input_spike.index
                )
                if self.inhibited_for_timestamp != input_spike.time:
                    for new_spike in new_spikes:
                        output_spikes.append(EventWithValue(neuron_i, new_spike.time, new_spike.value)) # add index
                if len(new_spikes) > 0 and self.inhibition:
                    self.inhibited_for_timestamp = input_spike.time # inhibit all other neurons for this timestamp
        return output_spikes


def test_dense_with_lif():
    lif_parameters = LIFParameters(
        tau_membrane=10,
        timestep_size=5,
        y_thresh=1.0,
    )
    n_out=2
    n_in=2
    weights = np.asarray([[0.1, 0.3], 
                          [0.2, 0.4]]) # (n_in, n_out)

    dense_lif_layer = DenseLIFLayer(n_in, n_out, lif_parameters, lif_type=LIF, weights=weights)
    beta = dense_lif_layer.neurons[-1].beta

    dense_lif_layer.forward([EventWithValue(0, 0, 1)])
    membrane_potentials_1 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    assert np.isclose(membrane_potentials_1, [0.1, 0.2]).all()
    dense_lif_layer.forward([EventWithValue(0, 1, 1)])
    membrane_potentials_2 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    assert np.isclose(membrane_potentials_2, [0.2, 0.4]).all()
    dense_lif_layer.forward([EventWithValue(0, 4, 1)])
    membrane_potentials_3 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    assert np.isclose(membrane_potentials_3, [0.3, 0.6]).all()
    dense_lif_layer.forward([EventWithValue(0, 5, 1)])
    membrane_potentials_4 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    test_membrane_potentials_4 = np.array(membrane_potentials_3)*beta + weights[:, 0]
    assert np.isclose(membrane_potentials_4, test_membrane_potentials_4).all()
    dense_lif_layer.forward([EventWithValue(0, 9, 0)])
    membrane_potentials_5 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    assert np.isclose(membrane_potentials_5, test_membrane_potentials_4).all() # so same as at t=5
    dense_lif_layer.forward([EventWithValue(0, 11, 0)])
    membrane_potentials_6 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons] 
    test_membrane_potentials_6 = np.array(membrane_potentials_5)*beta # expect decay, one step
    assert np.isclose(membrane_potentials_6, test_membrane_potentials_6).all()

def test_dense_with_lif_analytical():
    lif_parameters = LIFParameters(
        tau_membrane=10,
        timestep_size=0,
        y_thresh=1.0,
    )
    n_out=2
    n_in=2
    weights = np.asarray([[0.1, 0.3], 
                          [0.2, 0.4]]) # (n_in, n_out)

    dense_lif_layer = DenseLIFLayer(n_in, n_out, lif_parameters, lif_type=LIF, weights=weights)
    dense_lif_layer.forward([EventWithValue(0, 0, 1)])
    membrane_potentials_1 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    assert np.isclose(membrane_potentials_1, [0.1, 0.2]).all()
    dense_lif_layer.forward([EventWithValue(0, 1, 1)])
    membrane_potentials_2 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    test_membrane_potentials_2 = np.array(membrane_potentials_1)*np.exp(-1/lif_parameters.tau_membrane) + weights[:, 0]
    assert np.isclose(membrane_potentials_2, test_membrane_potentials_2).all()

def test_dense_with_barrier_lif_event_count():
    lif_parameters = BarrierLIFParameters(
        tau_membrane=10,
        timestep_size=5,
        y_thresh=1.0,
        min_barrier_events_for_spike=3
    )
    n_out=2
    n_in=2
    weights = np.asarray([[1, 0.3], 
                          [1, 0.4]]) # (n_in, n_out)

    dense_lif_layer = DenseLIFLayer(n_in, n_out, lif_parameters, lif_type=BarrierLIF, weights=weights)
    beta = dense_lif_layer.neurons[-1].beta
    dense_lif_layer.forward([EventWithValue(0, 0, 1)])
    membrane_potentials_1 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    assert np.isclose(membrane_potentials_1, [1, 1]).all()
    dense_lif_layer.forward([EventWithValue(0, 1, 1)])
    membrane_potentials_2 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    assert np.isclose(membrane_potentials_2, [2, 2]).all()
    dense_lif_layer.forward([EventWithValue(0, 1, 0)])
    membrane_potentials_3 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    assert np.isclose(membrane_potentials_3, [2, 2]).all()
    spikes_out = dense_lif_layer.forward([EventWithValue(0, 1, 0)])
    membrane_potentials_4 = [neuron.ys[-1] for neuron in dense_lif_layer.neurons]
    assert np.isclose(membrane_potentials_4, [0, 0]).all()
    assert len(spikes_out) == 2

if __name__ == "__main__":
    test_dense_with_lif()
    test_dense_with_lif_analytical()
    test_dense_with_barrier_lif_event_count()