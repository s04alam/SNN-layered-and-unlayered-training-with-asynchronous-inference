import torch
from typing import Any, List, Mapping, Tuple
from torch import nn

from aiexperiments.asynctorch.nn.architecture.base_architecture import BaseArchitecture

# PROBLEM: this is very slow, much slower than the sparse version

# This implementation separates the weights per layer, instead of having a single weight matrix.
class DenseFullyLinearArchitecture(BaseArchitecture):
    weights: nn.Parameter # shape = (max number of ouputs per neuron/input, n_neurons+n_inputs)
    
    def __init__(self, n_inputs: int, neurons_per_layer: List[int], device: torch.device, *args, **kwargs):
        self.neurons_per_layer = neurons_per_layer
        super().__init__(n_inputs, sum(neurons_per_layer), device, *args, **kwargs)
        self.init_state()

    def forward(self, s: torch.Tensor, is_input: bool, is_input_and_neurons_combined: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        # Make sure we always have shape (n_neurons+n_inputs)
        if not is_input_and_neurons_combined:
            if is_input:
                s = torch.cat((s, torch.zeros((s.shape[0], self.n_neurons), device=self.device)), dim=1)
            else:
                s = torch.cat((torch.zeros((s.shape[0], self.n_inputs), device=self.device), s), dim=1)
        
        # Compute output currents
        output_currents = torch.einsum('ij,bj->bij', self.weights, s)

        # Initialize lists
        input_currents_all_layers = []
        count_currents_all_layers = []
        neuron_start_idx = 0
        neuron_end_idx = self.n_inputs

        for n_neurons in self.neurons_per_layer:
            currents_for_layer = output_currents[:, :, neuron_start_idx:neuron_end_idx]
            sum_input_currents = currents_for_layer.sum(dim=2)[:, :n_neurons]
            input_currents_all_layers.append(sum_input_currents)
            count_input_currents = (currents_for_layer != 0).sum(dim=2)[:, :n_neurons]
            count_currents_all_layers.append(count_input_currents)
            if neuron_start_idx == 0:
                neuron_start_idx += self.n_inputs
            else:
                neuron_start_idx = neuron_end_idx
            neuron_end_idx += n_neurons

        sum_input_currents = torch.cat(input_currents_all_layers, dim=1)
        count_input_currents = torch.cat(count_currents_all_layers, dim=1)

        return sum_input_currents, count_input_currents
    
    def init_state(self):
        weights_list = []
        for i in range(len(self.neurons_per_layer)):
            if i == 0:
                n_neurons_in = self.n_inputs
            else:
                n_neurons_in = self.neurons_per_layer[i-1]
            n_neurons_out = self.neurons_per_layer[i]
            weights_list.append(
                torch.zeros(
                    (n_neurons_out, n_neurons_in),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            torch.nn.init.xavier_uniform_(weights_list[-1])
        state_dict = DenseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_list)
        self.load_state_dict(state_dict)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], *args, strict: bool = False, **kwargs):
        weights: torch.Tensor = state_dict["weights"]
        if len(weights.shape) != 2:
            raise ValueError("Weights must be 2-dimensional")
        self.weights = nn.Parameter(weights)

    def get_weights_list(self) -> List[torch.Tensor]:
        """
        Returns:
            List[torch.Tensor]: List of weights for each layer.
        """
        weights = []
        weights.append(self.weights[0:self.neurons_per_layer[0], 0:self.n_inputs])
        n_neuron_layers = len(self.neurons_per_layer)
        for i in range(n_neuron_layers):
            neurons_in_start = sum(self.neurons_per_layer[0 : i])+self.n_inputs
            neurons_in_end = sum(self.neurons_per_layer[0 : i + 1])+self.n_inputs
            neurons_out_start = 0
            neurons_out_end = self.neurons_per_layer[i + 1]

            weights.append(self.weights[
                neurons_out_start : neurons_out_end,
                neurons_in_start : neurons_in_end,
            ])
        return weights

    def create_state_dict_from_weights_list(weights_list: List[torch.Tensor]) -> Mapping[str, Any]:
        """
        Builds the weights matrix from a list of tensors.

        Args:
            weights_list: A list of tensors, where each tensor represents a layer transformation.
        Returns:
            weights: shape = (max number of ouputs per neuron/input, n_neurons+n_inputs)
        """
        # 1. Acquire shapes
        # >> Input layer
        n_inputs = weights_list[0].shape[1]
        # >> Hidden layer
        n_neuron_layers = len(weights_list) - 1
        n_neurons_per_layer = [weights_list[i].shape[0] for i in range(0, len(weights_list))]
        n_neurons = sum(n_neurons_per_layer)
        # >> Max number of outputs per neuron/input
        max_n_outputs = max([weights_list[i].shape[0] for i in range(0, len(weights_list))])

        # 2. Build weights matrix and mask
        weights = torch.zeros((max_n_outputs, n_neurons+n_inputs), dtype=torch.float32, device=weights_list[0].device)

        # 3. Fill weights matrix
        weights[0:weights_list[0].shape[0], 0:n_inputs] = weights_list[0]
        for i in range(n_neuron_layers):
            neurons_in_start = sum(n_neurons_per_layer[0 : i])+n_inputs
            neurons_in_end = sum(n_neurons_per_layer[0 : i+1])+n_inputs

            weights[
                0:weights_list[i+1].shape[0],
                neurons_in_start : neurons_in_end,
            ] = weights_list[i+1]
        
        return {"weights": weights}