from torch import nn
import torch
import time
import numpy as np


class EarlyAccumulateLinear(nn.Linear):
    def __init__(self, *args, threshold=0.3, K=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.K = K

    def forward(self, x: torch.Tensor, out_neuron_membrane_potential: torch.Tensor):
        batch_size, _ = x.shape
        num_neurons_in = self.weight.shape[1]
        num_neurons_out = self.weight.shape[0]
        can_accumulate = torch.ones(
            (batch_size, num_neurons_out), dtype=torch.bool, device=x.device
        )
        k = 0
        output = torch.zeros((batch_size, num_neurons_out), device=x.device)
        for neuron_in_idx in np.random.permutation(num_neurons_in):
            k += 1
            weighted_input = x[:, neuron_in_idx].unsqueeze(1) * self.weight[
                :, neuron_in_idx
            ].unsqueeze(0)
            output += weighted_input * can_accumulate
            if (
                out_neuron_membrane_potential is not None
                and len(out_neuron_membrane_potential.shape) == 2
                and out_neuron_membrane_potential.shape[1] == num_neurons_out
            ):
                can_accumulate = can_accumulate & (
                    (output + out_neuron_membrane_potential < self.threshold)
                    | (k < self.K)
                )
            else:
                can_accumulate = can_accumulate & (
                    (output < self.threshold) | (k < self.K)
                )
        return output


if __name__ == "__main__":
    torch.random.manual_seed(0)
    inp_spikes = torch.asarray([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=torch.float32)
    standard_linear = nn.Linear(3, 4, bias=False)
    early_spiking_linear = EarlyAccumulateLinear(3, 4, threshold=0.1)
    standard_linear.weight = early_spiking_linear.weight

    start = time.time()
    out_standard = standard_linear(inp_spikes)
    print(time.time() - start)
    start = time.time()
    out = early_spiking_linear(inp_spikes, torch.zeros(1, 4))
    print(time.time() - start)
    print(out_standard)
    print(out)
