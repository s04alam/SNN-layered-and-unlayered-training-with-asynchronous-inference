import snntorch as snn
import torch
from snntorch._neurons.neurons import _SpikeTorchConv

# This is a modified version of the Leaky class from snntorch.
# It makes the reset happen at the end of the timestep, rather than the beginning.
# source: https://snntorch.readthedocs.io/en/latest/_modules/snntorch/_neurons/leaky.html#Leaky
class Leaky(snn.Leaky):
    def __init__(self, *args, init_hidden=True, reset_mechanism='zero', state_quant=False, **kwargs):
        if not init_hidden:
            raise NotImplementedError("The modified Leaky class is only implemented for init_hidden=True.")
        if reset_mechanism != 'zero':
            raise NotImplementedError("The modified Leaky class is only implemented for reset_mechanism='zero'.")
        if state_quant:
            raise NotImplementedError("The modified Leaky class is only implemented for state_quant=False.")
        super().__init__(*args, init_hidden=init_hidden, reset_mechanism=reset_mechanism, state_quant=state_quant, **kwargs)

    def forward(self, input_, mem=False):
        if hasattr(
            self.mem, "init_flag"
        ):  # init_hidden case
            self.mem = _SpikeTorchConv(self.mem, input_=input_)

        self._leaky_forward_cases(mem)
        self.mem = self._base_state_function_hidden(input_) # add input, decay
        if self.inhibition:
            self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
        else:
            self.spk = self.fire(self.mem)
        self.reset = self.mem_reset(self.mem) # check for reset
        self.mem = (1 - self.reset) * self.mem # apply reset

        if self.output:  # read-out layer returns output+states
            return self.spk, self.mem
        else:  # hidden layer e.g., in nn.Sequential, only returns output
            return self.spk