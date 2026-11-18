from torch import nn
import snntorch as snn

from aiexperiments.networks.base_network import calc_exp_decay_factor


class SynapticPPLayer(nn.Module):
    def __init__(
        self, timestep_size, pp_tau_s, pp_tau_m, pp_base_threshold, pp_base_threshold_dt
    ):
        super().__init__()
        alpha = calc_exp_decay_factor(timestep_size, pp_tau_s)
        beta = calc_exp_decay_factor(timestep_size, pp_tau_m)
        threshold = pp_base_threshold * (pp_base_threshold_dt / timestep_size)
        self.synaptic = snn.Synaptic(
            alpha=alpha,
            beta=beta,
            threshold=threshold,
            reset_mechanism="zero",
        )
        self.reset_state()

    def reset_state(self):
        self.syn, self.mem = self.synaptic.init_synaptic()

    def detach_state(self):
        self.syn = self.syn.detach()
        self.mem = self.mem.detach()

    def forward(self, x):
        x, self.syn, self.mem = self.synaptic(x, self.syn, self.mem)
        return x
