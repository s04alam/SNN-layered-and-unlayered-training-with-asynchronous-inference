
import torch

from aiexperiments.networks.components.matrix.leaky import Leaky

class VariableStepLeaky(Leaky):
    def __init__(
        self,
        tau_m,
        *args,
        refractory_period=0,
        random_refractory_period=False,
        **kwargs
    ):
        kwargs["beta"] = 1
        kwargs["init_hidden"] = True
        kwargs["learn_beta"] = False
        super(VariableStepLeaky, self).__init__(
            *args,
            **kwargs
        )
        self.tau_m = tau_m
        self.refractory_period = refractory_period
        self.random_refractory_period = random_refractory_period
        self.refrac_for = None

    def reset_hidden(self):
        super().reset_hidden()
        self.refrac_for = None

    def forward(self, input_, dt):  # override
        # update beta
        self.beta = torch.exp(-dt / self.tau_m)
        self._beta_buffer(self.beta, learn_beta=False)
        # adjust input such that it is 0 if in refractory period
        if self.refrac_for is not None:
            input_ = input_ * (self.refrac_for <= 0).float()
        # forward pass, update membrane potential
        output = super(VariableStepLeaky, self).forward(input_)
        if isinstance(output, tuple):
            spk = output[0]
        else:
            spk = output
        # update refractory period
        if self.refrac_for is None:
            self.refrac_for = torch.zeros_like(spk)
        dt_expanded = dt.unsqueeze(1)
        self.refrac_for = self.refrac_for - dt_expanded 
        if self.random_refractory_period and self.refractory_period > 0:
            refractory_period = torch.rand(1) * self.refractory_period
        else:
            refractory_period = self.refractory_period
        self.refrac_for = ((spk > 0).float() * refractory_period) + (
            (spk <= 0).float() * self.refrac_for
        )  # if spk > 0, set refrac_for to refractory_period, else keep old value
        return output

    def _beta_buffer(self, beta, learn_beta):  # override
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)
        self.register_buffer("beta", beta)

    def _base_state_function(self, input_, mem):  # override
        base_fn = self.beta.clamp(0, 1).unsqueeze(1) * mem + input_
        return base_fn

    def _base_state_function_hidden(self, input_):  # override
        base_fn = self.beta.clamp(0, 1).unsqueeze(1) * self.mem + input_
        return base_fn