from abc import ABC
from abc import abstractmethod
from abc import ABCMeta
from typing import List, Tuple
import torch.nn as nn
import torch
import numpy as np
import h5py
import uuid
from aiexperiments.constants import FineTuneNetworkExtension, NetworkExtension, NetworkBackend, SpikeSparsityLossNetworkExtension
from aiexperiments.networks.components.eventnet.event_layer import EventLayer
from aiexperiments.inference.event_utils import EventWithValue


def calc_exp_decay_factor(timestep_size, tau):
    return np.exp(-timestep_size / tau)


class BaseNetworkBuilderParameters(ABC):
    # all parameters are given in micrseconds
    def __init__(
        self,
        n_in,
        n_out,
        device,
        timestep_size=1,  # timestep_size may be None for networks that don't use it, in microseconds
        load_parameters_from=None,
        network_extensions: List[NetworkExtension] = [],
    ):
        self.timestep_size = timestep_size
        self.n_in = n_in  # may be a scalar, or a tuple of scalars
        self.n_out = n_out  # always a scalar
        self.device = device
        self.load_parameters_from = load_parameters_from
        self.network_extensions = network_extensions

    def get_n_inputs(self):
        if hasattr(self.n_in, "__len__"):
            return np.prod(self.n_in)
        else:
            return self.n_in

    @abstractmethod
    def get_neurons_per_layer(self):
        pass # expected to return [int, int, int ...] where the length is the number of layers, and each int is the number of neurons in that layer


class BaseNetwork(ABC):
    def __init__(self, builder_params: BaseNetworkBuilderParameters):
        super().__init__()
        self.device = builder_params.device
        if (builder_params.timestep_size is None or builder_params.timestep_size <= 0) and not self.supports_variable_timestep_size():
            raise ValueError("Timestep size must be greater than 0 for this network")
        for network_extension in builder_params.network_extensions:
            if not self.can_implement_network_extension(network_extension):
                raise ValueError(f"Cannot implement network extension {network_extension} in this network")

    def can_implement_network_extension(self, network_extension: NetworkExtension) -> bool:
        if isinstance(network_extension, FineTuneNetworkExtension) or isinstance(network_extension, SpikeSparsityLossNetworkExtension):
            return True
        else:
            return False
    
    def supports_variable_timestep_size(self) -> bool:
        return False

    def count_parameters(self) -> int:
        return 0

    def save_parameters(self, save_to: str):
        raise NotImplementedError


class BaseSNNTorchNetwork(BaseNetwork, nn.Module, metaclass=ABCMeta):

    def __init__(self, builder_params: BaseNetworkBuilderParameters):
        super().__init__(builder_params)
        self.n_neurons = sum(builder_params.get_neurons_per_layer())

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def detach_state(self):
        pass

    @abstractmethod
    def get_spike_counts(self) -> torch.Tensor:
        pass

    def get_latencies(self, targets) -> Tuple[List[int], List[int]]:
        return [], []
    
    def get_dynamics(self) -> Tuple[List[float], List[float], List[float]]:
        return [], [], [] # n_currents, exc_currents, inh_currents

    def get_current_state(self) -> torch.Tensor:
        raise NotImplementedError

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_parameters(self, save_to: str):
        torch.save(self.state_dict(), save_to) # TODO: only save weights, not other attributes
        

class BaseEventDrivenNetwork(BaseNetwork, metaclass=ABCMeta):
    layers: List[EventLayer]

    def __init__(self, builder_params: BaseNetworkBuilderParameters):
        super().__init__(builder_params)

    def supports_variable_timestep_size(self) -> bool:
        return True

    @abstractmethod
    def load_parameters(self, load_parameters_from: str):
        pass

    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()

    def forward(self, layer_idx: int, event: List[EventWithValue]) -> List[EventWithValue]:
        return self.layers[layer_idx].forward(event)


class BaseNetworkBuilder(ABC):
    # -------------------- SNNTorch networks --------------------
    # Both training and inference
    # Can only import parameters from SNNTorch networks
    def build_snntorch_network(
        self, builder_params: BaseNetworkBuilderParameters
    ) -> BaseSNNTorchNetwork:
        net = self.get_snntorch_network(builder_params)
        self.set_snntorch_network_params(net, builder_params.load_parameters_from)
        return net

    def get_snntorch_network(
        self, builder_params: BaseNetworkBuilderParameters
    ) -> BaseSNNTorchNetwork:
        raise NotImplementedError()

    def set_snntorch_network_params(
        self, net: BaseSNNTorchNetwork, load_parameters_from: str
    ) -> None:
        if load_parameters_from is not None:
            params = torch.load(load_parameters_from, map_location=net.device, weights_only=True)
            # remove all non-weight attributes
            for key in list(params.keys()):
                if not key.endswith("weight"):
                    del params[key]
            net.load_state_dict(
                params, strict=False
            )

    # -------------------- Event driven networks --------------------
    # Inference only
    def build_event_driven_network(
        self, builder_params: BaseNetworkBuilderParameters
    ) -> BaseEventDrivenNetwork:
        net = self.get_event_driven_network(builder_params)
        self.set_event_driven_network_params(net, builder_params.load_parameters_from)
        return net

    def get_event_driven_network(
        self, builder_params: BaseNetworkBuilderParameters
    ) -> BaseEventDrivenNetwork:
        raise NotImplementedError()

    def set_event_driven_network_params(
        self, net: BaseEventDrivenNetwork, load_parameters_from: str
    ) -> None:
        net.load_parameters(load_parameters_from)
