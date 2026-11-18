from typing import List, Tuple
import numpy as np
import torch
from aiexperiments.asynctorch.simulator.extensions.currents_to_spike_monitor import CurrentsToSpikeMonitor
from aiexperiments.asynctorch.simulator.extensions.spike_dropout_extension import SpikeDropoutExtension
from aiexperiments.asynctorch.simulator.extensions.early_stopping_extension import EarlyStoppingExtension
from aiexperiments.asynctorch.simulator.extensions.spikes_to_output_monitor import SpikesToOutputMonitor
from aiexperiments.asynctorch.nn.architecture.dense_fully_linear_architecture import DenseFullyLinearArchitecture
from aiexperiments.asynctorch.nn.architecture.semi_sparse_fully_linear_architecture import SemiSparseFullyLinearArchitecture
from aiexperiments.asynctorch.nn.architecture.sparse_fully_linear_architecture import SparseFullyLinearArchitecture
from aiexperiments.asynctorch.nn.neuron.lif_state import LIFState
from aiexperiments.asynctorch.simulator.async_simulator import AsyncSimulator
from aiexperiments.asynctorch.simulator.extensions.stop_on_output_extension import StopOnOutputExtension
from aiexperiments.asynctorch.simulator.spike_scheduler import MomentumSpikeScheduler, RandomSpikeScheduler
from aiexperiments.asynctorch.simulator.spike_selector import SpikeSelector
from aiexperiments.constants import (
    AsyncTorchMode,
    AsyncTorchNetworkExtension,
    BarrierLIFNetworkExtension,
    EarlyAccumulateNetworkExtension,
    EarlyStoppingNetworkExtension,
    FullDropoutNetworkExtension,
    InputBarrierNetworkExtension,
    InputDropoutNetworkExtension,
    LogSpikeDynamicsNetworkExtension,
    MomentumNoiseNetworkExtension,
    NetworkDropoutNetworkExtension,
    NetworkExtension,
    NoRefracNetworkExtension,
    PreProcessingNetworkExtension,
    RefractoryDropoutNetworkExtension,
    StopOnOutputNetworkExtension,
    SyncThresholdNetworkExtension,
    WTANetworkExtension,
    PositiveSkewDropoutNetworkExtension,
    NegativeSkewDropoutNetworkExtension,
    get_network_extension_from_list,
    is_network_extension_type_in_list,
)
from aiexperiments.networks.components.eventnet.event_layer import DenseLIFLayer
from aiexperiments.networks.components.eventnet.lif import (
    LIF,
    BarrierLIF,
    BarrierLIFParameters,
    LIFParameters,
)
from aiexperiments.networks.components.matrix.early_accumulate_linear import (
    EarlyAccumulateLinear,
)
from aiexperiments.networks.components.matrix.spike_dropout import SpikeDropout
from aiexperiments.networks.components.matrix.variable_leaky import VariableStepLeaky
from aiexperiments.networks.base_network import (
    BaseEventDrivenNetwork,
    BaseNetworkBuilder,
    BaseNetworkBuilderParameters,
    BaseSNNTorchNetwork,
    calc_exp_decay_factor,
)
import torch.nn as nn
import copy
from aiexperiments.networks.components.matrix.leaky import Leaky

from aiexperiments.networks.components.matrix.synaptic_pp_layer import SynapticPPLayer
from aiexperiments.networks.components.matrix.skew_current_dropout import (
    PositiveSkewCurrentDropout,
    NegativeSkewCurrentDropout,
)
import snntorch.surrogate as surrogate


class FcMLPNetworkBuilderParameters(BaseNetworkBuilderParameters):
    def __init__(
        self,
        n_hidden,
        n_hidden_layers,
        tau_m,
        inhibit_spiking_output,
        threshold=0.3,
        spike_grad: torch.autograd.Function = None,
        pp_params={},
        add_pool4_layer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if is_network_extension_type_in_list(self.network_extensions, WTANetworkExtension) and inhibit_spiking_output:
            raise ValueError("WTA and inhibit_spiking_output are incompatible")
        self.add_pool4_layer = add_pool4_layer
        self.n_hidden = n_hidden
        self.inhibit_spiking_output = inhibit_spiking_output
        self.spike_grad = spike_grad
        self.tau_m = tau_m
        self.threshold = threshold
        self.pp_params = pp_params
        self.n_hidden_layers = n_hidden_layers
        print("MLP network structure:", self.get_neurons_per_layer())

    def get_neurons_per_layer(self):
        return [self.n_hidden] * self.n_hidden_layers + [self.n_out]


class SNNTorchFcMLP(BaseSNNTorchNetwork):
    def __init__(self, builder_params: FcMLPNetworkBuilderParameters):
        super().__init__(builder_params)
        beta = calc_exp_decay_factor(builder_params.timestep_size, builder_params.tau_m)
        output_threshold = 10000 if builder_params.inhibit_spiking_output else builder_params.threshold
        self.add_pp_layer = is_network_extension_type_in_list(
            builder_params.network_extensions, PreProcessingNetworkExtension
        )
        # Determine to use variable step or fixed step neuron model depending on timestep_size
        if builder_params.timestep_size == 0 or builder_params.timestep_size is None:
            if self.add_pp_layer:
                raise NotImplementedError("PP layer not implemented for variable step neurons")
            neuron_model = VariableStepLeaky
            kwargs_neuron = {"tau_m": builder_params.tau_m}
        else:
            neuron_model = Leaky
            kwargs_neuron = {}
        # Network extensions
        use_wta = is_network_extension_type_in_list(builder_params.network_extensions, WTANetworkExtension)
        # >> Dropout
        input_dropout_extension: InputDropoutNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, InputDropoutNetworkExtension
        )
        input_dropout = input_dropout_extension.dropout if input_dropout_extension is not None else 0
        network_dropout_extension: NetworkDropoutNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, NetworkDropoutNetworkExtension
        )
        network_dropout = network_dropout_extension.dropout if network_dropout_extension is not None else 0
        full_dropout_extension: FullDropoutNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, FullDropoutNetworkExtension
        )
        if full_dropout_extension is not None:  # full dropout overrides other dropout
            network_dropout = full_dropout_extension.dropout
            input_dropout = full_dropout_extension.dropout
        positive_skew_dropout_extensions: PositiveSkewDropoutNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, PositiveSkewDropoutNetworkExtension
        )
        negative_skew_dropout_extensions: NegativeSkewDropoutNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, NegativeSkewDropoutNetworkExtension
        )
        if positive_skew_dropout_extensions is not None or negative_skew_dropout_extensions is not None:
            spike_dropout_model = None
            current_dropout_model = (
                PositiveSkewCurrentDropout
                if positive_skew_dropout_extensions is not None
                else NegativeSkewCurrentDropout
            )
            kwargs_spike_dropout = {}
            kwargs_current_dropout = {
                "threshold": positive_skew_dropout_extensions.threshold
                if positive_skew_dropout_extensions is not None
                else negative_skew_dropout_extensions.threshold,
                "slope": positive_skew_dropout_extensions.slope
                if positive_skew_dropout_extensions is not None
                else negative_skew_dropout_extensions.slope,
            }
            self.use_current_dropout = True
        else:
            spike_dropout_model = SpikeDropout # nn.Dropout
            current_dropout_model = None
            kwargs_spike_dropout = {}
            kwargs_current_dropout = {}
            self.use_current_dropout = False
        # >> Early accumulate
        early_accumulate_extension: EarlyAccumulateNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, EarlyAccumulateNetworkExtension
        )
        if early_accumulate_extension is not None:
            linear_model = EarlyAccumulateLinear
            kwargs_linear = {
                "threshold": builder_params.threshold,
                "K": early_accumulate_extension.K,
            }
        else:
            linear_model = nn.Linear
            kwargs_linear = {}
        # Pool4 and PP layers
        if builder_params.add_pool4_layer:
            self.pool4_layer = nn.AvgPool2d(kernel_size=4)
        self.add_pool4_layer = builder_params.add_pool4_layer
        if self.add_pp_layer:
            self.pp_layer = SynapticPPLayer(
                **builder_params.pp_params,
                timestep_size=builder_params.timestep_size,
                spike_grad=builder_params.spike_grad,
            )
        # Input layer
        layers = [
            nn.Linear(
                builder_params.n_in, builder_params.n_hidden, bias=False
            ),  # never add early accumulate to input layer
        ]
        if current_dropout_model is not None:
            layers.append(current_dropout_model(input_dropout, **kwargs_current_dropout))
        layers.append(
            neuron_model(
                beta=beta,
                init_hidden=True,
                spike_grad=builder_params.spike_grad,
                threshold=builder_params.threshold,
                reset_mechanism="zero",
                **kwargs_neuron,
            )
        )
        if spike_dropout_model is not None:
            layers.append(spike_dropout_model(input_dropout, **kwargs_spike_dropout))
        # Hidden layers
        for _ in range(builder_params.n_hidden_layers - 1):
            layers.append(
                linear_model(
                    builder_params.n_hidden,
                    builder_params.n_hidden,
                    bias=False,
                    **kwargs_linear,
                )
            )
            if current_dropout_model is not None:
                layers.append(current_dropout_model(network_dropout, **kwargs_current_dropout))
            layers.append(
                neuron_model(
                    beta=beta,
                    init_hidden=True,
                    spike_grad=builder_params.spike_grad,
                    threshold=builder_params.threshold,
                    reset_mechanism="zero",
                    **kwargs_neuron,
                )
            )
            if spike_dropout_model is not None:
                layers.append(spike_dropout_model(network_dropout, **kwargs_spike_dropout))
        # Output layer
        layers.extend(
            [
                linear_model(
                    builder_params.n_hidden,
                    builder_params.n_out,
                    bias=False,
                    **kwargs_linear,
                ),
                neuron_model(
                    beta=beta,
                    init_hidden=True,
                    output=True,
                    spike_grad=builder_params.spike_grad,
                    threshold=output_threshold,
                    reset_mechanism="zero",
                    inhibition=use_wta,
                    **kwargs_neuron,
                ),
            ]
        )
        self.layers = nn.Sequential(*layers)
        self.n_hidden = builder_params.n_hidden
        self.neurons_per_layer = builder_params.get_neurons_per_layer()
        self.reset_state()

    def can_implement_network_extension(self, network_extension: NetworkExtension) -> bool:
        if (
            isinstance(network_extension, WTANetworkExtension)
            or isinstance(network_extension, PreProcessingNetworkExtension)
            or isinstance(network_extension, EarlyAccumulateNetworkExtension)
            or isinstance(network_extension, InputDropoutNetworkExtension)
            or isinstance(network_extension, NetworkDropoutNetworkExtension)
            or isinstance(network_extension, FullDropoutNetworkExtension)
            or isinstance(network_extension, PositiveSkewDropoutNetworkExtension)
            or isinstance(network_extension, NegativeSkewDropoutNetworkExtension)
        ):
            return True
        else:
            return super().can_implement_network_extension(network_extension)

    def supports_variable_timestep_size(self) -> bool:
        return True

    def reset_state(self):
        if self.add_pp_layer:
            self.pp_layer.reset_state()
        for layer in self.layers:
            if isinstance(layer, Leaky) or isinstance(layer, VariableStepLeaky):
                layer.reset_hidden()
                layer.detach_hidden()
        self.order_recordings = []
        self.spike_counts = []

    def detach_state(self):
        if self.add_pp_layer:
            self.pp_layer.detach_state()
        for layer in self.layers:
            if isinstance(layer, Leaky) or isinstance(layer, VariableStepLeaky):
                layer.detach_hidden()

    def forward(
        self, x: torch.Tensor, dt: torch.Tensor = None
    ):  # shape (batch_size, n_in); dt is only used for variable step neurons
        batch_size = x.shape[0]
        if self.add_pool4_layer:
            x = self.pool4_layer(x)
        x = x.view(x.shape[0], -1)

        spike_counts = torch.zeros((batch_size,), device=self.device)
        #spike_counts = spike_counts + torch.sum(x, dim=1)
        if self.add_pp_layer:
            x = self.pp_layer(x)
            #spike_counts = spike_counts + torch.sum(x, dim=1)
        for i_layer, layer in enumerate(self.layers):
            if isinstance(layer, VariableStepLeaky):
                x = layer(x, dt)
            elif isinstance(layer, EarlyAccumulateLinear):
                if self.use_current_dropout and i_layer + 2 < len(
                    self.layers
                ):  # if using current dropout, need to skip the next layer (except for the last layer)
                    mem = self.layers[i_layer + 2].mem
                else:
                    mem = self.layers[i_layer + 1].mem
                x = layer(x, mem)
            else:
                x = layer(x)
            if isinstance(layer, Leaky) or isinstance(layer, VariableStepLeaky):
                spike_counts = spike_counts + layer.spk.sum(dim=1)

        if len(self.spike_counts) == 0:
            self.spike_counts = spike_counts
        else:
            self.spike_counts = self.spike_counts + spike_counts
        return x  # final layer output, will be a tuple (spk, mem)

    def get_current_state(self) -> torch.Tensor:
        memories = []
        for layer in self.layers:
            if isinstance(layer, Leaky) or isinstance(layer, VariableStepLeaky):
                memories.append(layer.mem.clone())
        return torch.cat(memories, dim=1)

    def get_spike_counts(self) -> torch.Tensor:
        return self.spike_counts


class EventDrivenFcMLP(BaseEventDrivenNetwork):
    def __init__(self, builder_params: FcMLPNetworkBuilderParameters):
        super().__init__(builder_params)
        if builder_params.add_pool4_layer:
            raise NotImplementedError("Event-driven MLP does not support pool4 layer")

        # Setup barrier LIF if necessary
        add_input_barrier_extension = is_network_extension_type_in_list(
            builder_params.network_extensions, InputBarrierNetworkExtension
        )
        barrier_lif_extension: BarrierLIFNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, BarrierLIFNetworkExtension
        )
        lif_params_kwargs = {
            "tau_membrane": builder_params.tau_m,
            "timestep_size": builder_params.timestep_size,
            "y_thresh": builder_params.threshold,
            #"refractory_period": None, # TODO: wrong place to put this
        }
        if barrier_lif_extension is not None:
            lif_type = BarrierLIF
            lif_params = BarrierLIFParameters(
                min_barrier_events_for_spike=barrier_lif_extension.barrier_size,
                send_barrier_events=barrier_lif_extension.send_barrier_events,
                **lif_params_kwargs,
            )
            input_lif_type = BarrierLIF
            input_lif_params = copy.deepcopy(lif_params)
            if add_input_barrier_extension:
                # only need one (true) barrier message
                input_lif_params.min_barrier_events_for_spike = 1
                input_lif_params.ignore_spike_as_barrier_message = True
            else:
                # no barrier messages needed for input layer, since the input does not
                # emit any barrier messages and thus we cannot rely on them
                input_lif_params.min_barrier_events_for_spike = 0
            output_lif_params = copy.deepcopy(lif_params)
            output_lif_params.send_barrier_events = False
        else:
            lif_type = LIF
            lif_params = LIFParameters(
                **lif_params_kwargs,
            )
            output_lif_params = lif_params
            if add_input_barrier_extension:
                input_lif_type = BarrierLIF
                input_lif_params = BarrierLIFParameters(
                    tau_membrane=builder_params.tau_m,
                    timestep_size=builder_params.timestep_size,
                    y_thresh=builder_params.threshold,
                    min_barrier_events_for_spike=1,
                    send_barrier_events=False,
                    ignore_spike_as_barrier_message=True,
                )
            else:
                input_lif_type = LIF
                input_lif_params = lif_params

        # WTA
        use_wta = is_network_extension_type_in_list(builder_params.network_extensions, WTANetworkExtension)

        # Setup layers
        self.layers = [
            DenseLIFLayer(
                builder_params.n_in,
                builder_params.n_hidden,
                lif_params=input_lif_params,
                lif_type=input_lif_type,
            ),
        ]
        for i in range(builder_params.n_hidden_layers - 1):
            self.layers.append(
                DenseLIFLayer(
                    builder_params.n_hidden,
                    builder_params.n_hidden,
                    lif_params=lif_params,
                    lif_type=lif_type,
                )
            )
        self.layers.append(
            DenseLIFLayer(
                builder_params.n_hidden,
                builder_params.n_out,
                lif_params=output_lif_params,
                lif_type=lif_type,
                inhibition=use_wta,
            )
        )
        final_layer_threshold = 10000 if builder_params.inhibit_spiking_output else builder_params.threshold
        for neuron in self.layers[-1].neurons:  # TODO: set this earlier
            neuron.y_thresh = final_layer_threshold

    def can_implement_network_extension(self, network_extension: NetworkExtension) -> bool:
        if (
            isinstance(network_extension, WTANetworkExtension)
            or isinstance(network_extension, BarrierLIFNetworkExtension)
            or isinstance(network_extension, InputBarrierNetworkExtension)
        ):
            return True
        else:
            return super().can_implement_network_extension(network_extension)

    def supports_variable_timestep_size(self) -> bool:
        return True

    def load_parameters(self, load_parameters_from: str):
        if load_parameters_from is not None:
            params = torch.load(load_parameters_from, map_location=self.device)
            for i, layer in enumerate(self.layers):
                layer.weights = params[f"layers.{i*3}.weight"].cpu().numpy() 
            if "sync_thresholds" in params:
                sync_thresholds = params["sync_thresholds"]
                i_neuron = 0
                for layer in self.layers:
                    for neuron in layer.neurons:
                        if isinstance(neuron, BarrierLIF):
                            if neuron.min_barrier_events_for_spike == 0:
                                threshold = np.floor(max(sync_thresholds[i_neuron].item(), 0.0)) + 1.0
                                neuron.min_barrier_events_for_spike = threshold
                        i_neuron += 1


class AsyncTorchFcMLP(BaseSNNTorchNetwork):
    def __init__(
        self,
        builder_params: FcMLPNetworkBuilderParameters,
        forward_group_size: int,
        mode: AsyncTorchMode,
    ):
        super().__init__(builder_params)
        neurons_per_layer = builder_params.get_neurons_per_layer()
        self.timestep_size = builder_params.timestep_size
        self.n_outputs = builder_params.n_out

        ## LIF
        sync_threshold_extension: SyncThresholdNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, SyncThresholdNetworkExtension
        )
        if sync_threshold_extension is not None:
            if sync_threshold_extension.trainable:
                sync_threshold = torch.empty((self.n_neurons,), dtype=torch.float32, device=self.device).normal_(
                    mean=sync_threshold_extension.mean, std=sync_threshold_extension.std
                )
            else:
                sync_threshold = torch.full(
                    (self.n_neurons,), sync_threshold_extension.mean, device=self.device, dtype=torch.float32
                )
            if not sync_threshold_extension.trainable_input_layer:
                sync_threshold[: neurons_per_layer[0]] = 0.0
        else:
            sync_threshold = 0.0
        if builder_params.inhibit_spiking_output:
            output_threshold = 1e6
            membrane_threshold = torch.full((self.n_neurons,), builder_params.threshold, device=self.device)
            membrane_threshold[self.n_neurons-self.n_outputs:] = output_threshold
        else:
            membrane_threshold = builder_params.threshold
        spike_grad = builder_params.spike_grad if builder_params.spike_grad is not None else surrogate.straight_through_estimator()
        sync_grad = builder_params.spike_grad if builder_params.spike_grad is not None else surrogate.straight_through_estimator()
        refrac_dropout_extension: RefractoryDropoutNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, RefractoryDropoutNetworkExtension
        )
        refrac_dropout = refrac_dropout_extension.dropout if refrac_dropout_extension is not None else 0
        apply_refrac = not is_network_extension_type_in_list(builder_params.network_extensions, NoRefracNetworkExtension)
        neuron_state = LIFState(
            neurons_per_layer,
            builder_params.tau_m,
            membrane_threshold,
            sync_threshold,
            spike_grad,
            sync_grad,
            builder_params.device,
            refrac_dropout=refrac_dropout,
            apply_refrac=apply_refrac,
        )

        ## Architecture
        architecture = SemiSparseFullyLinearArchitecture(
            builder_params.n_in,
            neurons_per_layer,
            builder_params.device
        )

        ## Spike selector
        if mode == AsyncTorchMode.RANDOM:
            spike_scheduler = RandomSpikeScheduler(neuron_state)
        elif mode == AsyncTorchMode.MEMBRANE_POTENTIAL:
            momentum_noise_extension: MomentumNoiseNetworkExtension = get_network_extension_from_list(
                builder_params.network_extensions, MomentumNoiseNetworkExtension
            )
            if momentum_noise_extension is not None:
                lambda_ = momentum_noise_extension.lambda_
            else:
                lambda_ = 1e-6
            spike_scheduler = MomentumSpikeScheduler(neuron_state, lambda_)
        else:
            raise NotImplementedError(f"Mode {mode} not implemented")
        spike_selector = SpikeSelector(
            architecture,
            spike_scheduler,
            forward_group_size,
            builder_params.device,
            prioritize_input=True, # = input barrier
        )

        ## Simulator
        ### Extensions
        forward_step_extensions = []
        # > Dropout
        input_dropout_extension: InputDropoutNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, InputDropoutNetworkExtension
        )
        input_dropout = input_dropout_extension.dropout if input_dropout_extension is not None else 0
        network_dropout_extension: NetworkDropoutNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, NetworkDropoutNetworkExtension
        )
        network_dropout = network_dropout_extension.dropout if network_dropout_extension is not None else 0
        full_dropout_extension: FullDropoutNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, FullDropoutNetworkExtension
        )
        if full_dropout_extension is not None:  # full dropout overrides other dropout
            network_dropout = full_dropout_extension.dropout
            input_dropout = full_dropout_extension.dropout
        if input_dropout > 0:
            forward_step_extensions.append(
                SpikeDropoutExtension(input_dropout, True, False)
            )
        if network_dropout > 0:
            forward_step_extensions.append(
                SpikeDropoutExtension(network_dropout, False, True)
            )
        # > Early stopping
        early_stopping_extension: EarlyStoppingNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, EarlyStoppingNetworkExtension
        )
        if early_stopping_extension is not None:
            forward_step_extensions.append(
                EarlyStoppingExtension(early_stopping_extension.n_steps)
            )
        # > Stop on output
        stop_on_output_extension: StopOnOutputNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, StopOnOutputNetworkExtension
        )
        if stop_on_output_extension is not None:
            forward_step_extensions.append(
                StopOnOutputExtension(self.n_neurons, self.n_outputs, stop_on_output_extension.max_post_output_steps)
            )

        ### Monitors
        # > Spikes to classification monitor
        self.spikes_to_classification_monitor = SpikesToOutputMonitor(self.n_neurons, self.n_outputs)
        forward_step_extensions.append(
            self.spikes_to_classification_monitor
        )
        # > Log spike dynamics
        log_spike_dynamics_extension: LogSpikeDynamicsNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, LogSpikeDynamicsNetworkExtension
        )
        if log_spike_dynamics_extension is not None:
            self.spike_dynamics_monitor = CurrentsToSpikeMonitor(self.n_neurons, neurons_per_layer[0], self.device)
            forward_step_extensions.append(
                self.spike_dynamics_monitor
            )
        else:
            self.spike_dynamics_monitor = None

        self.async_simulator = AsyncSimulator(
            neuron_state,
            spike_selector,
            forward_step_extensions,
        )   

    def get_latencies(self, target_classes: torch.Tensor) -> Tuple[List[int], List[int]]:
        return self.spikes_to_classification_monitor.compute_latencies(target_classes)
    
    def get_dynamics(self) -> Tuple[List[float], List[float], List[float]]:
        if self.spike_dynamics_monitor is not None:
            return self.spike_dynamics_monitor.n_currents_to_spike_log, self.spike_dynamics_monitor.sum_excitatory_currents_to_spike_log, self.spike_dynamics_monitor.sum_inhibitory_currents_to_spike_log
        else:
            return [], [], []

    def can_implement_network_extension(self, network_extension: NetworkExtension) -> bool:
        if (
            isinstance(network_extension, AsyncTorchNetworkExtension)
            or isinstance(network_extension, InputDropoutNetworkExtension)
            or isinstance(network_extension, NetworkDropoutNetworkExtension)
            or isinstance(network_extension, FullDropoutNetworkExtension)
            or isinstance(network_extension, SyncThresholdNetworkExtension)
            or isinstance(network_extension, RefractoryDropoutNetworkExtension)
            or isinstance(network_extension, NoRefracNetworkExtension)
            or isinstance(network_extension, MomentumNoiseNetworkExtension)
            or isinstance(network_extension, EarlyStoppingNetworkExtension)
            or isinstance(network_extension, StopOnOutputNetworkExtension)
            or isinstance(network_extension, LogSpikeDynamicsNetworkExtension)
        ):
            return True
        else:
            return super().can_implement_network_extension(network_extension)
        
    def reset_state(self):
        self.async_simulator.reset_state()

    def detach_state(self):
        self.async_simulator.detach_state()

    def get_current_state(self) -> torch.Tensor:
        return self.async_simulator.neuron_state.get_post_spike_state()
    
    def get_spike_counts(self) -> torch.Tensor:
        return self.async_simulator.spike_counts
    
    def forward(self, x: torch.Tensor):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        spk, mem = self.async_simulator(x, self.timestep_size)[:, self.n_neurons-self.n_outputs:], self.async_simulator.neuron_state.get_post_spike_state()[:, self.n_neurons-self.n_outputs:]
        return spk, mem

    # This converts the snntorch state_dict to a state_dict that can be loaded by asynctorch
    def load_state_dict(self, state_dict, strict=False):
        weights_per_layer = []
        sync_threshold = None
        for key in state_dict:
            key_split = key.split(".")
            if key_split[-1] == "weight":
                weights_per_layer.append(state_dict[key].to(self.device))
            elif key_split[-1] == "sync_thresholds":
                sync_threshold = state_dict[key].to(self.device)
        architecture_state_dict = SemiSparseFullyLinearArchitecture.create_state_dict_from_weights_list(weights_per_layer)
        state_dict = {
            "neuron_state.sync_threshold": sync_threshold,
            "spike_selector.architecture.weights": architecture_state_dict["weights"],
            "spike_selector.architecture.input_weights": architecture_state_dict["input_weights"],
            "spike_selector.architecture.weights_mask": architecture_state_dict["weights_mask"],
        }
        self.async_simulator.load_state_dict(state_dict, strict)
        
    # This converts the asynctorch state_dict to a state_dict that can be loaded by snntorch
    def state_dict(self, layer_state_dict_param_name_multiplier=3):
        state_dict = {}
        weights = self.async_simulator.spike_selector.architecture.get_weights_list()
        for i, weight in enumerate(weights):
            state_dict[f"layers.{i*layer_state_dict_param_name_multiplier}.weight"] = weight
        state_dict["sync_thresholds"] = self.async_simulator.neuron_state.sync_threshold
        return state_dict


class FcMLPBuilder(BaseNetworkBuilder):
    def __init__(self):
        super().__init__()

    def get_snntorch_network(self, builder_params: FcMLPNetworkBuilderParameters) -> BaseSNNTorchNetwork:
        async_torch_network_extension: AsyncTorchNetworkExtension = get_network_extension_from_list(
            builder_params.network_extensions, AsyncTorchNetworkExtension
        )
        if async_torch_network_extension is not None:
            return AsyncTorchFcMLP(
                builder_params,
                async_torch_network_extension.forward_group_size,
                async_torch_network_extension.mode,
            )
        else:
            return SNNTorchFcMLP(builder_params)

    def get_event_driven_network(self, builder_params: FcMLPNetworkBuilderParameters):
        return EventDrivenFcMLP(builder_params)
