from typing import Union
from aiexperiments.base_engine import StaticParameters
from aiexperiments.constants import DatasetProvider, LossFunctionType, NetworkBackend
from aiexperiments.inference.base_inference import (
    BaseInferenceEngine,
    InferenceParameters,
)
from aiexperiments.inference.event_utils import event_tensor_to_fixed_timestep_tensor
from aiexperiments.visualizing.metrics import ce_last_membrane_loss, ce_last_nonzero_membrane_loss, last_membrane_acc_fn, last_nonzero_membrane_acc_fn, max_membrane_acc_fn
from aiexperiments.visualizing.recording.inference_recorder import InferenceBatchResult
from aiexperiments.networks.base_network import (
    BaseNetwork,
    BaseNetworkBuilder,
    BaseNetworkBuilderParameters,
    BaseSNNTorchNetwork,
)
import torch

import snntorch.functional as SF


class MatrixInferenceParameters(InferenceParameters):
    def __init__(
        self,
        device,
        network_backend: NetworkBackend,
        loss_function_type: LossFunctionType,
        true_spike_rate=0.2,
        false_spike_rate=0.03,
        variable_dt_save_output_with_dt=1,  # Only used if timestep_size == 0, otherwise ignored. This is the dt to use for saving the output (e.g. for plotting).
        **kwargs,
    ):
        if network_backend == NetworkBackend.SNNTORCH:
            dataset_provider = DatasetProvider.TONIC_FRAME
        else:
            raise NotImplementedError(
                f"Network backend {network_backend} not implemented."
            )
        super().__init__(dataset_provider=dataset_provider, **kwargs)
        self.network_backend = network_backend
        self.device = device
        self.variable_dt_save_output_with_dt = variable_dt_save_output_with_dt
        if variable_dt_save_output_with_dt <= 0:
            raise ValueError("variable_dt_save_output_with_dt must be > 0")
        self.loss_function_type = loss_function_type
        if network_backend == NetworkBackend.SNNTORCH:
            if loss_function_type == LossFunctionType.RATE:
                self.loss_function = SF.ce_rate_loss()
                self.accuracy_function = SF.accuracy_rate
            elif loss_function_type == LossFunctionType.COUNT:
                self.loss_function = SF.ce_count_loss()
                self.accuracy_function = SF.accuracy_rate
            elif loss_function_type == LossFunctionType.TEMPORAL:
                self.loss_function = SF.ce_temporal_loss()
                self.accuracy_function = SF.accuracy_temporal
            elif loss_function_type == LossFunctionType.MAX_MEMBRANE:
                self.loss_function = SF.ce_max_membrane_loss()
                self.accuracy_function = max_membrane_acc_fn
            elif loss_function_type == LossFunctionType.LAST_MEMBRANE:
                if self.timestep_size > 0:
                    self.loss_function = ce_last_membrane_loss()
                    self.accuracy_function = last_membrane_acc_fn
                else:
                    self.loss_function = ce_last_nonzero_membrane_loss()
                    self.accuracy_function = last_nonzero_membrane_acc_fn
            else:
                raise NotImplementedError(
                    f"Loss function {loss_function_type} not implemented."
                )
        else:
            raise NotImplementedError(
                f"Network backend {network_backend} not implemented."
            )


class MatrixInferenceEngine(BaseInferenceEngine):
    def __init__(self, static_params: StaticParameters):
        super().__init__(static_params)

    def build_network(
        self,
        params: MatrixInferenceParameters,
        net_builder: BaseNetworkBuilder,
        net_builder_params: BaseNetworkBuilderParameters,
    ) -> BaseNetwork:
        if params.network_backend == NetworkBackend.SNNTORCH:
            return net_builder.build_snntorch_network(net_builder_params).to(
                params.device
            )
        else:
            raise NotImplementedError(
                f"Backend {params.network_backend} not implemented."
            )

    def infer_batch(
        self,
        net: Union[BaseSNNTorchNetwork, None],
        params: MatrixInferenceParameters,
        net_builder: BaseNetworkBuilder,
        net_builder_params: BaseNetworkBuilderParameters,
        data,  # is tuple of shape (data, dts) if timestep size == 0, else just data (torch.Tensor)
        targets,
    ) -> InferenceBatchResult:
        assert net is not None
        net.reset_state()
        net.eval()
        with torch.no_grad():
            if isinstance(data, tuple):
                timesteps = data[1]
                data = data[0] # shape [n_timesteps, batch_size, *]
                timesteps = timesteps.to(params.device)
            if not isinstance(data, torch.Tensor):
                raise ValueError("Data must be a torch.Tensor")
            y_save_overwrite = None # If not None, overwrite y with this value for saving purposes
            data = data.to(params.device)
            targets = targets.to(params.device)
            if params.network_backend == NetworkBackend.SNNTORCH:
                if params.track_membrane_recordings:
                    membrane_recordings_per_t = []
                    membrane_timesteps = []
                ys = []
                last_spike_count = 0
                for i in range(data.size(0)):
                    if params.timestep_size >= 1:
                        spk_out, mem_out = net(data[i])
                    else:
                        spk_out, mem_out = net(data[i], timesteps[i])
                    if (
                        params.loss_function_type == LossFunctionType.LAST_MEMBRANE
                        or params.loss_function_type == LossFunctionType.MAX_MEMBRANE
                    ):
                        ys.append(mem_out)
                    else:
                        ys.append(spk_out)
                    spike_count = torch.sum(net.get_spike_counts()).item()
                    if params.track_membrane_recordings and (data[i].sum() > 0 or spike_count > last_spike_count):
                        net_state = net.get_current_state()
                        membrane_recordings_per_t.append(net_state)
                        if params.timestep_size >= 1:
                            membrane_timesteps.append(torch.full((data.size(1),), i*params.timestep_size).long())
                        else:
                            membrane_timesteps.append(torch.sum(timesteps[:(i+1), :], dim=0).long())
                    last_spike_count = spike_count
                if params.track_membrane_recordings and len(membrane_recordings_per_t) > 0:
                    membrane_recordings = torch.stack(membrane_recordings_per_t)
                    membrane_recordings = membrane_recordings.permute(1, 0, 2)
                    membrane_timesteps = torch.stack(membrane_timesteps)
                    membrane_timesteps = membrane_timesteps.permute(1, 0)
                else:
                    membrane_recordings = []
                    membrane_timesteps = []
                y = torch.stack(ys)
                if params.timestep_size < 1:
                    # Convert event tensor to fixed timestep tensor
                    # This is necessary to have all logging and plotting work (which expects fixed timestep tensors)
                    # Only keep last membrane potential for a single timestep, for rate: keep all (spike) outputs, refractory period should make sure it is <= 1 per neuron per timestep
                    if params.variable_dt_save_output_with_dt != 1:
                        # Compute output for saving purposes
                        y_save_overwrite = event_tensor_to_fixed_timestep_tensor(
                            y,
                            timesteps,
                            timestep_size=params.variable_dt_save_output_with_dt,
                            max_n_timesteps=params.fix_sample_length_to//params.variable_dt_save_output_with_dt,
                            only_keep_last=(
                                params.loss_function_type
                                == LossFunctionType.LAST_MEMBRANE
                                or params.loss_function_type
                                == LossFunctionType.MAX_MEMBRANE
                            ),
                        )
                    # Compute output for loss and accuracy (always use dt=1 for this)
                    y = event_tensor_to_fixed_timestep_tensor( # This tensor can get very large (up to 4 MB per sample for N-MNIST)
                        y,
                        timesteps,
                        timestep_size=1,
                        max_n_timesteps=params.fix_sample_length_to,
                        only_keep_last=(
                            params.loss_function_type == LossFunctionType.LAST_MEMBRANE
                            or params.loss_function_type
                            == LossFunctionType.MAX_MEMBRANE
                        ),
                    )
                loss = params.loss_function(y, targets.long())
            accuracy = params.accuracy_function(y, targets)

            classification_latencies, correct_classification_latencies = net.get_latencies(targets)
            n_currents_to_spike, sum_excitatory_currents_to_spike, sum_inhibitory_currents_to_spike = net.get_dynamics()
            if y_save_overwrite is not None:
                y = y_save_overwrite
            return InferenceBatchResult(
                loss=loss.item(),
                accuracy=accuracy,
                batch_size=data.size(1),
                output_recordings=y,
                spike_counts=net.get_spike_counts(),
                classification_latencies=classification_latencies,
                correct_classification_latencies=correct_classification_latencies,
                n_currents_to_spike=n_currents_to_spike,
                sum_excitatory_currents_to_spike=sum_excitatory_currents_to_spike,
                sum_inhibitory_currents_to_spike=sum_inhibitory_currents_to_spike,
                membrane_recordings=membrane_recordings,
                membrane_timesteps=membrane_timesteps,
                labels=targets.cpu(),
            )
