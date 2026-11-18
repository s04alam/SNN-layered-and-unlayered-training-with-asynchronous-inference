from enum import Enum
from aiexperiments.base_engine import StaticParameters
from aiexperiments.constants import DatasetProvider, LossFunctionType, OptimizerType
from aiexperiments.networks.base_network import (
    BaseNetworkBuilder,
    BaseNetworkBuilderParameters,
    BaseSNNTorchNetwork,
)
from aiexperiments.visualizing.metrics import ce_last_membrane_loss, l2_rate_sparsity, last_membrane_acc_fn, max_membrane_acc_fn
from aiexperiments.visualizing.recording.recorder import BatchResult
from aiexperiments.training.base_training import (
    BaseTrainingEngine,
    TrainingParameters,
)
import torch
import snntorch.functional as SF


class BPTTTrainingParameters(TrainingParameters):
    """
    Parameters for the BPTT training.

    Attributes:
        device: The device to use for training. Either 'cpu' or 'cuda'.
        loss_function_type: The loss function to use. All loss functions are supported.
        learning_rate: The learning rate to use.
        K: Number of time steps to process per weight update. Defaults to None, which means that the entire sequence is processed in one batch.
    """

    def __init__(
        self,
        device,
        loss_function_type: LossFunctionType,
        learning_rate=1e-3,
        weight_decay=0,
        spike_sparsity_loss_Lambda=None, # None means no spike sparsity loss
        K=None, # currently not set/used
        **kwargs,
    ):
        super().__init__(dataset_provider=DatasetProvider.TONIC_FRAME, **kwargs)
        self.device = device
        self.loss_function_type = loss_function_type
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
            self.loss_function = ce_last_membrane_loss()
            self.accuracy_function = last_membrane_acc_fn
        else:
            raise NotImplementedError(
                f"Loss function {loss_function_type} not implemented."
            )
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.K = K
        if spike_sparsity_loss_Lambda is not None:
            self.spike_sparsity_loss_fn = l2_rate_sparsity(Lambda=spike_sparsity_loss_Lambda)
        else:
            self.spike_sparsity_loss_fn = None


class BPTTTrainingEngine(BaseTrainingEngine):
    def __init__(self, static_params: StaticParameters):
        super().__init__(static_params)

    def build_network(
        self,
        params: BPTTTrainingParameters,
        net_builder: BaseNetworkBuilder,
        net_builder_params: BaseNetworkBuilderParameters,
    ) -> BaseSNNTorchNetwork:
        return net_builder.build_snntorch_network(net_builder_params).to(params.device)

    def setup_optimizer(
        self, net: BaseSNNTorchNetwork, params: BPTTTrainingParameters
    ) -> torch.optim.Optimizer:
        if params.optimizer_type == OptimizerType.ADAM:
            return torch.optim.Adam(
                net.parameters(),
                lr=params.learning_rate,
                weight_decay=params.weight_decay,
            )
        elif params.optimizer_type == OptimizerType.ADAMAX:
            return torch.optim.Adamax(
                net.parameters(),
                lr=params.learning_rate,
                weight_decay=params.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {params.optimizer_type} not implemented."
            )

    def train_batch(
        self,
        net: BaseSNNTorchNetwork,
        optimizer: torch.optim.Optimizer,
        params: BPTTTrainingParameters,
        data: torch.Tensor,
        targets: torch.Tensor,
    ) -> BatchResult:
        data = data.to(params.device)
        targets = targets.to(params.device)

        if params.K is not None and params.K > 0:
            data_slices = torch.split(data, params.K, dim=0)
        else:
            data_slices = (data,)
        loss_avg = 0
        regularization_loss_avg = 0
        accuracy = 0

        net.reset_state()
        for i_slice, data_slice in enumerate(data_slices):
            # Forward pass
            net.train()
            ys = []
            for i in range(data.size(0)):
                spk_out, mem_out = net(data[i])
                if params.loss_function_type == LossFunctionType.MAX_MEMBRANE or params.loss_function_type == LossFunctionType.LAST_MEMBRANE:
                    ys.append(mem_out)
                else:
                    ys.append(spk_out)
            y = torch.stack(ys)

            loss = params.loss_function(y, targets)
            loss_avg += loss.item() * (len(data_slice) / data.size(0))
            if i_slice == len(data_slices) - 1:
                accuracy = params.accuracy_function(y, targets)
            if params.spike_sparsity_loss_fn is not None:
                spike_sparsity_loss = params.spike_sparsity_loss_fn(net.get_spike_counts(), data.size(0), net.n_neurons, data.size(1))
                loss = loss + spike_sparsity_loss
                regularization_loss_avg += spike_sparsity_loss.item() * (len(data_slice) / data.size(0))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Detach memory from graph
            net.detach_state()
        return BatchResult(loss_avg, accuracy, data.size(1), regularization_loss=regularization_loss_avg)
