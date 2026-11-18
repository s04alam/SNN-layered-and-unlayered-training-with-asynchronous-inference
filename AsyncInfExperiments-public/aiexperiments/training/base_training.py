from abc import abstractmethod, ABCMeta
from aiexperiments.base_engine import BaseEngine, EngineParameters, StaticParameters
from aiexperiments.constants import OptimizerType
from aiexperiments.networks.base_network import (
    BaseNetwork,
    BaseNetworkBuilder,
    BaseNetworkBuilderParameters,
)
from aiexperiments.visualizing.recording.recorder import BatchResult, Recorder
from tqdm import tqdm
import os
import torch


class TrainingParameters(EngineParameters):
    """
    Parameters for an experiment applied during training. Note that even if only testing is performed, these parameters are still required.

    Attributes:
        num_epochs: The number of epochs to train for.
        save_network_parameters_to: The path to save network parameters to. If None, the network parameters will not be saved.
        load_train_recorder_from: The path to load the training recorder from. If None, the training recorder will be initialized.
        save_train_recorder_to: The path to save the training recorder to. If None, the training recorder will not be saved.
        backup: Whether to backup the network parameters and training recorder after each epoch. Path must be specified for both.
        num_repeats: The number of times to repeat the training. The best model parameters will be saved.
    """

    def __init__(
        self,
        num_epochs,
        optimizer_type: OptimizerType,
        save_network_parameters_to=None,
        load_train_recorder_from=None,
        save_train_recorder_to=None,
        backup=False,
        num_repeats=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.timestep_size <= 0:
            raise ValueError("Timestep size must be greater than 0 for training.")
        self.num_epochs = num_epochs
        self.save_network_parameters_to = save_network_parameters_to
        self.load_train_recorder_from = load_train_recorder_from
        self.save_train_recorder_to = save_train_recorder_to
        self.backup = backup
        self.num_repeats = num_repeats
        self.optimizer_type = optimizer_type


class BaseTrainingEngine(BaseEngine, metaclass=ABCMeta):
    def __init__(self, static_params: StaticParameters):
        super().__init__(static_params)

    @abstractmethod
    def build_network(
        self,
        params: TrainingParameters,
        net_builder: BaseNetworkBuilder,
        net_builder_params: BaseNetworkBuilderParameters,
    ) -> BaseNetwork:
        pass

    @abstractmethod
    def setup_optimizer(self, net: BaseNetwork, params: TrainingParameters):
        pass

    @abstractmethod
    def train_batch(
        self,
        net: BaseNetwork,
        optimizer,
        params: TrainingParameters,
        data: torch.Tensor,
        targets: torch.Tensor,
    ) -> BatchResult:
        pass

    def train(
        self,
        params: TrainingParameters,
        net_builder: BaseNetworkBuilder,
        net_builder_params: BaseNetworkBuilderParameters,
    ) -> Recorder:
        best_accuracy = 0
        best_recorder = None
        best_net = None

        for repeat in range(params.num_repeats):
            if self.static_params.verbose:
                print(f"Repeat: {repeat + 1}/{params.num_repeats}")
                print("Loading training data...")

            dataloader = self.load_data(
                params,
                is_training=True,
            )

            if self.static_params.verbose:
                print("Initializing network...")
            net = self.build_network(params, net_builder, net_builder_params)
            optimizer = self.setup_optimizer(net, params)

            if self.static_params.verbose:
                print("Training...")
            if params.load_train_recorder_from is not None:
                recorder = Recorder.load(params.load_train_recorder_from)
            else:
                recorder = Recorder()
            for epoch in range(params.num_epochs):
                batch_iter = iter(dataloader)
                for sample in tqdm(batch_iter):
                    if len(sample) != 2:
                        raise ValueError(
                            "Sample must be a tuple of (data, targets) for training."
                        )
                    data, targets = sample
                    train_result = self.train_batch(
                        net, optimizer, params, data, targets
                    )
                    recorder.record(train_result)
                if len(recorder.losses) > 0:
                    print("Last loss:", str(recorder.losses[-1]))
                if params.backup:
                    if self.static_params.verbose:
                        print("Backing up..., epoch: " + str(epoch + 1))
                    if params.save_network_parameters_to is not None:
                        net.save_parameters(
                            params.save_network_parameters_to + ".backup"
                        )
                    if params.save_train_recorder_to is not None:
                        recorder.save(params.save_train_recorder_to + ".backup")

            # Compare the accuracy and update the best_model_params and best_recorder if needed
            current_accuracy = recorder.accuracies[-1]
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_recorder = recorder
                best_net = net

            # Remove backups
            if (
                params.save_network_parameters_to is not None
                and params.backup
                and os.path.exists(params.save_network_parameters_to + ".backup")
            ):
                os.remove(params.save_network_parameters_to + ".backup")
            if (
                params.save_train_recorder_to is not None
                and params.backup
                and os.path.exists(params.save_train_recorder_to + ".backup")
            ):
                os.remove(params.save_train_recorder_to + ".backup")

        # Save parameters
        if params.save_network_parameters_to is not None:
            best_net.save_parameters(params.save_network_parameters_to)

        # Save recorder
        if params.save_train_recorder_to is not None:
            best_recorder.save(params.save_train_recorder_to)

        return best_recorder
