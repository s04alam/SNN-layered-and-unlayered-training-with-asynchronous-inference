from abc import abstractmethod, ABCMeta
from typing import List, Union
from aiexperiments.base_engine import BaseEngine, EngineParameters, StaticParameters
from aiexperiments.visualizing.recording.inference_recorder import (
    InferenceBatchResult,
    InferenceRecorder,
)
from aiexperiments.networks.base_network import (
    BaseNetwork,
    BaseNetworkBuilder,
    BaseNetworkBuilderParameters,
)
from tqdm import tqdm
import torch
import numpy as np

class InferenceParameters(EngineParameters):
    """
    Parameters for inference.

    Attributes:

    """

    def __init__(
        self,
        n_order_histogram_bins,
        limit_n_samples=None,
        track_order_recordings=True,
        track_membrane_recordings=False,
        pre_build_network=True, # Build the network before inference. If False, the network must manually be built during inference at every batch.
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_order_histogram_bins = n_order_histogram_bins
        self.limit_n_samples = limit_n_samples
        self.track_order_recordings = track_order_recordings
        self.track_membrane_recordings = track_membrane_recordings
        self.pre_build_network = pre_build_network
        if self.limit_n_samples == 0:
            self.limit_n_samples = None


class BaseInferenceEngine(BaseEngine, metaclass=ABCMeta):
    def __init__(self, static_params: StaticParameters):
        super().__init__(static_params)

    @abstractmethod
    def build_network(
        self,
        params: InferenceParameters,
        net_builder: BaseNetworkBuilder,
        net_builder_params: BaseNetworkBuilderParameters,
    ) -> BaseNetwork:
        pass

    @abstractmethod
    def infer_batch(
        self,
        net: Union[BaseNetwork, None],
        params: InferenceParameters,
        net_builder: BaseNetworkBuilder,
        net_builder_params: BaseNetworkBuilderParameters,
        data, # can be a tuple
        targets: torch.Tensor,
    ) -> Union[InferenceBatchResult, List[InferenceBatchResult]]:
        pass

    def init_recorder(
        self,
        params: InferenceParameters,
        net_builder_params: BaseNetworkBuilderParameters,
    ):
        return InferenceRecorder(
            n_bins=params.n_order_histogram_bins,
            n_inputs=net_builder_params.get_n_inputs(),
            track_order_recordings=params.track_order_recordings,
            track_membrane_recordings=params.track_membrane_recordings,
        )

    def infer(
        self,
        params: InferenceParameters,
        net_builder: BaseNetworkBuilder,
        net_builder_params: BaseNetworkBuilderParameters,
    ) -> InferenceRecorder:
        if self.static_params.verbose:
            print("Loading test data...")
        dataloader = self.load_data(
            params,
            is_training=False,
        )

        if params.pre_build_network:
            if self.static_params.verbose:
                print("Initializing network...")
            net = self.build_network(params, net_builder, net_builder_params)
        else:
            net = None

        if self.static_params.verbose:
            print("Testing...")
        recorder = self.init_recorder(params, net_builder_params)
        batch_iter = iter(dataloader)
        tqdm_total = None
        if params.limit_n_samples is not None:
            tqdm_total = int(params.limit_n_samples//params.batch_size)
            print("Limiting number of samples to", params.limit_n_samples)
        counter = 0
 
        for sample in tqdm(batch_iter, total=tqdm_total):
            if len(sample) == 2:
                data, targets = sample
            elif len(sample) == 3:
                data, timesteps, targets = sample
                data = (data, timesteps) # note that not every engine accepts this
            else:
                raise ValueError("Sample must be a tuple of length 2 or 3")
            test_result = self.infer_batch(
                net, params, net_builder, net_builder_params, data, targets
            )

            if isinstance(test_result, list):
                for result in test_result:
                    recorder.record(result)
            else:
                recorder.record(test_result)

            counter += 1
            if params.limit_n_samples is not None and counter >= int(np.ceil(params.limit_n_samples/params.batch_size)):
                break
        return recorder
