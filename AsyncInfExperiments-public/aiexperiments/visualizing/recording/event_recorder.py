from typing import List, Tuple, Dict
from aiexperiments.inference.event_utils import (
    EventWithValue,
    event_order_recordings_to_tensor,
)
from aiexperiments.visualizing.recording.inference_recorder import (
    InferenceBatchResult,
    InferenceRecorder,
)
import numpy as np


class EventBatchResult(InferenceBatchResult):
    def __init__(
        self,
        output_recordings: List[np.ndarray],
        event_order_recordings: List[Tuple[int, EventWithValue]],
        membrane_recordings: List[np.ndarray],
        membrane_timesteps: List[int],
        timestep_size,
        max_n_timesteps,
        neurons_per_layer: List[int],
        n_inputs: int,
        n_currents_to_spike: np.ndarray, # shape [n_neurons]
        classification_latencies: List[float], # any length
        correct_classification_latencies: List[float], # any length
        sum_excitatory_currents_to_spike: np.ndarray, # shape [n_neurons]
        sum_inhibitory_currents_to_spike: np.ndarray, # shape [n_neurons]
        currents_histogram: np.ndarray, # shape [n_neurons, n_bins]
        currents_for_spike_histogram: np.ndarray, # shape [n_neurons, n_bins]
        *args,
        priorities_given: List[float] = [],
        **kwargs,
    ):
        assert isinstance(event_order_recordings, list)
        assert isinstance(event_order_recordings[0], tuple)
        assert isinstance(event_order_recordings[0][0], int)
        assert isinstance(event_order_recordings[0][1], EventWithValue)
        order_recordings = event_order_recordings_to_tensor(
            event_order_recordings,
            timestep_size,
            max_n_timesteps,
            neurons_per_layer,
            n_inputs
        )
        super().__init__(
            *args,
            output_recordings=output_recordings,
            classification_latencies=classification_latencies,
            correct_classification_latencies=correct_classification_latencies,
            order_recordings=order_recordings,
            membrane_recordings=membrane_recordings,
            membrane_timesteps=membrane_timesteps,
            n_currents_to_spike=n_currents_to_spike,
            sum_excitatory_currents_to_spike=sum_excitatory_currents_to_spike,
            sum_inhibitory_currents_to_spike=sum_inhibitory_currents_to_spike,
            **kwargs,
        )
        self.event_order_recordings = event_order_recordings
        self.priorities_given = priorities_given
        self.currents_histogram = currents_histogram
        self.currents_for_spike_histogram = currents_for_spike_histogram


class EventRecorder(InferenceRecorder):
    event_order_recordings: List[List[Tuple[int, EventWithValue]]]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_order_recordings = []
        self.priorities_given = []
        self.currents_histogram = []
        self.currents_for_spike_histogram = []

    def record(self, result: EventBatchResult):
        super().record(result)
        self.event_order_recordings.append(result.event_order_recordings)
        self.priorities_given.append(result.priorities_given)
        self.currents_histogram.append(result.currents_histogram)
        self.currents_for_spike_histogram.append(result.currents_for_spike_histogram)
