from aiexperiments.utils import combine_tensor_batches, convert_np_array_list_to_tensor_list, normalize_neuron_spike_order
from aiexperiments.visualizing.recording.recorder import BatchResult, Recorder
import torch
import numpy as np


class InferenceBatchResult(BatchResult):

    """
    output_recordings: shape [n_timesteps, tensor(n_samples_per_batch, n_neurons))]
                        for each timestep, contains the output of the network
    order_recordings: shape [n_timesteps, tensor(n_samples_per_batch, n_inputs+n_neurons)]
                        (optional, since not all inference methods can record this)
                        for each timestep, contains the order in which neurons fired
    membrane_recordings: shape [n_samples_per_batch, torch.Tensor(n_event_steps, n_neurons)]
                        (optional, since not all inference methods can record this)
                        for each step (triggered by an event), contains the membrane potential of each neuron
                        note that for a matrix-like inference method, n_event_steps = n_timesteps
    membrane_timesteps: shape [n_samples_per_batch, torch.Tensor(n_event_steps)]
                        (optional, since not all inference methods can record this)
                        for each step (triggered by an event), contains the timestep of each membrane recording
    """
    # labels: shape [n_samples_per_batch]
    # spike_counts: shape [n_samples_per_batch]
    # classification_latencies: any length
    # correct_classification_latencies: any length

    def __init__(
        self,
        output_recordings,
        spike_counts,
        classification_latencies,
        correct_classification_latencies,
        labels,
        *args,
        order_recordings=[],
        membrane_recordings=[],
        membrane_timesteps=[],
        n_currents_to_spike=[],
        sum_excitatory_currents_to_spike=[],
        sum_inhibitory_currents_to_spike=[],
        **kwargs,
    ):
        assert isinstance(order_recordings, list)
        if len(order_recordings) > 0:
            assert isinstance(order_recordings[0], torch.Tensor) or isinstance(
                order_recordings[0], np.ndarray
            )
            assert len(order_recordings[0].shape) == 2
        if len(output_recordings) > 0:
            assert isinstance(output_recordings[0], torch.Tensor) or isinstance(
                output_recordings[0], np.ndarray
            )
            assert len(output_recordings[0].shape) == 2
        if len(membrane_recordings) > 0:
            assert isinstance(membrane_recordings[0], torch.Tensor) or isinstance(
                membrane_recordings[0], np.ndarray
            )
            assert len(membrane_recordings[0].shape) == 2
        super().__init__(*args, **kwargs)
        self.order_recordings = []
        for recording in order_recordings:
            if isinstance(recording, torch.Tensor):
                recording = recording.cpu()
            self.order_recordings.append(
                recording
            )  # order recordings are always tensors
        self.output_recordings = []
        for recording in output_recordings:
            if isinstance(recording, torch.Tensor):
                recording = recording.cpu()
            self.output_recordings.append(recording)
        self.membrane_recordings = []
        for recording in membrane_recordings:
            if isinstance(recording, torch.Tensor):
                recording = recording.cpu()
            self.membrane_recordings.append(recording)
        self.membrane_timesteps = []
        for membrane_timestep in membrane_timesteps:
            if isinstance(recording, torch.Tensor):
                recording = recording.cpu()
            self.membrane_timesteps.append(membrane_timestep)
        self.labels = labels
        self.spike_counts = spike_counts
        self.classification_latencies = classification_latencies
        self.correct_classification_latencies = correct_classification_latencies
        self.n_currents_to_spike = n_currents_to_spike
        self.sum_excitatory_currents_to_spike = sum_excitatory_currents_to_spike
        self.sum_inhibitory_currents_to_spike = sum_inhibitory_currents_to_spike

class InferenceRecorder(Recorder):
    def __init__(
        self,
        n_bins,  # number of bins to use for the histogram, only used if track_order_recordings is False
        n_inputs, # number of inputs
        *args,
        track_order_recordings=True,  # if true, will save exact order of spikes for each neuron, otherwise will save a histogram of the order of spikes for each neuron
        track_membrane_recordings=False,  # if true, will save membrane potential for each neuron
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.track_order_recordings = track_order_recordings
        self.n_inputs = n_inputs
        if track_order_recordings:
            self.all_order_recordings = []  # shape [n_batches, n_timesteps, tensor(n_samples_per_batch, n_neurons+n_inputs)]
        else:
            self.n_bins = n_bins
            self.order_histograms = {}  # neuron index -> histogram
        self.all_labels = []  # shape [n_batches, n_samples_per_batch]
        self.all_spike_counts = []  # shape [n_batches, n_samples_per_batch]
        self.all_output_recordings = [] # shape [n_batches, n_timesteps, tensor(n_samples_per_batch, n_neurons)]
        self.spike_counts = []  # shape [n_batches]
        self.track_membrane_recordings = track_membrane_recordings
        if track_membrane_recordings:
            self.all_membrane_recordings = [] # shape [n_batches, n_samples_per_batch, tensor(n_event_steps, n_neurons)]
            self.all_membrane_timesteps = [] # shape [n_batches, n_samples_per_batch, tensor(n_event_steps)]
        self.classification_latencies = []
        self.correct_classification_latencies = []
        self.n_currents_to_spike = []
        self.sum_excitatory_currents_to_spike = []
        self.sum_inhibitory_currents_to_spike = []


    def record(self, result: InferenceBatchResult):
        super().record(result)
        # Anything in np.ndarray format should be converted to torch.Tensor
        # TODO: get rid of torch.Tensor format
        if isinstance(result.labels, np.ndarray): # TODO: get rid of this
            self.all_labels.append(torch.from_numpy(result.labels))
        else:
            self.all_labels.append(result.labels)
        if isinstance(result.spike_counts, np.ndarray): # TODO: get rid of this
            self.all_spike_counts.append(torch.from_numpy(result.spike_counts))
        else:
            self.all_spike_counts.append(result.spike_counts)
        self.correct_classification_latencies.extend(result.correct_classification_latencies)
        self.classification_latencies.extend(result.classification_latencies)
        self.n_currents_to_spike.extend(result.n_currents_to_spike)
        self.sum_excitatory_currents_to_spike.extend(result.sum_excitatory_currents_to_spike)
        self.sum_inhibitory_currents_to_spike.extend(result.sum_inhibitory_currents_to_spike)
        self.all_output_recordings.append(convert_np_array_list_to_tensor_list(result.output_recordings))
        # save membrane recordings (if applicable)
        if self.track_membrane_recordings:
            self.all_membrane_recordings.append(convert_np_array_list_to_tensor_list(result.membrane_recordings))
            self.all_membrane_timesteps.append(convert_np_array_list_to_tensor_list(result.membrane_timesteps))
        # save order recordings or histograms
        if self.track_order_recordings:  # save exact order of spikes
            self.all_order_recordings.append(convert_np_array_list_to_tensor_list(result.order_recordings))
        else:  # save histograms
            if len(result.order_recordings) > 0:
                # Combine order recordings from all timesteps into one tensor and then remove the inputs
                combined_order_recordings = combine_tensor_batches([convert_np_array_list_to_tensor_list(result.order_recordings)])[:, :, self.n_inputs:]
                normalized_order_recordings = normalize_neuron_spike_order(
                    combined_order_recordings
                )
                for neuron_index in range(normalized_order_recordings.shape[2]): # also includes inputs
                    neuron_data = normalized_order_recordings[:, :, neuron_index]
                    non_zero_data = neuron_data[neuron_data > 0].cpu()
                    if len(non_zero_data) > 0:
                        histogram = torch.histogram(
                            non_zero_data, bins=self.n_bins, range=(0, 1)
                        ).hist
                        if neuron_index not in self.order_histograms:
                            self.order_histograms[neuron_index] = histogram
                        else:
                            self.order_histograms[neuron_index] += histogram

    def get_normalized_order_recordings(self) -> torch.Tensor:
        if not self.track_order_recordings:
            raise Exception(
                "Cannot get normalized order recordings if order recordings were not saved"
            )
        cat_order_recordings = combine_tensor_batches(self.all_order_recordings)[:, :, self.n_inputs:]
        normalized_order_recordings = normalize_neuron_spike_order(cat_order_recordings)
        return normalized_order_recordings

    def get_per_neuron_order_histogram(self, n_bins: int, density: bool = False):
        if self.track_order_recordings:  # compute histograms from order recordings
            normalized_order_recordings = self.get_normalized_order_recordings()
            histograms = {}
            for neuron_index in range(normalized_order_recordings.shape[2]):
                neuron_data = normalized_order_recordings[:, :, neuron_index]
                assert neuron_data.max() <= 1
                non_zero_data = neuron_data[neuron_data > 0].cpu()
                if len(non_zero_data) > 0:
                    histogram = torch.histogram(
                        non_zero_data, bins=n_bins, range=(0, 1), density=density
                    ).hist
                    histograms[neuron_index] = histogram
                else:
                    histograms[neuron_index] = torch.zeros(n_bins)
            return histograms
        else:  # return the histograms that were already computed
            if n_bins != self.n_bins:
                print(
                    f"Warning: requested {n_bins} bins, but only {self.n_bins} bins were used to compute the histograms"
                )
            if density:
                # Normalize the histograms, for each neuron in self.order_histograms
                normalized_histograms = {}
                for neuron_index in self.order_histograms:
                    histogram = self.order_histograms[neuron_index]
                    normalized_histograms[neuron_index] = histogram / histogram.sum()
                return normalized_histograms
            else:
                return self.order_histograms
