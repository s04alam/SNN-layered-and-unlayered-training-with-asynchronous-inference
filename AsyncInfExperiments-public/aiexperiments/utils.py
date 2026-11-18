from typing import List
import torch
from snntorch import surrogate
import numpy as np


def load_weights_from_file(path, device) -> List[torch.Tensor]: # TODO: use this function in the future
    state_dict = torch.load(path)
    weights_per_layer = []
    for key in state_dict:
        key_split = key.split('.')
        if key_split[-1] == 'weight':
            weights_per_layer.append(state_dict[key].to(device))
    return weights_per_layer

def reduce_bins_in_dict(dict_of_histograms, new_num_bins):
    dict_of_histograms_reduced = {}
    for key in dict_of_histograms:
        if new_num_bins > len(dict_of_histograms[key]):
            raise ValueError(
                f"new_num_bins ({new_num_bins}) must be less than or equal to the number of bins in the histogram ({len(dict_of_histograms[key])})"
            )
        elif new_num_bins < len(dict_of_histograms[key]):
            dict_of_histograms_reduced[key] = reduce_bins(
                dict_of_histograms[key], new_num_bins
            )
        else:
            dict_of_histograms_reduced[key] = dict_of_histograms[key]
    return dict_of_histograms_reduced

# assumes bins are always 0 to 1
# floors the bin edges
def reduce_bins(histogram: torch.Tensor, new_num_bins):
    bin_counts = histogram
    current_bin_edges = torch.linspace(0, 1, len(histogram) + 1)

    # Calculate the new bin edges
    total_range = current_bin_edges[-1] - current_bin_edges[0]

    # Calculate the new bin index for each original bin
    new_bin_width = total_range / new_num_bins
    new_bin_idx = (
        ((current_bin_edges[:-1] - current_bin_edges[0]) / new_bin_width).floor().long()
    )

    # Scatter add the original bin counts to the new bin counts
    new_bin_counts = torch.zeros(new_num_bins, dtype=bin_counts.dtype).scatter_add(0, new_bin_idx, bin_counts)

    return new_bin_counts


# Transforms [n_batches, n_timesteps, tensor(n_samples_per_batch, n_neurons)] to tensor(n_timesteps, n_total_samples, n_neurons)
def combine_tensor_batches(recordings) -> torch.Tensor:
    combined_data = []
    for t in range(len(recordings[0])):  # iterate over timesteps
        # Collect tensors for current timestep from all batches
        tensors_at_t = [batch[t] for batch in recordings]
        # Stack them together
        combined_data.append(torch.cat(tensors_at_t, dim=0))
    
    # Stack along timesteps
    result = torch.stack(combined_data, dim=0)
    return result

# Transforms [n_batches, n_samples_per_batch, torch.Tensor(n_event_steps, n_neurons)] into [n_samples, torch.Tensor(n_event_steps, n_neurons)]
def combine_list_batches(recordings) -> List[torch.Tensor]:
    combined_data = []
    for batch in recordings:
        combined_data.extend(batch)
    return combined_data

# Transforms [n_batches, max_n_events, tensor(n_samples_per_batch, n_neurons)] to tensor(max_timesteps, n_total_samples, n_neurons)
# timesteps is of shape [max_n_events, n_batches]
def combine_event_batches(recordings, timesteps, max_t):
    combined_data = []
    for t in range(max_t):  # iterate over timesteps
        # Collect tensors for current timestep from all batches
        tensors_at_t = []
        for batch_index in range(len(recordings)):
            batch = recordings[batch_index]
            batch_timesteps = timesteps[batch_index]
            if t < batch_timesteps:
                tensors_at_t.append(batch[t])
            else:
                tensors_at_t.append(torch.zeros_like(batch[0]))
        # Stack them together
        combined_data.append(torch.cat(tensors_at_t, dim=0))
    
    # Stack along timesteps
    result = torch.stack(combined_data, dim=0)
    return result


# Converts a surrogate type string to a surrogate type object, may contain a parameter separated by a dash
# e.g. "atan-1.0" -> surrogate.atan(1.0)
# e.g. "fast_sigmoid" -> surrogate.fast_sigmoid()
# e.g. "" -> None
def surrogate_type_str_to_surrogate_type(s: str):
    s_split = s.split("-")  # for like atan-1.0
    s_name = s_split[0]
    if len(s_split) > 1:
        s_param = float(s_split[1])
    else:
        s_param = None

    if s_name == "atan":
        s_function = surrogate.atan
    elif s_name == "fast_sigmoid":
        s_function = surrogate.fast_sigmoid
    else:
        s_function = None

    if s_function is None:
        return None
    elif s_param is not None:
        return s_function(s_param)
    else:
        return s_function()


# input tensor: shape (n_timesteps, n_total_samples, n_neurons)
def normalize_neuron_spike_order(tensor: torch.Tensor) -> torch.Tensor:
    # Get the shape of the tensor
    n_timesteps, n_samples, _ = tensor.shape
    
    # Iterate over the timesteps and samples
    for i in range(n_timesteps):
        for j in range(n_samples):
            # Get the maximum order of the spikes in this sample
            max_order = tensor[i, j, :].max().item()
            min_order = tensor[i, j, :].min().item()
            order_range = max_order - min_order
            if order_range > 0:
                # Normalize values > 0
                tensor[i, j, tensor[i, j, :] > 0] /= order_range
                assert tensor[i, j, :].max().item() <= 1.0
                assert tensor[i, j, :].min().item() >= 0.0
                
    return tensor


def convert_np_array_list_to_tensor_list(arrays: List[np.ndarray]) -> List[torch.Tensor]:
    if len(arrays) == 0 or isinstance(arrays[0], torch.Tensor):
        return arrays
    else:
        return [torch.from_numpy(arr) for arr in arrays]