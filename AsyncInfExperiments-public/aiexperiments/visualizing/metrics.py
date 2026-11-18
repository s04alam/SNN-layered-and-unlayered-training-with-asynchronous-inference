import torch
from scipy.special import softmax
import numpy as np
import snntorch.functional as SF

class l2_rate_sparsity:
    def __init__(self, Lambda=1e-5):
        self.Lambda = Lambda
        self.__name__ = "l2_rate_sparsity"

    def __call__(self, spk_count, n_timesteps, n_neurons, batch_size):
        return self.Lambda * (torch.sum(spk_count) / (n_timesteps * n_neurons * batch_size))


def max_membrane_acc_fn(mem_out, targets):
    # Find the index of the maximum membrane potential for each neuron
    max_vals, preds = torch.max(mem_out, dim=-1)
    # Find the index of the maximum value over time for each neuron
    _, max_time_indices = torch.max(max_vals, dim=0)
    # Calculate accuracy
    acc = (preds[max_time_indices] == targets).float().mean()

    return acc


class ce_last_membrane_loss(SF.ce_max_membrane_loss):
    def __init__(self):
        super().__init__()
        self.__name__ = "ce_last_membrane_loss"

    def __call__(self, mem_out, targets):
        last_mem = mem_out[-1]
        last_mem = last_mem.unsqueeze(0)
        loss = super().__call__(last_mem, targets)
        return loss


class ce_last_nonzero_membrane_loss(SF.ce_max_membrane_loss):
    def __init__(self):
        super().__init__()
        self.__name__ = "ce_last_nonzero_membrane_loss"

    def __call__(self, mem_out, targets):
        timesteps, batch_size, classes_out = mem_out.shape

        # Find the last nonzero timestep for each sample in the batch
        last_nonzero = torch.zeros(batch_size, dtype=torch.long)
        for t in range(timesteps):
            nonzero_mask = mem_out[t].any(dim=-1)
            last_nonzero[nonzero_mask] = t

        # Gather the last nonzero membrane potentials for each sample in the batch
        last_nonzero_membrane_potentials = mem_out[
            last_nonzero, torch.arange(batch_size)
        ]
        last_nonzero_membrane_potentials = last_nonzero_membrane_potentials.unsqueeze(0)

        # Calculate loss using the last nonzero membrane potentials
        loss = super().__call__(last_nonzero_membrane_potentials, targets)

        return loss


def last_membrane_acc_fn(mem_out, targets):
    _, preds = torch.max(mem_out[-1], dim=-1)
    acc = (preds == targets).float().mean()
    return acc


def last_nonzero_membrane_acc_fn(mem_out, targets):
    timesteps, batch_size, classes_out = mem_out.shape

    # Find the last nonzero timestep for each sample in the batch
    last_nonzero = torch.zeros(batch_size, dtype=torch.long)
    for t in range(timesteps):
        nonzero_mask = mem_out[t].any(dim=-1)
        last_nonzero[nonzero_mask] = t

    # Gather the last nonzero membrane potentials for each sample in the batch
    last_nonzero_membrane_potentials = mem_out[last_nonzero, torch.arange(batch_size)]

    # Calculate accuracy using the last nonzero membrane potentials
    _, preds = torch.max(last_nonzero_membrane_potentials, dim=-1)
    acc = (preds == targets).float().mean()

    return acc


def compute_normalized_order_overlap(histograms):
    """
    Computes the average pairwise histogram overlap for each time bin.

    Parameters:
    - histograms: dict with neuron indices as keys and their histograms as values

    Returns:
    - A list of average overlaps for each time bin
    """
    hist_tensor = torch.stack(list(histograms.values()), dim=0)
    n_neurons = hist_tensor.shape[0]
    indices = torch.combinations(torch.arange(n_neurons), r=2)
    min_hist = torch.min(hist_tensor[indices[:, 0]], hist_tensor[indices[:, 1]])
    overlaps = min_hist.sum(dim=0) / (n_neurons * (n_neurons - 1) / 2)

    return overlaps.numpy()



def compute_mean_confidence_true_class(
    output_recordings, # (n_timesteps, n_total_samples, n_neurons)
    labels, # (n_total_samples)
): # output shape: (n_total_samples)
    softmax_outputs = softmax(output_recordings, axis=2)
    relevant_softmax_outputs = softmax_outputs[np.arange(output_recordings.shape[0])[:, None], np.arange(output_recordings.shape[1]), labels]
    mean_confidences = relevant_softmax_outputs.mean(axis=0)
    return mean_confidences
    

def compute_max_confidence_true_class(
    output_recordings,
    labels
):
    softmax_outputs = softmax(output_recordings, axis=2)
    relevant_softmax_outputs = softmax_outputs[np.arange(output_recordings.shape[0])[:, None], np.arange(output_recordings.shape[1]), labels]
    max_confidences = relevant_softmax_outputs.max(axis=0)
    return max_confidences


def compute_latency(
    output_recordings,
    inference_timestep_size: int, # in simulation timesteps (same as n_timesteps in output_recordings)
    latency_timestep_size: int, # in simulation timesteps (same as n_timesteps in output_recordings)
    max_confidence_mean=0.9,
    stability_duration=3, # in latency_timesteps, so latency_timestep_size * stability_duration in simulation timesteps
):
    assert inference_timestep_size >= 1, "inference_timestep_size must be >= 1"
    assert latency_timestep_size >= 1, "latency_timestep_size must be >= 1"
    n_timesteps, _, _ = output_recordings.shape

    confidences = [np.max(softmax(output_recordings[t], axis=1)) for t in range(0, n_timesteps, latency_timestep_size)]
    mean_confidence = min(np.mean(confidences)*0.95 + np.std(confidences) * 0.9, max_confidence_mean)
    threshold = mean_confidence

    consecutive_count = 0
    for confidence, t in zip(confidences, range(0, n_timesteps, latency_timestep_size)):
        if confidence >= threshold:
            consecutive_count += 1
            if consecutive_count >= stability_duration:
                return (t+1) * inference_timestep_size # Return the latency of the first of the consecutive timesteps
        else:
            consecutive_count = 0  # Reset count if confidence drops

    return None  # No timesteps met the criteria


## NP loss functions
# Numpy equivalent of log_softmax
def log_softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return np.log(e_x / e_x.sum(axis=-1, keepdims=True))

# Numpy equivalent of NLLLoss
def nll_loss(log_p_y, targets):
    return -log_p_y[np.arange(len(targets)), targets].mean()


class ce_rate_loss_numpy():
    """Cross Entropy Spike Rate Loss for numpy arrays.
    """

    def __init__(self):
        self.__name__ = "ce_rate_loss_numpy"

    def __call__(self, spk_out, targets):
        num_steps = spk_out.shape[0]
        
        log_p_y = log_softmax(spk_out)
        loss = 0.0

        for step in range(num_steps):
            loss += nll_loss(log_p_y[step], targets)

        return loss / num_steps
    

class ce_count_loss_numpy():
    """Cross Entropy Spike Count Loss for numpy arrays.
    """

    def __init__(self):
        self.__name__ = "ce_count_loss_numpy"

    def __call__(self, spk_out, targets):
        spike_count = np.sum(spk_out, axis=0) 

        log_p_y = log_softmax(spike_count)
        loss = nll_loss(log_p_y, targets)

        return loss
    

class ce_max_membrane_loss_numpy():
    """Cross Entropy Max Membrane Loss for numpy arrays.
    """

    def __init__(self):
        self.__name__ = "ce_max_membrane_loss_numpy"

    def __call__(self, mem_out, targets):
        max_mem_out = np.max(mem_out, axis=0)

        log_p_y = log_softmax(max_mem_out)
        loss = nll_loss(log_p_y, targets)

        return loss