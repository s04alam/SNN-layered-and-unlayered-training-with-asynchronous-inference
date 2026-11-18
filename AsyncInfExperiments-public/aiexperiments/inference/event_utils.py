from typing import List, Tuple

from collections import namedtuple
import torch
import numpy as np

# event_tensor is of shape [max_n_events, batch_size, n_out]
# timesteps is of shape [max_n_events, batch_size]
# timestep_size is the size of a timestep in microseconds to be used for the conversion
# max_n_timesteps is the maximum number of timesteps in the output, each timestep will be of size timestep_size
# output will be of shape [max_n_timesteps, batch_size, n_out]
# the idea is to put the events in the correct timestep based on their time (which can be inferred from timesteps)
def event_tensor_to_fixed_timestep_tensor(
    event_tensor: torch.Tensor,
    timesteps: torch.Tensor,
    timestep_size: int,
    max_n_timesteps: int,
    only_keep_last: bool,
) -> torch.Tensor:
    max_n_events, batch_size, n_out = event_tensor.shape
    output = torch.zeros(
        (max_n_timesteps, batch_size, n_out), device=event_tensor.device
    )

    for batch in range(batch_size):
        current_timestep = 0

        for event_index in range(max_n_events):
            current_timestep += timesteps[
                event_index, batch
            ].item()  # current_timestep is not yet influenced by timestep_size

            if current_timestep >= max_n_timesteps * timestep_size:
                break
            target_index = int(current_timestep // timestep_size)

            if target_index < max_n_timesteps:
                if only_keep_last:
                    output[target_index, batch] = event_tensor[
                        event_index, batch
                    ]  # Only keep the last value
                else:
                    output[target_index, batch] += event_tensor[event_index, batch]

    return output


# ------------------------------- Event-driven inference utils -------------------------------
UnindexedEventWithValue = namedtuple("UnindexedEventWithValue", ["time", "value"])
EventWithValue = namedtuple("EventWithValue", ["index", "time", "value"])


# Will return shape [max_n_timesteps+1, 1, n_classes] (note: 1 is the batch size, fixed to 1)
def classification_events_to_array(
    events: List[EventWithValue], timestep_size, max_n_timesteps, n_classes
):
    # Initialize the tensor with zeros
    arr = np.zeros((max_n_timesteps + 1, 1, n_classes))

    # Iterate through the list of events
    for event in events:
        divide_by = timestep_size if timestep_size > 0 else 1
        arr[int(event.time // divide_by), 0, event.index] = event.value

    return arr


def event_order_recordings_to_tensor(
    event_order_recordings: List[Tuple[int, EventWithValue]],
    timestep_size,
    max_n_timesteps,
    neurons_per_layer: List[int],
    n_inputs: int,
) -> List[np.ndarray]:
    """
        Converts event recordings to an array of shape [max_n_timesteps, np.ndarray(1, n_neurons)]
        for network events and [max_n_timesteps, np.ndarray(1, n_inputs)] for input events.
        Expects data for one sample only.

        Parameters:
            event_order_recordings: shape [n_events, (layer_idx, event)]
            timestep_size: int - the size of a timestep
            max_n_timesteps: int - the maximum number of timesteps in the recording
            neurons_per_layer: List[int]
            n_inputs: int

        Returns:
            order_recordings: shape [max_n_timesteps, np.ndarray(1, n_inputs+n_neurons)]
    """

    order_recordings = []
    n_neurons = sum(neurons_per_layer)
    for i in range(max_n_timesteps + 1):
        order_recordings.append(np.zeros((1, n_inputs + n_neurons)))

    ts_event_count = {}
    for layer_idx, event in event_order_recordings:
        if timestep_size > 1:
            event_time_index = int(event.time // timestep_size)
        else:
            event_time_index = event.time
        if ts_event_count.get(event_time_index) is None:
            ts_event_count[event_time_index] = 1
        else:
            ts_event_count[event_time_index] += 1

        if layer_idx > 0:  # inside the network
            network_idx = (
                event.index + sum(neurons_per_layer[: (layer_idx - 1)]) + n_inputs
            )

            order_recordings[event_time_index][0][network_idx] = ts_event_count[
                event_time_index
            ]
        else:  # input layer
            order_recordings[event_time_index][0][event.index] = ts_event_count[
                event_time_index
            ]

    return order_recordings


def classification_events_to_output_recordings(
    events: List[EventWithValue], timestep_size, max_n_timesteps, n_neurons
) -> List[np.ndarray]:
    """
    Converts classification events from the event-driven simulator to output recordings in the correct shape.
    Output shape: [n_timesteps, tensor(1, n_neurons)]
    """
    arrs = []
    timestep_events = {i: [] for i in range(max_n_timesteps + 1)}
    events = sorted(events, key=lambda x: x.time)

    # Distribute events to corresponding timestep based on their time attribute
    for event in events:
        if timestep_size > 1:
            event_time_index = event.time // timestep_size
        else:
            event_time_index = event.time
        timestep_events[event_time_index].append(event)

    # Convert the events in each timestep to an array and add to the arrays list
    for i in range(max_n_timesteps + 1):
        # Create a zero array for this timestep
        arr = np.zeros((1, n_neurons))
        for event in timestep_events[i]:
            arr[0][event.index] = event.value
        # Add the array for this timestep to the arrays list
        arrs.append(arr)

    return arrs


def membrane_potential_recordings_to_output_recordings(
    ts: List[List[float]], ys: List[List[float]], timestep_size, max_n_timesteps
):
    """
    Converts membrane potential recordings from the event-driven simulator to output recordings in the correct shape.
    Output shape: [n_timesteps, np.ndarray(1, n_neurons)]

    Parameters:
        ts: List[List[float]] - recording timestamps, note that this may contain duplicates.
                                If this is the case, the last value in the list should be used.
        ys: List[List[float]] - membrane potential recordings
    """
    n_neurons = len(ts)
    arrays = [np.zeros((1, n_neurons)) for i in range(max_n_timesteps + 1)]
    for i in range(len(ts)):
        for j in range(len(ts[i])):
            if timestep_size > 1:
                event_time_index = int(ts[i][j] // timestep_size)
            else:
                event_time_index = int(ts[i][j])
            arrays[event_time_index][0][i] = ys[i][j]
    return arrays


def find_highest_index_for_which_value_smaller_than(arr, value):
    for idx in range(len(arr) - 1, -1, -1):
        if value > arr[idx]:
            return idx
    return None
