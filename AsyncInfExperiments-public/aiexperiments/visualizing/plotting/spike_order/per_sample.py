
from typing import Dict, List, Tuple
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from aiexperiments.inference.event_utils import EventWithValue
import math
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FuncFormatter

def get_layer_index(neuron_index: int, neurons_per_layer: List[int], n_inputs) -> int:
    _neurons_per_layer = [n_inputs] + list(neurons_per_layer) # add the input layer
    remaining_neurons = neuron_index
    for i, layer_size in enumerate(_neurons_per_layer):
        if remaining_neurons < layer_size:
            return i
        remaining_neurons -= layer_size
    return len(_neurons_per_layer) - 1

def plot_event_order_recordings_from_tensor(
    event_order_recordings_tensor: torch.Tensor,  # shape (n_timesteps, n_neurons+n_inputs)
    neurons_per_layer: List[int],
    n_inputs: int,
    timestep_size: int,
    show_inputs: bool,
    show_output: bool,
    title: str,
    weights: List[torch.Tensor] = None,
    figsize=(12, 5),
):
    # For each timestep, we have a tensor like tensor([1., 0., 0., 0., 0., 0., 4., 0., 0., 0., 2., 0., 6., 5., 7., 3., 0., 0., 0., 0.]), in this example for n_neurons=20
    # The goal is to convert this to a list of Tuple[int, EventWithValue], where the first element is the target layer index (based on neurons_per_layer and n_inputs)
    # and the second element is the event (event.time (given by the timestep), event.index (within a layer), event.value (always 1)). The list should be sorted by the values
    # given in the tensor as shown in the example above.
    event_order_recordings = []
    timestep_size = timestep_size if timestep_size > 0 else 1

    for timestep, recording in enumerate(event_order_recordings_tensor):
        for neuron_index, value in enumerate(
            recording
        ):  # TODO: fix that neuron_index must be corrected for its layer
            if not math.isnan(value) and value != 0:
                layer_index = get_layer_index(
                    neuron_index, neurons_per_layer, n_inputs
                ) # 0 = input layer, 1 = first hidden layer, etc.
                neuron_index_in_layer = neuron_index - sum(
                    neurons_per_layer[:layer_index]
                )
                order_index = value.item()
                event = EventWithValue(
                    time=timestep*timestep_size, index=neuron_index_in_layer, value=1
                )
                event_order_recordings.append((layer_index, order_index, event))

    event_order_recordings.sort(key=lambda x: (x[2].time, x[1]))
    # throw away the order_index
    event_order_recordings = [
        (layer_index, event) for layer_index, _, event in event_order_recordings
    ]
    # assert that all events indices are less than np.sum(neurons_per_layer) + n_inputs and non-negative
    assert all(
        [event.index >= 0 and event.index < sum(neurons_per_layer) + n_inputs for _, event in event_order_recordings]
    )

    # Call the plotting function
    plot_event_order_recordings(
        event_order_recordings,
        neurons_per_layer,
        n_inputs,
        show_inputs,
        show_output,
        title,
        weights=weights,
        figsize=figsize,
    )

def plot_event_order_recordings(
    event_order_recordings: List[Tuple[int, EventWithValue]], # (target_layer_index, event), where target_layer_index = 0 means input events
    neurons_per_layer: List[int],
    n_inputs: int,  # 0 means that the input layer is not included
    show_inputs: bool,
    show_output: bool,
    title: str,
    weights: List[torch.Tensor] = None,
    figsize=(12, 5),
):
    plt.figure(figsize=figsize)
    # Step 1: Create a dictionary to hold the event order indices
    # key: (timestamp), value: list of network indices, in order of event arrival (note that same network index can appear multiple times)
    event_order_dict: Dict[int, List[int, int]] = {}

    for layer_idx, event in event_order_recordings:
        if layer_idx == 0 and not show_inputs:
            continue

        if layer_idx == 0:
            network_idx = event.index
        else:
            network_idx = (
                sum(neurons_per_layer[: layer_idx]) + event.index
            )
            if not show_inputs:
                network_idx -= n_inputs

        if event_order_dict.get(event.time) is None:
            event_order_dict[event.time] = []

        event_value = event.value
        if weights is not None and layer_idx > 0 and layer_idx < len(neurons_per_layer):
            neuron_idx_in_layer = network_idx - sum(neurons_per_layer[:layer_idx-1])
            event_value *= weights[layer_idx][:, neuron_idx_in_layer].sum().item()

        event_order_dict[event.time].append((network_idx, event_value))
    # from here on, consider n_inputs to be 0 if we are not showing inputs
    if not show_inputs:
        n_inputs = 0

    # Step 2: Find the range of timestamps and layer indices
    timestamps = list(event_order_dict.keys())
    timestamps.sort()
    timestamp_ticks = []
    for timestamp in timestamps:
        timestamp_ticks.extend(
            [i for i in range(1, len(event_order_dict[timestamp]) + 1)]
        )
    x_max = len(timestamp_ticks)
    x_min = 0
    y_max = sum(neurons_per_layer) + n_inputs + 1
    y_min = 0
    yticks = [
        sum(neurons_per_layer[:layer_idx]) + n_inputs
        for layer_idx in range(0, len(neurons_per_layer) - 1)
    ]
    ytick_labels = [f"Hidden {i}" for i in range(1, len(yticks) + 1)]
    if n_inputs > 0:
        yticks.insert(0, 0)
        ytick_labels.insert(0, "Input")
    assert len(yticks) == len(ytick_labels)
    if (
        not show_output
    ):
        y_max -= neurons_per_layer[-1]
    else:
        yticks.append(yticks[-1] + neurons_per_layer[-2])
        ytick_labels.append("Output")

    plt.ylim(0, y_max - 1)
    labels = plt.yticks(yticks, ytick_labels)
    for label in labels[1]:
        label.set_x(-0.025)

    # Step 3: Create a 2D array to hold the counts
    order_matrix = np.zeros(
        (y_max - y_min, x_max - x_min)
    )

    events_seen = 0
    timestamp_borders = []  # by event index
    for (timestamp), events in event_order_dict.items():
        timestamp_borders.append(events_seen)
        for network_idx, event_value in events:
            y_idx = network_idx - y_min
            x_idx = events_seen
            events_seen += 1
            order_matrix[y_idx, x_idx] = event_value
    labels = plt.xticks(
        timestamp_borders, [f"{timestamp}" for timestamp in timestamps], rotation=0
    )
    for label in labels[1]:
        label.set_y(-0.01)

    # Step 4: Create a heatmap using matplotlib
    plt.grid()
    x_values = []
    y_values = []
    for i, row in enumerate(order_matrix):
        for j, value in enumerate(row):
            x_values.append(
                x_min + (x_max - x_min) * j / len(row)
            )
            y_values.append(
                y_min
                + (y_max - y_min) * i / len(order_matrix)
            )

    cmap = 'binary' if weights is None else 'viridis'
    plt.imshow(
        order_matrix,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
    )
    if weights is not None:
        plt.colorbar(label="Spike weights summed", orientation="horizontal") 

    # Step 5: Set the labels for the axes
    plt.xlabel("Time and ordering")
    plt.ylabel("Neuron index")

    # Step 6: Add minor ticks for each possible timestamp and neuron index
    ax = plt.gca()
    ax.set_yticks(
        [i for i in range(y_min, y_max)], minor=True
    )
    ax.tick_params(axis='x', which='both', length=0)

    # Step 7: Label each minor tick on the y-axis with the neuron index
    ax.yaxis.set_minor_locator(
        ticker.FixedLocator(
            [i - 0.5 for i in range(y_min, y_max)]
        )
    )
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, _: int(x)))

    # Break the x-axis at each timestamp
    break_positions = [timestamp_borders[i] for i in range(1, len(timestamps))]
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for pos in break_positions:
        # Position the "//" text at the center of the break position
        plt.text(pos, 0, "//", ha='center', va='center', color="black", transform=transform, 
                fontsize='large', clip_on=False)


    # Display the plot
    plt.title(title)
    plt.show()

def plot_membrane_potentials(
    membrane_potentials: torch.Tensor, # shape [n_timesteps, n_neurons]
    timesteps: torch.Tensor, # shape [n_timesteps]
    neurons_per_layer: np.ndarray,
    title: str,
    t_start: int = 0,
    figsize=(12, 5),
    xlim=None,
):
    membrane_potentials = membrane_potentials.t().numpy()
    timesteps = timesteps.numpy()

    plt.figure(figsize=figsize)
    plt.imshow(membrane_potentials, aspect='auto', cmap='viridis', vmin=-2, vmax=2, interpolation='none')
    plt.colorbar()

    timestamp_labels, timestamp_borders = np.unique(timesteps, return_index=True)
    timestamp_borders = timestamp_borders - .5
    labels = plt.xticks(timestamp_borders, timestamp_labels + t_start)
    for label in labels[1]:
        label.set_y(-0.01)

    # Calculate the position for the labels to be at the start of each layer
    label_positions = np.cumsum(neurons_per_layer) - neurons_per_layer

    # Adjust the label positions to not clash with the minor tick marks
    label_offsets = label_positions - .5  # slight offset to ensure visibility

    # Define layer labels
    layer_labels = [f"Hidden {i}" for i in range(1, len(label_offsets))] + ['Output']

    # Set y-ticks with the offset label positions
    labels = plt.yticks(label_offsets, layer_labels)
    for label in labels[1]:
        label.set_x(-0.025)

    # Invert y-axis
    plt.gca().invert_yaxis()

    # Set up minor ticks for each neuron index
    ax = plt.gca()
    ax.set_yticks(np.arange(0, np.sum(neurons_per_layer)), minor=True)
    ax.tick_params(axis='x', which='both', length=0)

    # Function to format minor tick labels
    def minor_tick_format(x, pos):
        return f"{int(x)}"

    ax.yaxis.set_minor_formatter(FuncFormatter(minor_tick_format))

    if xlim is not None:
        use_xlim = True
        start_x, end_x = xlim
        plt.xlim(start_x, end_x)
    else:
        use_xlim = False
        start_x, end_x = 0, len(timesteps)
        plt.xlim(-0.5)

    break_positions = [timestamp_borders[i] - 0 for i in range(1, len(timestamp_borders))]
    # Define the size of the break indicators in axes fraction units

    transform = blended_transform_factory(ax.transData, ax.transAxes)
    for pos in break_positions:
        if not use_xlim or (start_x <= pos <= end_x):
            # Position the "//" text at the center of the break position
            plt.text(pos, 0, "//", ha='center', va='center', color="black", transform=transform, 
                    fontsize='large', clip_on=False)

    plt.ylim(-.5)
    #plt.ylim(-0.5, 0.5)
    plt.xlabel('Time and ordering')
    plt.ylabel('Neuron index')
    plt.title(title)
    plt.grid(True, which='major')

    plt.show()