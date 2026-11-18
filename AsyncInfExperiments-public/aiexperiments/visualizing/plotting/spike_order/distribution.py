# Code for histograms, Violin plots
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from aiexperiments.utils import reduce_bins
from aiexperiments.visualizing.results import CombinedExperimentResults, ResultKey
from matplotlib import ticker

def plot_histogram(
    results: CombinedExperimentResults, plot_neurons: List[int], key: ResultKey
):
    plt.figure(figsize=(10, 6))

    for neuron_idx in plot_neurons:
        hist = reduce_bins(results.density_histograms[key][neuron_idx], 100)
        edges = np.linspace(0, 1, len(hist) + 1)
        plt.bar(
            edges[:-1],
            hist,
            width=(edges[1] - edges[0]),
            align="edge",
            edgecolor="k",
            alpha=0.5,
            label=f"Neuron {neuron_idx}",
        )

    plt.xlabel("Normalized order")
    plt.ylabel("Frequency")
    plt.title("Density histogram of normalized ordering data")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_violin(
    results: CombinedExperimentResults,
    order_recordings: Dict[ResultKey, torch.Tensor],
    key: ResultKey,
    only_plot_neurons: List[int] = None,
    plot_input=False,
    plot_output=False,
    plot_classes: List[int] = None # None means plot all classes
):
    data_tensor = order_recordings[key]
    data_tensor[data_tensor == 0] = torch.tensor(float("nan"))
    if plot_classes is not None:
        labels = results.labels[key]
        filtered_indices = torch.tensor(
            [label.item() in plot_classes for label in labels]
        )
        data_tensor = data_tensor[:, filtered_indices, :]

    # Average over timesteps
    data_averaged = torch.nanmean(data_tensor, dim=0)
    n_input_layers = data_averaged.size(1) - sum(results.neuron_layers) 

    data_for_violin = []
    neuron_indices = []
    for neuron_idx in range(data_averaged.size(1)):
        if only_plot_neurons is not None and neuron_idx not in only_plot_neurons:
            continue

        if neuron_idx < n_input_layers and not plot_input:
            continue

        if neuron_idx >= n_input_layers + sum(results.neuron_layers[:-1]) and not plot_output:
            continue
        
        samples = data_averaged[:, neuron_idx].numpy()
        valid_samples = samples[~np.isnan(samples)]
        if len(valid_samples) == 0:
            valid_samples = [0]
        data_for_violin.extend(valid_samples)
        neuron_indices.extend([neuron_idx] * len(valid_samples))

    # Use seaborn's violinplot for smoother plots
    plt.figure(figsize=(10, 10))
    sns.violinplot(
        y=neuron_indices,
        x=data_for_violin,
        orient="h",
        inner="quartile",
        scale="count",
        bw=0.6,
        cut=0,
    )
    # Reverse the y-axis
    ax = plt.gca()
    ax.invert_yaxis()

    # Determine layer boundaries
    neuron_layers = [n_input_layers] + results.neuron_layers
    if not plot_input:
        neuron_layers[0] = 0
    if not plot_output:
        neuron_layers[-1] = 0
    layer_boundaries = np.cumsum(neuron_layers)
    if only_plot_neurons is not None:
        adjusted_layer_boundaries = []
        for i in range(len(layer_boundaries)):
            adjusted_layer_boundaries.append(0)
            start_layer = layer_boundaries[i - 1] if i > 0 else 0
            end_layer = layer_boundaries[i]  # Exclusive
            for neuron_idx in only_plot_neurons:
                if neuron_idx >= start_layer and neuron_idx < end_layer:
                    adjusted_layer_boundaries[-1] += 1
        layer_boundaries = np.cumsum(adjusted_layer_boundaries)

    # Add horizontal lines and labels
    yticks = []
    ytick_labels = []
    for i, y_i in enumerate(layer_boundaries):
        y_layer_start = layer_boundaries[i - 1] if i > 0 else 0
        y_layer_end = y_i
        y_layer_mid = y_layer_start + (y_layer_end - y_layer_start - 2.5) / 2
        if y_layer_end == 0 or y_layer_end == layer_boundaries[i - 1]:
            continue
        if i < len(neuron_layers) - 1:
            plt.axhline(
                y=y_layer_end - 0.5, color="grey", linestyle="--", alpha=1, c="red"
            )

        if i == len(neuron_layers) - 1:
            layer_name = "Output layer"
        elif i == 0:
            layer_name = "Input layer"
        else:
            layer_name = f"Hidden Layer {i}"

        yticks.append(y_layer_start - 0.5)
        ytick_labels.append(layer_name)

    labels = plt.yticks(yticks, ytick_labels)
    for label in labels[1]:
        label.set_x(-0.025)
    ax.yaxis.set_minor_locator(
        ticker.FixedLocator(
            [i for i in range(0, sum(neuron_layers))]
        )
    )
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, _: int(x)))

    plt.xlabel("Normalized order")
    plt.ylabel("Neuron index")
    plt.title(
        f"Normalized order of when neurons are active during one single inference timestep,\n train dt={key.training_dt}, inference dt={key.inference_dt}, inference method={key.inference_method.name}, sync barrier={key.sync_barrier.name}, extensions={key.inference_nn_extensions}"
    )
    plt.xlim([0, 1])  # Assuming the data is normalized and lies between [0, 1]
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()
