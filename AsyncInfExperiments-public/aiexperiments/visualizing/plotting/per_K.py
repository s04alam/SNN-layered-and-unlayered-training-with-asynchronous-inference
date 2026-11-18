from enum import Enum
from typing import Any, Dict, List, Tuple, Union
import numpy as np

from aiexperiments.constants import EventOrderingMethod, InferenceMethod
from aiexperiments.visualizing.results import ResultKey
import copy

class SpecialBarrierType(Enum):
    PER_LAYER_SYNC = 1
    MATRIX_SYNC = 2

    def to_simple_label(self):
        if self == SpecialBarrierType.PER_LAYER_SYNC:
            return "L"
        elif self == SpecialBarrierType.MATRIX_SYNC:
            return "M"
        else:
            raise ValueError("Unknown special barrier type")


def calculate_percentiles_from_histogram(
    hist: np.ndarray, edges: np.ndarray, percentiles: np.ndarray
):
    cumulative_hist = np.cumsum(hist)
    total = cumulative_hist[-1]

    # A function to find the closest index in the cumulative histogram
    def find_nearest_index(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    # Finding the bin edges for each percentile
    percentile_values = []
    for percentile in percentiles:
        idx = find_nearest_index(cumulative_hist, total * percentile / 100)
        percentile_value = edges[idx]
        percentile_values.append(percentile_value)

    return percentile_values


# To be used for: currents_for_spike_histogram, currents_histogram
def fn_compute_hist_statistics(
    hist_per_K: List[np.ndarray], skip_first_n_neurons: int, **kwargs
):
    means = []
    lower_percentiles = []
    medians = []
    upper_percentiles = []
    for hist in hist_per_K:  # hist shape (samples, neuron, hist_bins)
        if hist is not None:
            assert len(hist.shape) == 3
            hist = hist[:, skip_first_n_neurons:, :].sum(axis=0).sum(axis=0)
            edges = np.linspace(-0.5, 0.5, len(hist) + 1)  # Edges are fixed
            mean = np.sum(hist * edges[:-1]) / np.sum(hist)
            (
                lower_percentile,
                median,
                upper_percentile,
            ) = calculate_percentiles_from_histogram(hist, edges, [25, 50, 75])
            means.append(mean)
            lower_percentiles.append(lower_percentile)
            medians.append(median)
            upper_percentiles.append(upper_percentile)
        else:
            means.append(0)
            lower_percentiles.append(0)
            medians.append(0)
            upper_percentiles.append(0)
    return (
        np.array(means),
        np.array(lower_percentiles),
        np.array(medians),
        np.array(upper_percentiles),
    )


# To be used for: histograms (spike order)
def fn_compute_spike_order_hist_statistics_per_layer(
    hist_per_K: List[np.ndarray],
    neuron_layers: np.ndarray,
    plot_layer_idx: int,
    **kwargs,
):
    means = []
    lower_percentiles = []
    medians = []
    upper_percentiles = []
    for hist_per_neuron in hist_per_K:  # hist_per_neuron neuron_index -> hist_bins
        if hist_per_neuron is not None:
            assert isinstance(hist_per_neuron, dict)
            neuron_indices_per_layer = np.cumsum(neuron_layers)
            first_neuron_idx = (
                neuron_indices_per_layer[plot_layer_idx] - neuron_layers[plot_layer_idx]
            )
            combined_hist = np.zeros(hist_per_neuron[first_neuron_idx].shape[0])
            edges = np.linspace(0, 1, len(combined_hist) + 1)  # Edges are fixed
            for i_neuron in range(
                first_neuron_idx, first_neuron_idx + neuron_layers[plot_layer_idx]
            ):
                combined_hist += hist_per_neuron[i_neuron].cpu().numpy()
            mean = np.sum(combined_hist * edges[:-1]) / np.sum(combined_hist)
            (
                lower_percentile,
                median,
                upper_percentile,
            ) = calculate_percentiles_from_histogram(combined_hist, edges, [25, 50, 75])
            means.append(mean)
            lower_percentiles.append(lower_percentile)
            medians.append(median)
            upper_percentiles.append(upper_percentile)
        else:
            means.append(0)
            lower_percentiles.append(0)
            medians.append(0)
            upper_percentiles.append(0)
    return (
        np.array(means),
        np.array(lower_percentiles),
        np.array(medians),
        np.array(upper_percentiles),
    )


# To be used for: n_currents_for_spike, abs_currents_for_spike
def fn_compute_2d_array_statistics(
    array_per_K: List[np.ndarray],
    skip_first_n_neurons: int,
    ignore_0_elements=True,
    **kwargs,
):
    means = []
    lower_percentiles = []
    medians = []
    upper_percentiles = []
    for array in array_per_K:  # array shape (samples, neuron)
        if array is not None and len(array) > 0:
            array = np.concatenate(array[:, skip_first_n_neurons:])
            if ignore_0_elements:
                array = array[array != 0]
        if array is not None and len(array) > 0:
            means.append(np.mean(array))
            lower_percentiles.append(np.percentile(array, 25))
            medians.append(np.median(array))
            upper_percentiles.append(np.percentile(array, 75))
        else:
            means.append(0)
            lower_percentiles.append(0)
            medians.append(0)
            upper_percentiles.append(0)
    return (
        np.array(means),
        np.array(lower_percentiles),
        np.array(medians),
        np.array(upper_percentiles),
    )


def fn_compute_1d_array_statistics(
    array_per_K: List[np.ndarray],
    skip_first_n_neurons: int,
    ignore_0_elements=True,
    **kwargs,
):
    means = []
    lower_percentiles = []
    medians = []
    upper_percentiles = []
    for array in array_per_K:  # array shape (samples, neuron)
        if array is not None and len(array) > 0:
            array = array[skip_first_n_neurons:]
            if ignore_0_elements:
                array = array[array != 0]
        if array is not None and len(array) > 0:
            means.append(np.mean(array))
            lower_percentiles.append(np.percentile(array, 25))
            medians.append(np.median(array))
            upper_percentiles.append(np.percentile(array, 75))
        else:
            means.append(0)
            lower_percentiles.append(0)
            medians.append(0)
            upper_percentiles.append(0)
    return (
        np.array(means),
        np.array(lower_percentiles),
        np.array(medians),
        np.array(upper_percentiles),
    )


# Retrieve K values for on the x-axis
def retrieve_xaxis_for_K_and_sync_barriers(
    keys: List[ResultKey], plot_special_barrier_types: List[SpecialBarrierType]
):
    # >> Ks, retrieve from keys
    Ks = []
    Kmin = 0  # with special Ks
    Kmax = 0
    n_special_Ks = 0  # logged as K < 0 (to reuse code), e.g., per layer synchronization
    special_K_ticks = []  # tuples (K, label)
    for key in keys:
        if "train_sync_params" in key.training_nn_extensions:
            continue
        inference_nn_extensions = str(key.inference_nn_extensions).split(",")
        for inference_nn_extension in inference_nn_extensions:
            if inference_nn_extension.startswith("barrier_lif"):
                K = int(inference_nn_extension.split("/")[1])
                if K < Kmin:
                    Kmin = K
                if K > Kmax:
                    Kmax = K
                if K not in Ks:
                    Ks.append(K)
    Ks = sorted(Ks)
    # >> Special barrier types to plot
    for special_barrier_type in plot_special_barrier_types:
        if K >= 0:
            Kmin -= 2
        K = Kmin
        Kmin -= 2
        n_special_Ks += 1
        special_K_ticks.append((K, special_barrier_type))
        Ks.insert(0, K)
    return np.array(Ks), n_special_Ks, special_K_ticks, Kmin, Kmax


# Given some default_key_values fitting ResultKey, except for the indp_variable_name, this function returns the indp_variable values and the data for each indp_variable.
def acquire_dep_variable_per_indp_variable(
    dep_variables: Dict[ResultKey, Any],
    indp_variable_name: str,
    default_key_values: Dict[str, Any],
):
    data = {}
    indp_variables = list(
        np.unique([getattr(key, indp_variable_name) for key in dep_variables.keys()])
    )
    # remove "train_sync_params" from indp_variables (if present in of the values)
    for indp_variable in indp_variables.copy():
        if "train_sync_params" in indp_variable:
            indp_variables.remove(indp_variable)

    for indp_variable in indp_variables:
        key_values = default_key_values.copy()
        key_values[indp_variable_name] = indp_variable
        key = ResultKey(**key_values)
        if key in dep_variables.keys():
            data[indp_variable] = dep_variables[key]
        else:
            print(f"WARNING: key {key} not found")
            data[indp_variable] = None
    return indp_variables, data


# In this case, K is the first independent variable, and the second independent variable is custom
# This function fills in the network extension key value and sync barrier key value
def acquire_dep_variable_per_indp_variable_per_K(
    dep_variables: Dict[ResultKey, Any],
    indp_variable_name: str,
    Ks: List[float],
    special_K_ticks: List[Tuple[float, SpecialBarrierType]],
    default_key_values: Dict[str, Any],
):
    data_per_indp_variable = {}
    for K in Ks:
        sync_barrier = EventOrderingMethod.NONE
        inference_method = InferenceMethod.EVENT
        if K in [k for k, _ in special_K_ticks]:
            special_barrier_type = [sb for k, sb in special_K_ticks if k == K][0]
            if special_barrier_type == SpecialBarrierType.PER_LAYER_SYNC:
                sync_barrier = EventOrderingMethod.PER_LAYER
                barrier = "barrier_lif/1,input_barrier"
            elif special_barrier_type == SpecialBarrierType.MATRIX_SYNC:
                inference_method = InferenceMethod.MATRIX
                barrier = ""
            else:
                raise ValueError("Unknown special barrier type")
        else:
            barrier = f"barrier_lif/{K},input_barrier"
        key_values = default_key_values.copy()
        key_values["inference_nn_extensions"] = barrier
        key_values["sync_barrier"] = sync_barrier
        key_values["inference_method"] = inference_method
        indp_variables, data = acquire_dep_variable_per_indp_variable(
            dep_variables, indp_variable_name, key_values
        )
        for indp_variable in indp_variables:
            if indp_variable not in data_per_indp_variable.keys():
                data_per_indp_variable[indp_variable] = []
            data_per_indp_variable[indp_variable].append(data[indp_variable])
    return indp_variables, data_per_indp_variable


def compute_special_Ks_correction_factor(Kmin, Kmax, fixed_proportion_of_x_axis):
    full_range = Kmax - Kmin
    special_k_range = -Kmin
    desired_special_k_length = fixed_proportion_of_x_axis * full_range
    if special_k_range != 0:
        special_Ks_correction = desired_special_k_length / special_k_range
    else:
        special_Ks_correction = 1
    return special_Ks_correction


def format_legend_label(
    indp_variable_name: str,
    indp_variable: Any,
    override_legend_labels: Union[Dict[str, str], str, None],
):
    if override_legend_labels is not None:
        if isinstance(override_legend_labels, dict):
            if indp_variable in override_legend_labels.keys():
                return override_legend_labels[indp_variable].format(
                    indp_variable_name=indp_variable_name, indp_variable=indp_variable
                )
        else:
            return override_legend_labels.format(
                indp_variable_name=indp_variable_name, indp_variable=indp_variable
            )

    return f"{indp_variable_name} = {indp_variable}"


# Plot the following:
# - K as independent variable on x-axis + Strict per Layer (SL), Layer (L), and Matrix (M) methods on x-axis
# - Any "single line statistic" on y-axis (so no distribution)
# - One line per indepent variable
# Note that this function only handles complete results (i.e., all experiments have the same Ks and training timesteps)
def plot_results_for_single_line_statistic(
    ax,
    dep_variables: Dict[ResultKey, Any],
    indp_variable_name: str,
    default_key_values: Dict[str, Any],
    y_label=None,
    title=None,
    plot_legend=True,
    override_legend_labels: Union[
        Dict[str, str], str, None
    ] = None,  # can be a dictionary mapping indp_variable to legend label, or a string to use for all indp_variables
    plot_y_label=True,
    plot_special_barrier_types: List[SpecialBarrierType] = [
        SpecialBarrierType.MATRIX_SYNC,
        SpecialBarrierType.PER_LAYER_SYNC,
    ],
    start_color_idx=0,
    fix_frac_xaxis_for_special_barrier_types=0.1,
):
    (
        Ks,
        n_special_Ks,
        special_K_ticks,
        Kmin,
        Kmax,
    ) = retrieve_xaxis_for_K_and_sync_barriers(
        dep_variables.keys(), plot_special_barrier_types
    )
    (
        indp_variables,
        data_per_indp_variable,
    ) = acquire_dep_variable_per_indp_variable_per_K(
        dep_variables, indp_variable_name, Ks, special_K_ticks, default_key_values
    )

    # Plot
    colors = [
        "C" + str(i)
        for i in range(start_color_idx, start_color_idx + len(indp_variables))
    ]
    # >> Non-sync barrier methods
    for i, indp_variable in enumerate(indp_variables):
        ax.plot(
            Ks[n_special_Ks:],
            data_per_indp_variable[indp_variable][n_special_Ks:],
            c=colors[i],
            label=format_legend_label(
                indp_variable_name, indp_variable, override_legend_labels
            ),
        )
    # >> Sync barrier methods
    special_Ks_correction = compute_special_Ks_correction_factor(
        Kmin, Kmax, fix_frac_xaxis_for_special_barrier_types
    )

    for i, indp_variable in enumerate(indp_variables):
        ax.scatter(
            Ks[:n_special_Ks] * special_Ks_correction,
            data_per_indp_variable[indp_variable][:n_special_Ks],
            marker="_",
            s=50,
            c=colors[i],
        )
    # >> Labels
    ax.set_xlabel("$K$")
    ax.set_xlim(Kmin * special_Ks_correction, Kmax)
    ax.set_xticks(
        [k * special_Ks_correction for k, _ in special_K_ticks]
        + [k for k in Ks[n_special_Ks:]]
    )
    ax.set_xticklabels(
        [label.to_simple_label() for _, label in special_K_ticks]
        + [k for k in Ks[n_special_Ks:]]
    )
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
    if plot_legend:
        ax.legend()
    if y_label is not None and plot_y_label:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    return len(indp_variables), special_Ks_correction


# fn_compute_statistics: function that takes a list of values and returns a tuple (mean, lower 25%, median, upper 75%)
#   if None: do not compute or plot statistics, directly plot the values
def plot_results_for_distributed_statistic(
    ax,
    dep_variables: Dict[ResultKey, Any],
    indp_variable_name: str,
    default_key_values: Dict[str, Any],
    y_label=None,
    title=None,
    plot_legend=True,
    plot_extensive_legend=True,
    plot_indp_variable_in_legend=True,
    override_legend_labels: Union[
        Dict[str, str], str, None
    ] = None,  # can be a dictionary mapping indp_variable to legend label, or a string to use for all indp_variables
    plot_y_label=True,
    fn_compute_statistics=None,
    fn_compute_statistics_kwargs={},
    plot_special_barrier_types: List[SpecialBarrierType] = [
        SpecialBarrierType.PER_LAYER_SYNC
    ],
    start_color_idx=0,
    fix_frac_xaxis_for_special_barrier_types=0.1,
):
    (
        Ks,
        n_special_Ks,
        special_K_ticks,
        Kmin,
        Kmax,
    ) = retrieve_xaxis_for_K_and_sync_barriers(
        dep_variables.keys(), plot_special_barrier_types
    )
    (
        indp_variables,
        data_per_indp_variable,
    ) = acquire_dep_variable_per_indp_variable_per_K(
        dep_variables, indp_variable_name, Ks, special_K_ticks, default_key_values
    )

    colors = [
        "C" + str(i)
        for i in range(start_color_idx, start_color_idx + len(indp_variables))
    ]
    special_Ks_correction = compute_special_Ks_correction_factor(
        Kmin, Kmax, fix_frac_xaxis_for_special_barrier_types
    )
    # >> Non-sync barrier methods
    for i, indp_variable in enumerate(indp_variables):
        if fn_compute_statistics is not None:
            means, lowers, medians, uppers = fn_compute_statistics(
                data_per_indp_variable[indp_variable][n_special_Ks:],
                **fn_compute_statistics_kwargs,
            )
            percentile_label = (
                "25th to 75th percentile" if i == 0 and plot_extensive_legend else None
            )
            mean_label = "Mean" if i == 0 and plot_extensive_legend else None
            if plot_indp_variable_in_legend:
                formatted_label = format_legend_label(
                    indp_variable_name, indp_variable, override_legend_labels
                )
                median_label = (
                    f"Median\n{formatted_label}"
                    if i == 0 and plot_extensive_legend
                    else formatted_label
                )
            else:
                median_label = "Median" if i == 0 and plot_extensive_legend else None
            ax.fill_between(
                Ks[n_special_Ks:],
                lowers,
                uppers,
                color=colors[i],
                alpha=0.2,
                label=percentile_label,
            )
            ax.plot(
                Ks[n_special_Ks:],
                means,
                color=colors[i],
                linestyle=":",
                label=mean_label,
            )
            ax.plot(Ks[n_special_Ks:], medians, color=colors[i], label=median_label)
        else:
            ax.plot(
                Ks[n_special_Ks:],
                data_per_indp_variable[indp_variable][n_special_Ks:],
                c=colors[i],
            )
    # >> Sync barrier methods
    for i, indp_variable in enumerate(indp_variables):
        if fn_compute_statistics is not None:
            means, lowers, medians, uppers = fn_compute_statistics(
                data_per_indp_variable[indp_variable][:n_special_Ks],
                **fn_compute_statistics_kwargs,
            )
            ax.scatter(
                Ks[:n_special_Ks] * special_Ks_correction,
                medians[:n_special_Ks],
                color=colors[i],
                marker="_",
            )
            for j, K in enumerate(Ks[:n_special_Ks]):
                ax.plot(
                    [
                        (K - 0.5) * special_Ks_correction,
                        (K + 0.5) * special_Ks_correction,
                    ],
                    [means[j], means[j]],
                    c=colors[i],
                    linestyle=":",
                )
            ax.errorbar(
                Ks[:n_special_Ks] * special_Ks_correction,
                medians[:n_special_Ks],
                yerr=[
                    medians[:n_special_Ks] - lowers[:n_special_Ks],
                    uppers[:n_special_Ks] - medians[:n_special_Ks],
                ],
                fmt="none",
                ecolor=colors[i],
                elinewidth=6,
                alpha=0.2,
            )
        else:
            ax.scatter(
                Ks[:n_special_Ks] * special_Ks_correction,
                data_per_indp_variable[indp_variable][:n_special_Ks],
                marker="_",
                s=50,
                c=colors[i],
            )
    # >> Labels
    if plot_legend:
        ax.legend()
    if y_label is not None and plot_y_label:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("$K$")
    ax.set_xlim(Kmin * special_Ks_correction, Kmax)
    ax.set_xticks(
        [k * special_Ks_correction for k, _ in special_K_ticks]
        + [k for k in Ks[n_special_Ks:]]
    )
    ax.set_xticklabels(
        [label.to_simple_label() for _, label in special_K_ticks]
        + [k for k in Ks[n_special_Ks:]]
    )
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
    return len(indp_variables), special_Ks_correction


def plot_trained_sync_params(
    ax,
    K_approx: float,
    special_Ks_correction: float,
    dep_variable: Dict[ResultKey, Any],
    default_config: Dict[str, Any],
    training_extension: str,
    color_idx: int,
    fn_compute_statistics=None,
    fn_compute_statistics_kwargs={},
    plot_legend=True,
    label='',
):
    if not plot_legend:
        label = None
    color = "C" + str(color_idx)
    config = copy.deepcopy(default_config)
    config["training_nn_extensions"] = training_extension
    config["inference_nn_extensions"] = "barrier_lif/0/0,input_barrier"
    config["inference_method"] = InferenceMethod.EVENT
    config["sync_barrier"] = EventOrderingMethod.NONE
    config["config"] = "trained_sync_params"
    key = ResultKey(**config)
    if key in dep_variable:
        if fn_compute_statistics is not None:
            means, lowers, medians, uppers = fn_compute_statistics(
                [dep_variable[key]], **fn_compute_statistics_kwargs
            )
            ax.scatter(
                K_approx,
                medians,
                color=color,
                marker="_",
                label=label,
            )
            ax.plot(
                [
                    (K_approx - 0.5 * special_Ks_correction),
                    (K_approx + 0.5 * special_Ks_correction),
                ],
                [means, means],
                c=color,
                linestyle=":",
            )
            ax.errorbar(
                K_approx,
                medians,
                yerr=[
                    medians - lowers,
                    uppers - medians,
                ],
                fmt="none",
                ecolor=color,
                elinewidth=12,
                alpha=0.2,
            )
        else:
            ax.scatter(
                K_approx,
                dep_variable[key],
                color=color,
                marker="x",
                label=label,
            )

        if plot_legend:
            ax.legend()
