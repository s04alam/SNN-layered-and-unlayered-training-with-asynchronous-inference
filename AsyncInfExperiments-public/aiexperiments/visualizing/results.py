from collections import namedtuple
import os
from typing import Any, Dict, List
import pandas as pd
import pickle
import numpy as np
import torch

from tqdm import tqdm
from aiexperiments.config import SharedExperimentConfig, load_config_from_files
from aiexperiments.constants import EventOrderingMethod, InferenceMethod
from aiexperiments.utils import reduce_bins_in_dict
from aiexperiments.visualizing.metrics import (
    compute_latency,
    compute_normalized_order_overlap,
    compute_mean_confidence_true_class,
    compute_max_confidence_true_class
)

def load_experiments_per_series(experiments_per_series: Dict[str, List[int]], load_advanced_statistics=False):
    results_per_series: Dict[str, CombinedExperimentResults] = {}
    for series, plot_experiments in experiments_per_series.items():
        series_internal_name = series.split('_')[0] # what it is called in the results folder
        results_per_series[series] = CombinedExperimentResults(series_internal_name, plot_experiments, base_path='..')
        results_per_series[series].load()
        if load_advanced_statistics:
            results_per_series[series].load_current_statistics()
            results_per_series[series].load_histograms()
            results_per_series[series].load_order_recordings()
            results_per_series[series].compute_normalized_order_overlaps()
            results_per_series[series].compute_confidences()
            #results_per_series[series].compute_latencies()
    return results_per_series

### This is the format used for saving
SingleExperimentOutputKey = namedtuple(
    "SingleExperimentOutputKey",
    [
        "inference_dt",
        "sync_barrier",
        "inference_nn_extensions",
    ],
)
# Class to store the output of a single experiment
# It represents the format in which the output is saved
class SingleExperimentOutput:
    # key = SingleExperimentOutputKey
    output_recordings: dict
    labels: dict
    order_recordings: dict
    membrane_recordings: dict
    membrane_timesteps: dict
    histograms: dict
    density_histograms: dict
    accuracy: float
    loss: float
    spike_counts: dict
    elapsed_time: float
    n_currents_to_spike: dict
    sum_excitatory_currents_to_spike: dict 
    sum_inhibitory_currents_to_spike: dict 
    currents_histogram: dict # shape key-> np.ndarray(n_samples, n_neurons, n_bins)
    currents_for_spike_histogram: dict # shape key->np.ndarray(n_samples, n_neurons, n_bins)
    classification_latencies: dict
    correct_classification_latencies: dict

    def __init__(self):
        self.output_recordings = {}
        self.labels = {}
        self.order_recordings = {}
        self.membrane_recordings = {}
        self.membrane_timesteps = {}
        self.histograms = {}
        self.density_histograms = {}
        self.n_currents_to_spike = {}
        self.sum_excitatory_currents_to_spike = {}
        self.sum_inhibitory_currents_to_spike = {}
        self.currents_histogram = {}
        self.currents_for_spike_histogram = {}
        self.classification_latencies = {}
        self.correct_classification_latencies = {}
        self.spike_counts = {}
        self.accuracy = 0
        self.loss = 0
        self.elapsed_time = 0


### This is the format for the combined results
ResultKey = namedtuple(
    "ResultKey",
    [
        "training_dt",
        "training_nn_extensions",
        "inference_dt",
        "inference_method",
        "sync_barrier",
        "inference_nn_extensions",
        "config",
    ],
)

# Represents one single experiment (a row in the experiments excel file)
class ExperimentResults:
    def __init__(self, series_id, experiment_id, base_path="."):
        self.experiment_id = experiment_id
        self.series_id = series_id
        self.base_path = base_path
        self.results_folder_path = (
            f"{base_path}/results/{series_id}/{self.experiment_id}"
        )
        self.is_event_driven = os.path.exists(
            f"{self.results_folder_path}/event_order_recording_histograms.pickle"
        )
        self.order_recordings_saved = os.path.exists(
            f"{self.results_folder_path}/event_order_recordings.pickle"
        )

    def load_experiment_description(self):
        experiment_descriptions_path = (
            f"{self.base_path}/experiments/{self.series_id}.xlsx"
        )
        experiment_descriptions = pd.read_excel(
            experiment_descriptions_path, index_col=0
        )
        return experiment_descriptions.loc[self.experiment_id]

    def load_config(
        self, experiment_description=None, model_timestep_size=None, base_path="."
    ) -> SharedExperimentConfig:
        if experiment_description is None:
            experiment_description = self.load_experiment_description()
        config_name = experiment_description["config"]
        return load_config_from_files(
            f"{base_path}/experiments/shared_config.json",
            f"{base_path}/experiments/experiment_config_{config_name}.json",
            model_timestep_size=model_timestep_size
        ).shared_experiment_configs[self.series_id][
            experiment_description["inference method"]
        ]

    def load_model_description(self, experiment_description=None):
        if experiment_description is None:
            experiment_description = self.load_experiment_description()
        model_id = experiment_description["model id"]
        model_descriptions_path = f"{self.base_path}/models/model_descriptions_{self.series_id}.xlsx"
        model_descriptions = pd.read_excel(model_descriptions_path, index_col=0)
        return model_descriptions.loc[model_id]
    
    def load_model_params(self, experiment_description=None):
        if experiment_description is None:
            experiment_description = self.load_experiment_description()
        model_id = experiment_description["model id"]
        model_params_path = f"{self.base_path}/models/{self.series_id}/{model_id}.pt"
        model_params = torch.load(model_params_path)
        return model_params

    def load_results(self):
        results_path = f"{self.results_folder_path}/results.txt"
        return pd.read_csv(results_path, index_col=0)

    def load_event_order_recordings(self):
        event_order_recordings_path = (
            f"{self.results_folder_path}/event_order_recordings.pickle"
        )
        with open(event_order_recordings_path, "rb") as handle:
            event_order_recordings = pickle.load(
                handle
            )  # SingleExperimentOutputKey -> tensor of shape (num_timesteps, num_samples, num_neurons)
        return event_order_recordings

    def load_event_order_recordings_histograms(self):
        event_order_recordings_histograms_path = (
            f"{self.results_folder_path}/event_order_recording_histograms.pickle"
        )
        with open(event_order_recordings_histograms_path, "rb") as handle:
            event_order_recordings_histograms = pickle.load(handle)
        return event_order_recordings_histograms

    def load_event_order_recordings_density_histograms(self):
        event_order_recordings_density_histograms_path = f"{self.results_folder_path}/event_order_recording_density_histograms.pickle"
        with open(event_order_recordings_density_histograms_path, "rb") as handle:
            event_order_recordings_density_histograms = pickle.load(handle)
        return event_order_recordings_density_histograms

    def load_output_recordings(self):
        # NOTE: for ts=0 (variable dt), the output recordings are saved with a dt equal to variable_dt_save_output_with_dt
        output_recordings_path = f"{self.results_folder_path}/output_recordings.pickle"
        with open(output_recordings_path, "rb") as handle:
            output_recordings = pickle.load(handle)
        return output_recordings

    def load_labels(self):
        labels_path = f"{self.results_folder_path}/labels.pickle"
        with open(labels_path, "rb") as handle:
            labels = pickle.load(handle)
        return labels

    def supports_current_statistics(self):
        return os.path.exists(f"{self.results_folder_path}/n_currents_to_spike.pickle")

    def load_n_currents_to_spike(self):
        n_currents_to_spike_path = f"{self.results_folder_path}/n_currents_to_spike.pickle"
        with open(n_currents_to_spike_path, "rb") as handle:
            n_currents_to_spike = pickle.load(handle)
        return n_currents_to_spike
    
    def load_sum_excitatory_currents_to_spike(self):
        excitatory_currents_for_spike_path = f"{self.results_folder_path}/sum_excitatory_currents_to_spike.pickle"
        with open(excitatory_currents_for_spike_path, "rb") as handle:
            excitatory_currents_for_spike = pickle.load(handle)
        return excitatory_currents_for_spike
    
    def load_sum_inhibitory_currents_to_spike(self):
        inhibitory_currents_for_spike_path = f"{self.results_folder_path}/sum_inhibitory_currents_to_spike.pickle"
        with open(inhibitory_currents_for_spike_path, "rb") as handle:
            inhibitory_currents_for_spike = pickle.load(handle)
        return inhibitory_currents_for_spike
    
    def load_currents_histogram(self):
        currents_histogram_path = f"{self.results_folder_path}/event_currents_histogram.pickle"
        with open(currents_histogram_path, "rb") as handle:
            currents_histogram = pickle.load(handle)
        return currents_histogram
    
    def load_currents_for_spike_histogram(self):
        currents_for_spike_histogram_path = f"{self.results_folder_path}/event_currents_for_spike_histogram.pickle"
        with open(currents_for_spike_histogram_path, "rb") as handle:
            currents_for_spike_histogram = pickle.load(handle)
        return currents_for_spike_histogram
    
    def supports_latencies(self):
        return os.path.exists(f"{self.results_folder_path}/classification_latencies.pickle")
    
    def load_classification_latencies(self):
        classification_latencies_path = f"{self.results_folder_path}/classification_latencies.pickle"
        with open(classification_latencies_path, "rb") as handle:
            classification_latencies = pickle.load(handle)
        return classification_latencies
    
    def load_correct_classification_latencies(self):
        correct_classification_latencies_path = f"{self.results_folder_path}/correct_classification_latencies.pickle"
        with open(correct_classification_latencies_path, "rb") as handle:
            correct_classification_latencies = pickle.load(handle)
        return correct_classification_latencies
    
    def load_spike_counts(self):
        spike_counts_path = f"{self.results_folder_path}/spike_counts.pickle"
        with open(spike_counts_path, "rb") as handle:
            spike_counts = pickle.load(handle)
        return spike_counts


# CombinedExperimentResults is a class that combines the results of multiple experiments into dictionaries
class CombinedExperimentResults:
    def __init__(self, series_id, experiment_ids, base_path="."):
        self.experiment_ids = experiment_ids
        self.experiment_results = [
            ExperimentResults(series_id, experiment_id, base_path=base_path)
            for experiment_id in self.experiment_ids
        ]

        # All dicts will have keys (training dt, inference dt, inference method, sync barrier (or None if not event driven), inference nn extensions)
        self.accuracies = {}
        self.losses = {}
        self.spike_counts = {}
        self.latencies = {}
        self.mean_confidences = {}
        self.max_confidences = {}
        self.order_recordings = {}
        self.labels = {}
        self.histograms = {}
        self.density_histograms = {}
        self.normalized_order_overlaps = {}
        self.n_currents_to_spike = {}
        self.sum_excitatory_currents_to_spike = {}
        self.sum_inhibitory_currents_to_spike = {}
        self.abs_currents_to_spike = {}
        self.currents_histogram = {}
        self.currents_for_spike_histogram = {}
        self.classification_latencies = {}
        self.correct_classification_latencies = {}
        self.keys = []

        if series_id == "nmnist":
            self.n_inputs = 32 * 32 * 2
            self.neuron_layers = [64, 64, 64, 10]
            self.latency_timestep_size = 1000  # in us
        elif series_id == "circle":
            self.n_inputs = 3 * 3 * 1
            self.neuron_layers = [6, 6, 6, 2]
            self.latency_timestep_size = 1
        elif series_id == "shd":
            self.n_inputs = 350 * 1 * 2
            self.neuron_layers = [256, 256, 256, 20]
            self.latency_timestep_size = 1000
        elif series_id == "dvsgesture":
            self.n_inputs = 32 * 32 * 2
            self.neuron_layers = [256, 256, 256, 11]
            self.latency_timestep_size = 50000
        else:
            raise ValueError(f"Unknown experiment serie {series_id}")
        self.n_neurons = sum(self.neuron_layers)

    def get_output_key(self, inference_dt: int, sync_barrier: str, inference_nn_extensions: Any):
        inference_nn_extensions_clean = str(inference_nn_extensions)
        if inference_nn_extensions_clean == 'nan':
            inference_nn_extensions_clean = ''
        inference_nn_extensions_clean = inference_nn_extensions_clean.replace("+", ",")
        return SingleExperimentOutputKey(
            int(inference_dt), sync_barrier, inference_nn_extensions_clean
        )

    def get_dict_key(
        self, experiment_result: ExperimentResults, output_key: SingleExperimentOutputKey
    ) -> ResultKey:
        experiment_description = experiment_result.load_experiment_description()
        model_description = experiment_result.load_model_description(
            experiment_description=experiment_description
        )
        inference_method = InferenceMethod.from_string(
            experiment_description["inference method"]
        )
        training_dt = int(model_description["timestep size"])
        standard_nn_extensions = str(model_description["standard nn extensions"])
        training_nn_extensions = str(model_description["training nn extensions"])
        if standard_nn_extensions == "nan":
            standard_nn_extensions = ""
        if training_nn_extensions == "nan":
            training_nn_extensions = ""
        training_nn_extensions = ' '.join(filter(None, [standard_nn_extensions, training_nn_extensions]))
        config = str(experiment_description["config"])
        return ResultKey(
            training_dt,
            training_nn_extensions,
            output_key.inference_dt,
            inference_method,
            EventOrderingMethod.from_string(output_key.sync_barrier),
            output_key.inference_nn_extensions,
            config,
        )

    def load(self):  # always call load() first
        print("Loading results...")
        self.keys = []
        for experiment_result in tqdm(self.experiment_results):
            df_results = experiment_result.load_results()
            labels = experiment_result.load_labels()
            spike_counts = experiment_result.load_spike_counts()
            if experiment_result.supports_latencies():
                classification_latencies = experiment_result.load_classification_latencies()
                correct_classification_latencies = experiment_result.load_correct_classification_latencies()
            if experiment_result.supports_current_statistics():
                n_currents_to_spike = experiment_result.load_n_currents_to_spike()
                sum_inhibitory_currents_to_spike = experiment_result.load_sum_inhibitory_currents_to_spike()
                sum_excitatory_currents_to_spike = experiment_result.load_sum_excitatory_currents_to_spike()
            for idx in df_results.index:
                output_key = self.get_output_key(df_results.loc[idx]["timestep_size"], df_results.loc[idx]["sync_barrier"], df_results.loc[idx]["inference_nn_extensions"])
                dict_key = self.get_dict_key(experiment_result, output_key)
                self.keys.append(dict_key)
                accuracy = df_results.loc[idx]["accuracy"]
                loss = df_results.loc[idx]["loss"]
                self.accuracies[dict_key] = accuracy
                self.losses[dict_key] = loss
                self.spike_counts[dict_key] = spike_counts[output_key]
                self.labels[dict_key] = labels[output_key]
                if experiment_result.supports_current_statistics():
                    self.n_currents_to_spike[dict_key] = n_currents_to_spike[output_key]
                    self.sum_inhibitory_currents_to_spike[dict_key] = sum_inhibitory_currents_to_spike[output_key]
                    self.sum_excitatory_currents_to_spike[dict_key] = sum_excitatory_currents_to_spike[output_key]
                    self.abs_currents_to_spike[dict_key] = np.abs(self.sum_inhibitory_currents_to_spike[dict_key]) + self.sum_excitatory_currents_to_spike[dict_key]
                if experiment_result.supports_latencies():
                    self.classification_latencies[dict_key] = classification_latencies[output_key]
                    self.correct_classification_latencies[dict_key] = correct_classification_latencies[output_key]

    def compute_confidences(self):
        print("Computing confidences...")
        for experiment_result in tqdm(self.experiment_results):
            output_recordings = experiment_result.load_output_recordings()
            df_results = experiment_result.load_results()
            for idx in df_results.index:
                timestep_size = df_results.loc[idx]["timestep_size"]
                output_key = self.get_output_key(timestep_size, df_results.loc[idx]["sync_barrier"], df_results.loc[idx]["inference_nn_extensions"])
                dict_key = self.get_dict_key(experiment_result, output_key)
                self.mean_confidences[dict_key] = compute_mean_confidence_true_class(output_recordings[output_key], self.labels[dict_key])
                self.max_confidences[dict_key] = compute_max_confidence_true_class(output_recordings[output_key], self.labels[dict_key])

    def compute_latencies(self):
        print("Computing latencies...")
        for experiment_result in tqdm(self.experiment_results):
            output_recordings = experiment_result.load_output_recordings()
            df_results = experiment_result.load_results()
            for idx in df_results.index:
                timestep_size = df_results.loc[idx]["timestep_size"]
                config = experiment_result.load_config(model_timestep_size=timestep_size)
                output_key = self.get_output_key(timestep_size, df_results.loc[idx]["sync_barrier"], df_results.loc[idx]["inference_nn_extensions"])
                dict_key = self.get_dict_key(experiment_result, output_key)
                if timestep_size > 0:
                    inference_dt_to_use = timestep_size
                else:
                    if dict_key.inference_method == InferenceMethod.MATRIX:
                        inference_dt_to_use = (
                            config.variable_dt_save_output_with_dt
                        )
                    else:
                        inference_dt_to_use = 1
                latency_timestep_size = max(
                    1, self.latency_timestep_size // inference_dt_to_use
                )
                latency = compute_latency(
                    output_recordings[output_key],
                    inference_dt_to_use,
                    latency_timestep_size,
                )
                self.latencies[dict_key] = latency

    # --- Event driven only ---
    # n_bins is the number of bins to use for the histograms, if None, the number of bins is inferred from the data
    def compute_normalized_order_overlaps(
        self, n_bins=None
    ):  # requires load_histograms() to be called first
        print("Computing density histogram overlaps...")
        for dict_key in tqdm(self.density_histograms.keys()):
            density_histograms = self.density_histograms[dict_key]
            if len(density_histograms) > 0:
                if n_bins is not None:
                    density_histograms = reduce_bins_in_dict(density_histograms, n_bins)
                avg_overlaps = compute_normalized_order_overlap(density_histograms)
                self.normalized_order_overlaps[dict_key] = np.mean(avg_overlaps)
            else:
                self.normalized_order_overlaps[dict_key] = 0

    # --- Event driven only ---
    def load_order_recordings(self):  # always call load() first
        print("Loading order recordings...")
        for experiment_result in tqdm(self.experiment_results):
            if experiment_result.order_recordings_saved:
                event_order_recordings = experiment_result.load_event_order_recordings()
                for output_key in event_order_recordings.keys():
                    dict_key = self.get_dict_key(experiment_result, output_key)
                    order_recordings = event_order_recordings[output_key]
                    self.order_recordings[dict_key] = order_recordings

    # --- Event driven only ---
    def load_histograms(self):
        print("Loading histograms...")
        for experiment_result in tqdm(self.experiment_results):
            if experiment_result.is_event_driven:
                event_order_recordings_histograms = (
                    experiment_result.load_event_order_recordings_histograms()
                )
                event_order_recordings_density_histograms = (
                    experiment_result.load_event_order_recordings_density_histograms()
                )
                for output_key in event_order_recordings_histograms.keys():
                    dict_key = self.get_dict_key(experiment_result, output_key)
                    histograms = event_order_recordings_histograms[output_key]
                    density_histograms = event_order_recordings_density_histograms[
                        output_key
                    ]
                    self.histograms[dict_key] = histograms
                    self.density_histograms[dict_key] = density_histograms

    # --- Event driven only ---
    def load_current_statistics(self):
        print("Loading current statistics...")
        for experiment_result in tqdm(self.experiment_results):
            if experiment_result.is_event_driven:
                currents_histogram = experiment_result.load_currents_histogram()
                currents_for_spike_histogram = experiment_result.load_currents_for_spike_histogram()
                for output_key in currents_histogram.keys():
                    dict_key = self.get_dict_key(experiment_result, output_key)
                    self.currents_histogram[dict_key] = currents_histogram[output_key]
                    self.currents_for_spike_histogram[dict_key] = currents_for_spike_histogram[output_key]


def generate_dummy_order_recordings(
    results: CombinedExperimentResults,
    n_inputs,
    n_neurons,
    n_samples=100,
    skip_inputs=True,
) -> Dict[ResultKey, torch.Tensor]:
    dummy_order_recordings = {}
    for key in results.density_histograms.keys():
        histograms = results.density_histograms[key]
        # keys in histograms refer to neuron index
        # variance is in n_samples, therefore for simplicity n_timesteps is fixed to 1
        track_n_inputs = n_inputs if not skip_inputs else 0
        dummy_order_recordings[key] = torch.zeros(
            (1, n_samples, n_neurons + track_n_inputs)
        )
        for neuron_i in range(n_inputs - track_n_inputs, n_neurons + n_inputs):
            if neuron_i in histograms.keys():
                histogram = histograms[neuron_i]
                histogram_sum = sum(histogram)
                if histogram_sum > 0:
                    p = (histogram / histogram_sum).numpy()
                    idx = neuron_i if not skip_inputs else neuron_i - n_inputs
                    dummy_order_recordings[key][0, :, idx] = torch.tensor(
                        np.random.choice(len(histogram), size=n_samples, p=p)
                        / len(histogram)
                    )
    return dummy_order_recordings
