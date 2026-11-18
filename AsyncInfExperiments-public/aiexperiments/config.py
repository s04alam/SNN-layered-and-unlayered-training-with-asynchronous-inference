from typing import Dict, List
from aiexperiments.constants import (
    EventOrderingMethod,
    EventSchedulingMethod,
    NetworkExtension,
    network_extension_from_string,
)
import json


class SubExperimentConfig:
    def __init__(
        self,
        timestep_size: int,
        batch_size: int,
        tau_m: float = None, # If None, use value from model
        sync_barrier: EventOrderingMethod = EventOrderingMethod.NONE,
        event_scheduling_method: EventSchedulingMethod = EventSchedulingMethod.RANDOM_SAMPLING,
        network_extensions: List[NetworkExtension] = [],
        limit_n_samples: int = None,
        shuffle_samples = True,
    ):
        self.timestep_size = timestep_size
        self.batch_size = batch_size
        self.sync_barrier = sync_barrier
        self.network_extensions = network_extensions
        self.limit_n_samples = limit_n_samples
        self.shuffle_samples = shuffle_samples
        self.tau_m = tau_m
        self.event_scheduling_method = event_scheduling_method


class SharedExperimentConfig:
    def __init__(
        self,
        sub_experiments: List[SubExperimentConfig],
        variable_dt_save_output_with_dt: int,
        track_order_recordings: bool,
        order_histogram_bins: int,
        track_membrane_recordings: bool,
    ):
        self.sub_experiments = sub_experiments  # sub experiments are those experiments which will be executed for each experiment in the series
        self.variable_dt_save_output_with_dt = variable_dt_save_output_with_dt
        self.track_order_recordings = track_order_recordings
        self.order_histogram_bins = order_histogram_bins
        self.track_membrane_recordings = track_membrane_recordings


class SharedConfig:
    def __init__(
        self, shared_experiment_configs: Dict[str, Dict[str, SharedExperimentConfig]]
    ):
        self.shared_experiment_configs = shared_experiment_configs  # dataset_name -> inference_method_name -> ExperimentConfig


def load_config_from_files(shared_config_file_path: str, experiment_config_file_path: str, model_timestep_size=None) -> SharedConfig:
    with open(shared_config_file_path, "r") as file:
        shared_config_json_obj = json.load(file)
    with open(experiment_config_file_path, "r") as file:
        experiment_config_json_obj = json.load(file)
    return load_config_from_json(shared_config_json_obj, experiment_config_json_obj, model_timestep_size=model_timestep_size)

def merge_dicts(primary, secondary):
    for key in secondary:
        if key in primary:
            if isinstance(primary[key], dict) and isinstance(secondary[key], dict):
                # If both values are dictionaries, recurse
                merge_dicts(primary[key], secondary[key])
            else:
                primary[key] = secondary[key]
        else:
            primary[key] = secondary[key]

    return primary

def load_config_from_json(shared_config_json_obj, experiment_config_json_obj, model_timestep_size=None) -> SharedConfig:
    merged_config = merge_dicts(shared_config_json_obj, experiment_config_json_obj)
    experiment_configs = {}
    for dataset_name, dataset_config in merged_config.items():
        inference_method_configs = {}
        for inference_method_name, inference_method_config in dataset_config.items():
            sub_experiments = []
            # Load default values for sub experiments
            default_limit_n_samples = inference_method_config.get(
                "limit_n_samples", None
            )
            default_batch_size = inference_method_config.get("batch_size", 1)
            default_event_scheduling_method = inference_method_config.get(
                "event_scheduling_method", None
            )
            # Load sub experiments
            if "experiments" in inference_method_config:
                for sub_experiment_config in inference_method_config["experiments"]:
                    limit_n_samples = sub_experiment_config.get(
                        "limit_n_samples", default_limit_n_samples
                    )
                    batch_size = sub_experiment_config.get("batch_size", default_batch_size)
                    sync_barrier = sub_experiment_config.get("sync_barrier", None)
                    tau_m = sub_experiment_config.get("tau_m", None)
                    if sync_barrier is not None:
                        sync_barrier = EventOrderingMethod.from_string(sync_barrier)
                    else:
                        sync_barrier = EventOrderingMethod.NONE
                    event_scheduling_method = sub_experiment_config.get("event_scheduling_method", default_event_scheduling_method)
                    if event_scheduling_method is not None:
                        event_scheduling_method = EventSchedulingMethod.from_string(event_scheduling_method)
                    else:
                        event_scheduling_method = EventSchedulingMethod.RANDOM_SAMPLING
                    network_extensions = []
                    for extension_str in sub_experiment_config.get("network_extensions", []):
                        network_extensions.append(network_extension_from_string(extension_str))
                    timestep_size = sub_experiment_config["timestep_size"]
                    if timestep_size == "model":
                        if model_timestep_size is None:
                            raise ValueError("Model timestep size must be specified if sub experiment timestep size is 'model'")
                        timestep_size = model_timestep_size
                    else:
                        timestep_size = int(timestep_size)
                    sub_experiment = SubExperimentConfig(
                        timestep_size=timestep_size,
                        batch_size=batch_size,
                        sync_barrier=sync_barrier,
                        network_extensions=network_extensions,
                        limit_n_samples=limit_n_samples,
                        tau_m=tau_m,
                        event_scheduling_method=event_scheduling_method,
                    )
                    sub_experiments.append(sub_experiment)
            # Put everything together
            experiment_config = SharedExperimentConfig(
                sub_experiments=sub_experiments,
                variable_dt_save_output_with_dt=inference_method_config.get(
                    "variable_dt_save_output_with_dt", 1
                ),
                track_order_recordings=inference_method_config.get(
                    "track_order_recordings", False
                ),
                order_histogram_bins=inference_method_config.get(
                    "order_histogram_bins", 0
                ),
                track_membrane_recordings=inference_method_config.get(
                    "track_membrane_recordings", False
                ),
            )
            inference_method_configs[inference_method_name] = experiment_config
        experiment_configs[dataset_name] = inference_method_configs
    return SharedConfig(shared_experiment_configs=experiment_configs)
