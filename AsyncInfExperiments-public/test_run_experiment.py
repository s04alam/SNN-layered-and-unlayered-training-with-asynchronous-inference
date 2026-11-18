# Having barrier neurons wait for all inputs to arrive before firing should
# result in the same output as a matrix multiplication.
# This test checks if this is the case.

from aiexperiments.config import SubExperimentConfig, load_config_from_files
from aiexperiments.constants import (
    EventOrderingMethod,
    network_extensions_from_string,
)
import numpy as np
import pandas as pd
from aiexperiments.visualizing.results import SingleExperimentOutputKey
import torch
from run_experiment import run_experiment

# Hard coded values, it may be necessary to change these if the experiments
# are changed.
circle_barrier_driven_experiment_id = 1 # I do not understand the meaning of it
circle_matrix_experiment_id = 2

def run_test_experiment(
    device,
    inference_dt,
    experiment_id,
    sync_barrier,
    network_extensions,
    series_id,
) -> np.ndarray:
    sub_experiment_config = SubExperimentConfig(
        inference_dt,
        batch_size=1,
        sync_barrier=sync_barrier,
        network_extensions=network_extensions_from_string(network_extensions),
        limit_n_samples=1,
        shuffle_samples=False,
    )

    # Load experiment
    experiments_file = f"./experiments/{series_id}.xlsx"
    df_experiments = pd.read_excel(experiments_file, index_col=0)
    experiment_row = df_experiments.loc[experiment_id]
    model_id = experiment_row["model id"]
    config_name = experiment_row["config"]
    df_models = pd.read_excel(f"./models/model_descriptions_{series_id}.xlsx", index_col=0)
    model_row = df_models.loc[model_id]
    shared_experiment_configs = load_config_from_files(
        f"./experiments/shared_config.json",
        f"./experiments/experiment_config_{config_name}.json",
        model_timestep_size=int(model_row["timestep size"]),
    )
    shared_experiment_config = shared_experiment_configs.shared_experiment_configs[
        model_row["data"]
    ][experiment_row["inference method"]]
    shared_experiment_config.sub_experiments = [
        sub_experiment_config
    ]  # overwrite the default sub-experiment
    shared_experiment_config.track_order_recordings = True
    shared_experiment_config.track_membrane_recordings = True

    # Run experiment
    output = run_experiment(
        experiment_row, shared_experiment_config, model_row, series_id, device
    )
    output_key = SingleExperimentOutputKey(
        inference_dt,
        sub_experiment_config.sync_barrier.name.lower(),
        network_extensions,
    )
    return output.output_recordings[output_key].sum(axis=0)

def test_run_experiment_circle():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_dt = 0
    output_sum_barrier_driven = run_test_experiment(
        device,
        inference_dt,
        experiment_id=circle_barrier_driven_experiment_id,
        sync_barrier=EventOrderingMethod.NONE,
        network_extensions="barrier_lif/6,input_barrier",
        series_id="circle"
    )
    output_sum_matrix = run_test_experiment(
        device,
        inference_dt,
        experiment_id=circle_matrix_experiment_id,
        sync_barrier=EventOrderingMethod.NONE,
        network_extensions="",
        series_id="circle"
    )
    # Check if output membrane potentials are the same
    difference = np.abs(output_sum_barrier_driven - output_sum_matrix)
    assert np.allclose(
        output_sum_barrier_driven,
        output_sum_matrix,
    ),  f"Output is not the same. Difference: {difference}"
