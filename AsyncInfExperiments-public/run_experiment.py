import os
from aiexperiments.config import SharedExperimentConfig, load_config_from_files
from aiexperiments.constants import (
    InferenceMethod,
    network_extensions_to_string,
)

from aiexperiments.experiment_provider import setup_inference_experiment
from aiexperiments.utils import combine_list_batches
from aiexperiments.visualizing.recording.event_recorder import EventRecorder
from aiexperiments.visualizing.recording.inference_recorder import (
    combine_tensor_batches,
)
from aiexperiments.visualizing.results import SingleExperimentOutput, SingleExperimentOutputKey

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import pandas as pd
import torch
import time
import pickle
import multiprocessing as mp
import traceback
import numpy as np


# Function not intended to be called from other scripts
def _run_configured_experiment(experiment_id: int, experiment_row: pd.Series):
    model_id = experiment_row["model id"]
    df_models = pd.read_excel(f"./models/model_descriptions_{series_id}.xlsx", index_col=0)
    model_row = df_models.loc[model_id]
    shared_config_file = "./experiments/shared_config.json"
    experiment_config_file = f"./experiments/experiment_config_{experiment_row['config']}.json"
    shared_experiment_configs = load_config_from_files(
        shared_config_file, experiment_config_file, model_timestep_size=int(model_row["timestep size"])
    )
    shared_experiment_config = shared_experiment_configs.shared_experiment_configs[
        model_row["data"]
    ][experiment_row["inference method"]]
    run_experiment(
        experiment_row,
        shared_experiment_config,
        model_row,
        series_id,  # this is a reference to a global variable
        device,  # this is a reference to a global variable
        save_output=True,
    )


# Function accessible from other scripts
def run_experiment(
    experiment_row: pd.Series,
    shared_experiment_config: SharedExperimentConfig,
    model_row: pd.Series,
    series_id: str,
    device: torch.device,
    base_path=".",
    save_output=False,
) -> SingleExperimentOutput:
    experiment_id = experiment_row.name
    model_id = experiment_row["model id"]
    output = SingleExperimentOutput()

    # Load experiment information
    inference_method = InferenceMethod.from_string(experiment_row["inference method"])
    model_path = f"{base_path}/models/{series_id}/{model_id}.pt"
    series_folder_path = f"{base_path}/results/{series_id}"
    results_folder_path = f"{base_path}/results/{series_id}/{experiment_id}"
    results_path = f"{results_folder_path}/results.txt"

    # Create results folder and results file
    if save_output:
        if not os.path.exists(series_folder_path):
            os.makedirs(series_folder_path)
        if not os.path.exists(results_folder_path):
            os.makedirs(results_folder_path)
        f = open(results_path, "w")
        f.write(
            f"index,timestep_size,batch_size,sync_barrier,inference_nn_extensions,accuracy,loss,elapsed_time\n"
        )
        f.close()

    try:
        # Run tests
        for i_sub_experiment, sub_experiment in enumerate(shared_experiment_config.sub_experiments):
            if (
                inference_method != InferenceMethod.EVENT
                and inference_method != InferenceMethod.MATRIX
                and sub_experiment.timestep_size == 0
            ):
                print("Skipping timestep size 0")
                continue  # Loihi does not support timestep size == 0, Matrix may support it, though not all networks do
            (
                inference_params,
                net_builder,
                net_builder_params,
                inference_engine,
            ) = setup_inference_experiment(
                experiment_description=experiment_row,
                model_description=model_row,
                device=device,
                load_network_parameters_from=model_path,
                experiment_config=shared_experiment_config,
                sub_experiment_config=sub_experiment,
            )
            start_time = time.time()
            print(
                f"Testing timestep size {sub_experiment.timestep_size} and batch size {sub_experiment.batch_size}"
            )
            recorder = inference_engine.infer(
                inference_params, net_builder, net_builder_params
            )
            # Acquire and log results
            # 1. Readable results
            output.accuracy = recorder.calculate_weighted_accuracy()
            output.loss = recorder.calculate_weighted_loss()
            output.elapsed_time = time.time() - start_time
            if save_output:
                f = open(results_path, "a")
                f.write(
                    f"{i_sub_experiment},{sub_experiment.timestep_size},{sub_experiment.batch_size},{sub_experiment.sync_barrier.name.lower()},{network_extensions_to_string(sub_experiment.network_extensions, delimiter='+')},{output.accuracy},{output.loss},{output.elapsed_time}\n"
                )
                f.close()

            # 2. Raw results
            key = SingleExperimentOutputKey(sub_experiment.timestep_size, sub_experiment.sync_barrier.name.lower(), network_extensions_to_string(sub_experiment.network_extensions))
            output.output_recordings[
                key
            ] = combine_tensor_batches(recorder.all_output_recordings)
            if recorder.track_membrane_recordings:
                output.membrane_recordings[
                    key
                ] = combine_list_batches(recorder.all_membrane_recordings)
                output.membrane_timesteps[
                    key
                ] = combine_list_batches(recorder.all_membrane_timesteps)

            output.labels[key] = torch.cat(
                recorder.all_labels, dim=0
            ).cpu().numpy()
            output.spike_counts[key] = torch.cat(
                recorder.all_spike_counts, dim=0
            ).cpu().numpy()
            output.classification_latencies[key] = np.array(recorder.classification_latencies)
            output.correct_classification_latencies[key] = np.array(recorder.correct_classification_latencies)
            output.n_currents_to_spike[key] = np.array(recorder.n_currents_to_spike)
            output.sum_excitatory_currents_to_spike[key] = np.array(recorder.sum_excitatory_currents_to_spike)
            output.sum_inhibitory_currents_to_spike[key] = np.array(recorder.sum_inhibitory_currents_to_spike)
            if inference_method == InferenceMethod.EVENT and isinstance(
                recorder, EventRecorder
            ):
                if shared_experiment_config.track_order_recordings:
                    output.order_recordings[
                        key
                    ] = combine_tensor_batches(recorder.all_order_recordings)
                output.density_histograms[
                    key
                ] = recorder.get_per_neuron_order_histogram(
                    n_bins=shared_experiment_config.order_histogram_bins, density=True
                )
                output.histograms[
                    key
                ] = recorder.get_per_neuron_order_histogram(
                    n_bins=shared_experiment_config.order_histogram_bins, density=False
                )
                output.currents_histogram[key] = np.stack(
                    recorder.currents_histogram
                )
                output.currents_for_spike_histogram[key] = np.stack(
                    recorder.currents_for_spike_histogram
                )
            del recorder

    except Exception as e:
        print("Error:", e, type(e))
        traceback.print_exc()
        return

    if save_output:
        # Save general inference output data
        print("Saving output data")
        with open(f"{results_folder_path}/output_recordings.pickle", "wb") as handle:
            pickle.dump(
                output.output_recordings, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        with open(f"{results_folder_path}/labels.pickle", "wb") as handle:
            pickle.dump(output.labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{results_folder_path}/spike_counts.pickle", "wb") as handle:
            pickle.dump(output.spike_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
        with open(
            f"{results_folder_path}/classification_latencies.pickle", "wb"
        ) as handle:
            pickle.dump(output.classification_latencies, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(
            f"{results_folder_path}/correct_classification_latencies.pickle", "wb"
        ) as handle:
            pickle.dump(output.correct_classification_latencies, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(
            f"{results_folder_path}/n_currents_to_spike.pickle",
            "wb",
        ) as handle:
            pickle.dump(
                output.n_currents_to_spike, handle, protocol=pickle.HIGHEST_PROTOCOL
            )   

        with open(
            f"{results_folder_path}/sum_inhibitory_currents_to_spike.pickle",
            "wb",
        ) as handle:
            pickle.dump(
                output.sum_inhibitory_currents_to_spike, handle, protocol=pickle.HIGHEST_PROTOCOL
            )   

        with open(
            f"{results_folder_path}/sum_excitatory_currents_to_spike.pickle",
            "wb",
        ) as handle:
            pickle.dump(
                output.sum_excitatory_currents_to_spike, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        # Save event inference output data
        if inference_method == InferenceMethod.EVENT:
            if shared_experiment_config.track_order_recordings:
                with open(
                    f"{results_folder_path}/event_order_recordings.pickle", "wb"
                ) as handle:
                    pickle.dump(
                        output.order_recordings,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

            with open(
                f"{results_folder_path}/event_currents_histogram.pickle",
                "wb",
            ) as handle:
                pickle.dump(
                    output.currents_histogram, handle, protocol=pickle.HIGHEST_PROTOCOL
                )   

            with open(
                f"{results_folder_path}/event_currents_for_spike_histogram.pickle",
                "wb",
            ) as handle:
                pickle.dump(
                    output.currents_for_spike_histogram, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

            with open(
                f"{results_folder_path}/event_order_recording_density_histograms.pickle",
                "wb",
            ) as handle:
                pickle.dump(
                    output.density_histograms, handle, protocol=pickle.HIGHEST_PROTOCOL
                )

            with open(
                f"{results_folder_path}/event_order_recording_histograms.pickle", "wb"
            ) as handle:
                pickle.dump(output.histograms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output


if __name__ == "__main__":
    # Get model IDs from command line arguments
    experiment_ids = []
    parallel = 0
    series_id = None
    always_use_cpu = False
    for arg in sys.argv[1:]:
        if arg.startswith("p="):
            parallel = int(arg[2:])
            print(
                "WARNING: make sure to run python -u to see print statements when using parallel processes"
            )
        elif arg.startswith("s="):
            series_id = arg[2:]
        elif arg == "cpu":
            always_use_cpu = True
        elif arg.isdigit():
            id = int(arg)
            experiment_ids.append(id)
    experiments_file = f"./experiments/{series_id}.xlsx"

    if always_use_cpu or parallel >= 2 or not torch.cuda.is_available():
        print("WARNING: Running on CPU. This will be slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if (
        len(experiment_ids) == 0
        or series_id is None
        or not os.path.exists(experiments_file)
    ):
        print(
            "Usage: python run_experiment.py s=[series id] (p=[number of processes]) [experiment_id] [experiment_id] [experiment_id] ..."
        )
        print("Experiment series:")
        print("circle = Circle dataset based")
        print("nmnist = N-MNIST dataset based")
        print("shd = SHD dataset based")
        sys.exit(1)

    # Load model and experiment information
    df_experiments = pd.read_excel(experiments_file, index_col=0)
    df_experiments = df_experiments.loc[
        experiment_ids
    ]  # throws error if experiment_ids not found

    # Run experiments
    if parallel > 0:
        with mp.Pool(parallel) as p:
            p.starmap(_run_configured_experiment, df_experiments.iterrows())
    else:
        for experiment_id, experiment_row in df_experiments.iterrows():
            _run_configured_experiment(experiment_id, experiment_row)

    print("Done")
