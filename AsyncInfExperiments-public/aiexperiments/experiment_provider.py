from typing import List, Tuple
from pandas import Series
from aiexperiments.base_engine import StaticParameters
from aiexperiments.config import SharedExperimentConfig, SubExperimentConfig
from aiexperiments.datasets.shd_pooled import SHD
from aiexperiments.constants import (
    FineTuneNetworkExtension,
    InferenceMethod,
    InputBarrierNetworkExtension,
    NetworkExtension,
    LossFunctionType,
    EventOrderingMethod,
    EventSchedulingMethod,
    NetworkBackend,
    OptimizerType,
    SpikeSparsityLossNetworkExtension,
    get_network_extension_from_list,
    is_network_extension_type_in_list,
    network_extensions_from_string,
)
from aiexperiments.inference.event_driven import (
    EventDrivenInferenceEngine,
    EventDrivenInferenceParameters,
)
from aiexperiments.inference.matrix import (
    MatrixInferenceEngine,
    MatrixInferenceParameters,
)

from aiexperiments.networks.fc_mlp import (
    FcMLPBuilder,
    FcMLPNetworkBuilderParameters,
)
from aiexperiments.training.bptt import (
    BPTTTrainingEngine,
    BPTTTrainingParameters,
)
import os

from aiexperiments.utils import surrogate_type_str_to_surrogate_type


def setup_network(
    structure: str,
    data: str,
    timestep_size: float,  # in seconds
    tau_m: float,  # in microseconds
    load_network_parameters_from: str,
    inhibit_spiking_output: bool,
    surrogate,
    network_extensions: List[NetworkExtension],
    device,
):
    if structure.startswith("mlp"):  # like 'mlp4-64'means 4 layers with 64 neurons each
        # get the number of layers and the number of neurons per layer
        n_hidden_layers, n_hidden = structure.split("-")
        n_hidden_layers = int(n_hidden_layers[3:])
        n_hidden = int(n_hidden)
        net_builder = FcMLPBuilder()
        if data == "nmnist":
            n_in = 2 * 34 * 34
            n_out = 10
        elif data == "circle":
            n_in = 1 * 3 * 3
            n_out = 2

        # these are hardcoded input values, chang them if you want to reduce the input size
        elif data == "shd":
            # n_in = 2 * 1 * 350

            n_in = 2 * 1 * SHD.pool_size  # follows pool_size in shd_pooled.py (was hard-coded 350)
            n_out = 20

        elif data == "smnist":
            n_in = 99 * 1 * 1
            n_out = 10
        elif data == "dvsgesture":
            n_in = 2 * 32 * 32
            n_out = 11
        else:
            raise ValueError(f"Unknown dataset for MLP: {data}")
        net_builder_params = FcMLPNetworkBuilderParameters(
            n_in=n_in,
            n_out=n_out,
            n_hidden_layers=n_hidden_layers,
            n_hidden=n_hidden,
            device=device,
            timestep_size=timestep_size,
            tau_m=tau_m,
            spike_grad=surrogate,
            add_pool4_layer=False,
            load_parameters_from=load_network_parameters_from,
            inhibit_spiking_output=inhibit_spiking_output,
            network_extensions=network_extensions,
        )
    else:
        raise ValueError(f"Unknown network structure: {structure}")
    return net_builder, net_builder_params


def setup_training_experiment(
    model_description: Series,
    series_id: str,
    device=None,
    load_network_parameters_from: str = None,
    **kwargs,  # kwargs are passed to the training parameters
):
    if not model_description["backup"]:
        print("Backup disabled")
    loss_function_type = LossFunctionType.from_string(
        model_description["loss function"]
    )
    optimizer_type = OptimizerType.from_string(model_description["optimizer"])
    standard_network_extensions = network_extensions_from_string(
        str(model_description["standard nn extensions"])
    )
    training_network_extensions = network_extensions_from_string(
        str(model_description["training nn extensions"])
    )
    network_extensions = standard_network_extensions + training_network_extensions
    # Load parameters, if requested
    fine_tune_network_extension: FineTuneNetworkExtension = (
        get_network_extension_from_list(network_extensions, FineTuneNetworkExtension)
    )
    if fine_tune_network_extension is not None:
        load_network_parameters_from = (
            f"./models/{series_id}/{fine_tune_network_extension.source_model_id}.pt"
        )
    if load_network_parameters_from is not None and not os.path.exists(
        load_network_parameters_from
    ):
        print("No network parameters found, starting from scratch...")
        load_network_parameters_from = None
    else:
        print(f"Loading network parameters from {load_network_parameters_from}")
    net_builder, net_builder_params = setup_network(
        device=device,
        tau_m=model_description["tau_m"],
        structure=model_description["structure"],
        data=model_description["data"],
        timestep_size=model_description["timestep size"],
        load_network_parameters_from=load_network_parameters_from,
        surrogate=surrogate_type_str_to_surrogate_type(model_description["surrogate"]),
        inhibit_spiking_output=loss_function_type == LossFunctionType.LAST_MEMBRANE
        or loss_function_type == LossFunctionType.MAX_MEMBRANE,
        network_extensions=network_extensions,
    )
    static_params = StaticParameters()
    if model_description["training method"] == "bptt":
        train_engine = BPTTTrainingEngine(static_params)
        assert device is not None
        spike_sparsity_loss_network_extension: SpikeSparsityLossNetworkExtension = (
            get_network_extension_from_list(
                network_extensions, SpikeSparsityLossNetworkExtension
            )
        )
        train_params = BPTTTrainingParameters(
            device=device,
            dataset_name=model_description["data"],
            loss_function_type=loss_function_type,
            optimizer_type=optimizer_type,
            timestep_size=model_description["timestep size"],
            batch_size=int(model_description["batch size"]),
            learning_rate=model_description["learning rate"],
            num_epochs=int(model_description["epochs"]),
            backup=model_description["backup"],
            num_repeats=int(model_description["repeats"]),
            weight_decay=float(model_description["weight decay"]),
            spike_sparsity_loss_Lambda=spike_sparsity_loss_network_extension.Lambda if spike_sparsity_loss_network_extension is not None else None,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown training method: {model_description['training method']}"
        )

    return train_params, net_builder, net_builder_params, train_engine


def setup_inference_experiment(
    experiment_description: Series,
    model_description: Series,
    load_network_parameters_from: str,
    experiment_config: SharedExperimentConfig,
    sub_experiment_config: SubExperimentConfig,
    device,
    **kwargs,  # kwargs are passed to the inference parameters
):
    if model_description["training method"] == "bptt":
        network_backend = NetworkBackend.SNNTORCH
    else:
        raise ValueError(
            f"Unknown training method: {model_description['training method']}"
        )
    inference_method = InferenceMethod.from_string(
        experiment_description["inference method"]
    )
    loss_function_type = LossFunctionType.from_string(
        model_description["loss function"]
    )
    standard_network_extensions = network_extensions_from_string(
        str(model_description["standard nn extensions"])
    )
    network_extensions = (
        standard_network_extensions + sub_experiment_config.network_extensions
    )
    tau_m = model_description["tau_m"]
    if sub_experiment_config.tau_m is not None:
        tau_m = sub_experiment_config.tau_m
    net_builder, net_builder_params = setup_network(
        structure=model_description["structure"],
        data=model_description["data"],
        tau_m=tau_m,
        timestep_size=sub_experiment_config.timestep_size,
        surrogate=None,
        load_network_parameters_from=load_network_parameters_from,
        inhibit_spiking_output=loss_function_type == LossFunctionType.LAST_MEMBRANE
        or loss_function_type == LossFunctionType.MAX_MEMBRANE,
        network_extensions=network_extensions,
        device=device,
    )
    static_params = StaticParameters()
    if inference_method == InferenceMethod.MATRIX:
        assert device is not None
        inference_params = MatrixInferenceParameters(
            device=device,
            network_backend=network_backend,
            dataset_name=model_description["data"],
            loss_function_type=loss_function_type,
            timestep_size=sub_experiment_config.timestep_size,
            batch_size=sub_experiment_config.batch_size,
            limit_n_samples=sub_experiment_config.limit_n_samples,
            track_order_recordings=experiment_config.track_order_recordings,
            track_membrane_recordings=experiment_config.track_membrane_recordings,
            n_order_histogram_bins=experiment_config.order_histogram_bins,
            variable_dt_save_output_with_dt=experiment_config.variable_dt_save_output_with_dt,
            shuffle=sub_experiment_config.shuffle_samples,
            **kwargs,
        )
        inference_engine = MatrixInferenceEngine(static_params)
    elif inference_method == InferenceMethod.EVENT:
        if sub_experiment_config.batch_size > 1 and (
            experiment_config.track_order_recordings
            or experiment_config.track_membrane_recordings
        ):
            print(
                "WARNING: batch size > 1 can use a lot of memory when tracking order or membrane potentials. This can crash the program."
            )
        if sub_experiment_config.batch_size >= os.cpu_count():
            print(
                "WARNING: batch size >= number of CPUs leads to suboptimal performance."
            )
        add_input_barrier = is_network_extension_type_in_list(
            network_extensions, InputBarrierNetworkExtension
        )
        inference_params = EventDrivenInferenceParameters(
            ordering_method=sub_experiment_config.sync_barrier,
            scheduling_method=sub_experiment_config.event_scheduling_method,
            loss_function_type=loss_function_type,
            dataset_name=model_description["data"],
            timestep_size=sub_experiment_config.timestep_size,
            batch_size=sub_experiment_config.batch_size,
            limit_n_samples=sub_experiment_config.limit_n_samples,
            track_order_recordings=experiment_config.track_order_recordings,
            track_membrane_recordings=experiment_config.track_membrane_recordings,
            n_order_histogram_bins=experiment_config.order_histogram_bins,
            add_input_barrier=add_input_barrier,
            instantly_forward_input=add_input_barrier,
            shuffle=sub_experiment_config.shuffle_samples,
            **kwargs,
        )
        inference_engine = EventDrivenInferenceEngine(static_params)
    else:
        raise ValueError(f"Unknown inference method: {inference_method}")
    return inference_params, net_builder, net_builder_params, inference_engine
