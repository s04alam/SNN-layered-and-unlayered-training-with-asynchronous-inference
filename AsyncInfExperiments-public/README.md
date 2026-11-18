# How to install
Tested on: Ubuntu and Windows
1. Clone this repository
2. Open a terminal, navigate to the folder in which the clone was made
3. Create a venv:
   `python -m venv venv`  
4. Source the newly created venv  
    Windows: `./venv/Scripts/activate`  
    Linux: `source venv/bin/activate`
5. Install the required packages: `pip install -r requirements.txt`
6. Install this package: `pip install -e .`
7. Optionally, run the tests to make sure everything has been installed and configured properly: `pytest`


# How to use
The software supports four datasets: `shd`, `nmnist`, `dvsgesture` and `circle`.
For each, the following structures can be defined:
- **Models**, as defined in the file `./models/model_descriptions_[dataset name].xlsx`. These determine the model structure, training method and hyperparameters.
- **Experiments**, as defined in the file `./experiments/[dataset name].xlsx`. These determine the parameters and model combinations used during experiments. The paramaters are defined in config files. More about this later.

You can use one of the three files to train models or run experiments:
- `python train_model.py s=[dataset name] [model id 1] [model id 2] ...`: trains models. Specify the dataset name and model id(s) to train. Models are put in the `./models/[dataset name]` folder, training logs are put in the `./models/recorders/[dataset name]` folder.
- `python run_experiment.py s=[dataset name] [cpu] [experiment id 1] [experiment id 2] ...`: runs experiments. Specify the dataset name and experiment id(s). `cpu` can be added to block GPU use. Results are put in the `./results/[dataset name]` folder.
- `single_run.ipynb`: this notebook can be used to run a modified experiment and directly access the results in memory. Even though experiments can be adjusted, it is still necessary to define them.

The models are all on git, but not the full results. To generate the results from the paper, run the following commands:
- `python run_experiment.py s=shd 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28`
- `python run_experiment.py s=nmnist 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28`
- `python run_experiment.py s=dvsgesture 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28`


## Defining a model
In the models file, the following parameters can be set for each model:
- **model id**: a unique identifier (per dataset) for the model
- **structure**: the structure of the model, only `mlp[number of hidden layers]-[neurons per hidden layer]` is supported. For example, `mlp3-64` is a network with 3 hidden layers and 64 neurons per hidden layer.
- **data**: dataset name
- **training method**: only `bptt` supported
- **optimizer**: only `adam` supported
- **loss function**: can be `rate`, `last_membrane` or `max_membrane`
- **weight decay**: L2 decay constant
- **epochs**: number of epochs to train for
- **batch size**: batch size during training
- **learning rate**: learning rate
- **timestep size**: timestep size
- **tau_m**: membrane time constant, in the same unit as the timestep size
- **surrogate**: the surrogate function (used for bptt), can use `atan-[constant]` or `fast_sigmoid-[constant]`
- **standard nn extensions**: network extensions applied both during training and experimenting
- **training nn extensions**: network extensions applied only during training


## Defining an experiment
Experiments are defined in their respective file. These have the following settings:
- **experiment id**: unique identifier (per dataset) for the experiment
- **model id**: reference to the unique identifier (per dataset) of the model to test
- **inference method**: `matrix` for layered inference or vectorized network asynchrony or `event` for modular network asynchrony
- **config**: name of experiment config to use during the experiment

### Experiment configs
Experiments are for most part configured in their configuration file, which can be shared in between datasets. The `shared_config.json` file defines global configurations, while `experiment_config_[config name].json` defines specific configurations. A specific configuration has priority over the global one.

For each dataset, a definition can be added for an inference method. For each of those, the following can be defined:
- **variable_dt_save_output_with_dt**: if a variable timestep size is used, what timestep to use when saving output
- **limit_n_samples**: number of samples to use for the experiment. If not set, then use all samples.
- **track_order_recordings**: record all spike orderings, very memory intensive
- **order_histogram_bins**: how many bins to have in the order recording histogram
- **experiments**: list of sub-experiments
- Any of the sub-experiment config lines: these will apply to all sub-experiments (except if overwritten).

For each sub-experiment, the following can be defined:
- **batch_size**: batch size
- **timestep_size**: timestep size, can be set to "model", meaning it will use the timestep size from the training config
- **tau_m**: membrane time constant, only specify to overwrite the constant used during training
- **network_extensions**: network extensions to use during the sub-experiment


## Network extensions
Both during training and experiments, network extensions can be used. The following extensions are supported:
| Name |  Inference type | Purpose | First parameter | Second parameter | Third parameter |
|---|---|---|---|---|---|
| barrier_lif | event | Use K-barrier LIFs | K (int) | Send barrier events (bool) | |
| input_barrier | event | Do not allow input asynchrony | | | |
| async_torch | matrix | Enable vectorized network asynchrony (never allows input asynchrony) | Forward group size (int) | Scheduling mode (`random` or `membrane_potential`) |
| no_refrac | matrix (with async_torch) | Do not enforce single spike per forward pass |
| log_spike_dynamics | matrix (with async_torch) | Log spiking dynamics |
| stop_on_output | matrix (with async_torch) | Stop on spiking in the output layer | How many forward steps to allow after having activity in output layer (int) |
| input_dropout | matrix/event | Input spike dropout | p (float) |
| network_dropout | matrix/event | Network spike dropout | p (float) |
| full_dropout | matrix/event | Input and network spike droput | p (float) |
| refrac_dropout | matrix (with async_torch) | Refractory dropout (float) |
| fine_tune | matrix | Import model parameters | Model id to import from (int) |
| spike_sparsity_loss | matrix | Add spike sparsity loss factor | lambda (float) |
| early_stopping | matrix (with async_torch) | Stop after fixed number of forward steps | Number of steps (int) |
| sync_threshold | matrix (with async_torch) | Enable (trainable) sync threshold | Fixed value (or mean) (float) | Standard deviation (default 0) (float) | Trainable (default false) | 
| momentum_noise | matrix (with async_torch) | Enable momentum noise | Noise constant (float) |