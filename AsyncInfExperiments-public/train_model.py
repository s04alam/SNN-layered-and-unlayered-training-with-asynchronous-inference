import os
import traceback

from aiexperiments.experiment_provider import setup_training_experiment

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import pandas as pd
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Get model IDs from command line arguments
dataset = sys.argv[1]
series_id = None
model_ids = []
for arg in sys.argv[1:]:
    if arg.startswith("s="):
        series_id = arg[2:]
    elif arg.isdigit():
        id = int(arg)
        model_ids.append(id)

if len(model_ids) == 0 or series_id is None:
    print("Usage: python train_model.py s=[nmnist/shd/...] [model_id] [model_id] [model_id] ...")
    sys.exit(1)

# Load model information
df_models = pd.read_excel(f"./models/model_descriptions_{series_id}.xlsx", index_col=0) # how can it load model information if you are training the model yourself?
df_models = df_models.loc[model_ids]  # throws error if model_id not found

# Train models
for model_id, row in df_models.iterrows():
    model_id = int(model_id)
    model_id_str = f"{model_id}"
    model_path = f"./models/{series_id}/{model_id_str}.pt"
    recorder_path = f"./models/recorders/{series_id}/{model_id_str}.pickle"
    try:
        (
            train_params,
            net_builder,
            net_builder_params,
            train_engine,
        ) = setup_training_experiment(
            model_description=row,
            series_id=series_id,
            device=device,
            load_network_parameters_from=None,
            save_network_parameters_to=model_path,
            load_train_recorder_from=None,
            save_train_recorder_to=recorder_path,
        )
        train_engine.train(train_params, net_builder, net_builder_params)
    except Exception as e:
        print("Error:", e, type(e))
        traceback.print_exc() 
        break

print("Done")
