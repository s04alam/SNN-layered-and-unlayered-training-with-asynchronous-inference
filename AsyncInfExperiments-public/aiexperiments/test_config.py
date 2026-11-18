import json
from aiexperiments.config import load_config_from_json

from aiexperiments.constants import EventOrderingMethod, InferenceMethod

def test_config():
    shared_config_json_obj = json.loads(
        """
    {
        "nmnist": {
            "matrix": {
                "variable_dt_save_output_with_dt": 100,
                "order_histogram_bins": 100
            },
            "event": {
                "experiments": [
                    {
                        "timestep_size": 100000,
                        "sync_barrier": "layer"
                    },
                    {
                        "timestep_size": 3000
                    }
                ],
                "variable_dt_save_output_with_dt": 1,
                "limit_n_samples": 10,
                "track_order_recordings": true,
                "order_histogram_bins": 100
            }
        }
    }
    """
    )
    experiment_config_json_obj = json.loads(
        """
    {
        "nmnist": {
            "matrix": {
                "experiments": [
                    {
                        "timestep_size": 100000,
                        "batch_size": 1024
                    },
                    {
                        "timestep_size": 10000,
                        "batch_size": 1024
                    }
                ]
            },
            "event": {
                "limit_n_samples": 100
            }
        }
    }
        """
    )
    shared_experiment_config = load_config_from_json(shared_config_json_obj, experiment_config_json_obj)
    print(
        shared_experiment_config.shared_experiment_configs["nmnist"][
            str(InferenceMethod.MATRIX)
        ]
        .sub_experiments[0]
        .timestep_size
    )
    assert (
        shared_experiment_config.shared_experiment_configs["nmnist"][
            str(InferenceMethod.EVENT)
        ]
        .sub_experiments[0]
        .limit_n_samples
        == 100
    )
    assert (
        shared_experiment_config.shared_experiment_configs["nmnist"][
            str(InferenceMethod.MATRIX)
        ]
        .sub_experiments[0]
        .limit_n_samples
        is None
    )
    assert (
        shared_experiment_config.shared_experiment_configs["nmnist"][
            str(InferenceMethod.EVENT)
        ]
        .sub_experiments[0]
        .sync_barrier
        == EventOrderingMethod.PER_LAYER
    )
