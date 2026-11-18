from abc import ABC
from abc import abstractmethod
import tonic.transforms as transforms
import tonic
from tonic.dataset import Dataset
from torch.utils.data import DataLoader
import torch
from aiexperiments.constants import PLOT_LOSS_KEY, QUIT_KEY, DatasetProvider
from aiexperiments.datasets.circle_dataset import CircleDataset
from aiexperiments.datasets.dvs_gesture_pooled import DVSGesture
from aiexperiments.datasets.shd_pooled import SHD
from aiexperiments.datasets.transforms import NoPaddingCollate, PadTensorsWithTimestep, TMaxFilter, TimeSliceTransform, ToFrame, ToTensorFrameWithTimestep
from aiexperiments.visualizing.recording.recorder import Recorder
import numpy as np


# for training my own model I have to reduce the input size to 128, such that I will have SHD128 alongside SHD

class StaticParameters:
    """
    Parameters that are not changed during runtime.

    Attributes:
        verbose: Whether to print additional information during runtime.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose


class EngineParameters:
    """
    Attributes:
        dataset_name: The name of the dataset to use for training.
        dataset_provider: The type of dataset to use for training.
        timestep_size: The number of timesteps to use for training in microseconds, may be 0 if not required.
        force_int_timestep: Whether to force the timestep size to be an integer.
        batch_size: The number of samples to use per batch for training.
        fix_sample_time_to: Fix the temporal length of the train samples, given in microseconds. If None, default for the dataset will be used.
        save_data_to: The path to save the testing/trainig data to.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_provider: DatasetProvider,
        timestep_size,
        batch_size,
        force_int_timestep=True,
        fix_sample_time_to=None,
        shuffle=True,
        save_data_to="./data",
    ):
        self.dataset_name = dataset_name
        self.dataset_provider = dataset_provider
        if timestep_size is not None:
            self.timestep_size = timestep_size
            if force_int_timestep:
                self.timestep_size = int(self.timestep_size)
        else:
            self.timestep_size = 0
        self.batch_size = batch_size
        self.save_data_to = save_data_to
        self.shuffle = shuffle

        # Calculate slice length based on timestep size and time length to use
        if dataset_name == "nmnist":
            if dataset_provider == DatasetProvider.TONIC_FRAME or dataset_provider == DatasetProvider.TONIC_EVENTS:
                self.dataset = tonic.datasets.NMNIST
            else:
                raise NotImplementedError("Dataset provider not implemented")
            self.sensor_size = self.dataset.sensor_size
            dataset_sample_time = 0.1 * 1e6
        elif dataset_name == "smnist":
            if dataset_provider == DatasetProvider.TONIC_FRAME or dataset_provider == DatasetProvider.TONIC_EVENTS:
                self.dataset = tonic.datasets.SMNIST
            else:
                raise NotImplementedError("Dataset provider not implemented")
            self.sensor_size = (99, 1, 1)
            dataset_sample_time = 1.2 * 1e6
        elif dataset_name == "circle":
            self.save_data_to = None # circle dataset does not support saving data, so disable it
            if (
                dataset_provider == DatasetProvider.TONIC_EVENTS
                or dataset_provider == DatasetProvider.TONIC_FRAME
            ):
                self.dataset = CircleDataset
            else:
                raise NotImplementedError("Dataset provider not implemented")
            self.sensor_size = self.dataset.sensor_size
            dataset_sample_time = CircleDataset.sample_time
        elif dataset_name == "shd":
            if dataset_provider == DatasetProvider.TONIC_FRAME or dataset_provider == DatasetProvider.TONIC_EVENTS:
                self.dataset = SHD
            else:
                raise NotImplementedError("Dataset provider not implemented")
            self.sensor_size = self.dataset.sensor_size
            dataset_sample_time = 1.0 * 1e6
        elif dataset_name == "dvsgesture":
            if dataset_provider == DatasetProvider.TONIC_FRAME or dataset_provider == DatasetProvider.TONIC_EVENTS:
                self.dataset = DVSGesture
            else:
                raise NotImplementedError("Dataset provider not implemented")
            self.sensor_size = self.dataset.sensor_size
            dataset_sample_time = 3.0 * 1e6 # Half of 6 seconds, the total time of the gesture
        else:
            raise ValueError("dataset not implemented")
        if fix_sample_time_to is not None:
            self.fix_sample_time_to = fix_sample_time_to
        else:
            self.fix_sample_time_to = dataset_sample_time
        if self.timestep_size > 1:
            self.fix_sample_length_to = int(np.ceil(self.fix_sample_time_to / timestep_size)) # round up
        else:
            self.fix_sample_length_to = int(self.fix_sample_time_to)



class BaseEngine(ABC):
    def __init__(self, static_params: StaticParameters):
        self.static_params = static_params

    def load_data(
        self,
        engine_params: EngineParameters,
        is_training,
    ) -> DataLoader:
        # Prepare the dataset to use
        first_saccade_only = None
        if engine_params.dataset_name == "nmnist":
            first_saccade_only = True
        sensor_size = engine_params.sensor_size
        if self.static_params.verbose:
            print(
                "Fixing sample length to "
                + str(engine_params.fix_sample_length_to)
                + " steps, each step "
                + str(engine_params.timestep_size)
                + " microseconds"
            )
        # Prepare the transforms and args to use
        args = {
            "save_to": engine_params.save_data_to,
            "transform": None,
            "train": is_training,
        }
        collate_fn = None
        if engine_params.dataset_provider == DatasetProvider.TONIC_FRAME:
            if engine_params.timestep_size >= 1:
                frame_transforms = [
                    ToFrame(
                        sensor_size=engine_params.sensor_size,
                        timestep_size=engine_params.timestep_size,
                    ),
                    torch.from_numpy,
                    TimeSliceTransform(engine_params.fix_sample_length_to),
                ]
                transform = transforms.Compose(frame_transforms)
                collate_fn = tonic.collation.PadTensors(batch_first=False)
            else:
                t_max_transform = TMaxFilter(t_max=engine_params.fix_sample_length_to * (engine_params.timestep_size if engine_params.timestep_size > 0 else 1))
                frame_transform = ToTensorFrameWithTimestep(
                    sensor_size=sensor_size
                )
                transform = transforms.Compose([t_max_transform, frame_transform])
                collate_fn = PadTensorsWithTimestep(batch_first=False)
            args["transform"] = transform
        elif engine_params.dataset_provider == DatasetProvider.TONIC_EVENTS:
            t_max_transform = TMaxFilter(t_max=engine_params.fix_sample_length_to * (engine_params.timestep_size if engine_params.timestep_size > 0 else 1))
            np_transform = tonic.transforms.NumpyAsType(int)
            transform = transforms.Compose([t_max_transform, np_transform])
            collate_fn = NoPaddingCollate()
            args["transform"] = transform
        # Load the dataset and create the dataloader
        if first_saccade_only is not None:
            args["first_saccade_only"] = first_saccade_only

        # this line has been changed
        dataset = engine_params.dataset(**args)
        dataset = InMemoryCache(dataset)  # cache converted samples in RAM (speed only)
        # 
        
        dataloader = DataLoader(
            dataset,
            shuffle=engine_params.shuffle,
            batch_size=engine_params.batch_size,
            collate_fn=collate_fn,
        )
        return dataloader

# this has been added such that that the batches are sliced once for one epoch and then reused

class InMemoryCache(torch.utils.data.Dataset):
    """Stores each sample after its first conversion and returns the stored copy afterwards.
    Only valid because the transform pipeline is deterministic."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index not in self.cache:
            self.cache[index] = self.dataset[index]
        return self.cache[index]