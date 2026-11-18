from tonic import Dataset
import numpy as np
import os
import torch

class CircleDataset(Dataset):
    sensor_size = (3, 3, 1)
    sample_time = 50 # us?
    ordering = "xytp"

    def __init__(self, *args, save_to=None, train=True, transform=None, target_transform=None, sampling_time=None, sample_length=None, **kwargs):
        if save_to is None:
            save_to = os.path.join(os.path.dirname(__file__), "circle_dataset_data")
        super(CircleDataset, self).__init__(
            save_to=save_to, transform=transform, target_transform=target_transform
        )
        self.train = train

        if train:
            self.data, self.labels = CircleDataset.load_data(save_to + "/train_data.npz")
        else:
            self.data, self.labels = CircleDataset.load_data(save_to + "/test_data.npz")
        # For Slayer
        if sampling_time is not None and sample_length is not None:
            self.sampling_time = sampling_time
            self.num_time_bins = int(sample_length / sampling_time)   
        else:
            self.sampling_time = None
            self.num_time_bins = None


    def __getitem__(self, index):
        events = self.data[index]

        if self.transform is not None:
            events = self.transform(events)

        if self.sampling_time is not None and self.num_time_bins is not None:
            events = self.convert_to_tensor(events) # convert to tensor for Slayer

        label = self.labels[index]
        return events, label
    
    def convert_to_tensor(self, events):
        tensor_representation = torch.zeros(CircleDataset.sensor_size[0], 
                                            CircleDataset.sensor_size[1], 
                                            self.num_time_bins)
        for event in events:
            x, y, t, p = event
            t_bin = t // self.sampling_time
            if t_bin < self.num_time_bins:
                tensor_representation[x, y, t_bin] += p

        return tensor_representation.reshape(-1, self.num_time_bins)

    def __len__(self):
        return len(self.data)

    def load_data(from_file):
        with np.load(from_file, allow_pickle=True) as data:
            labels = data['labels']
            samples = [data[str(i)] for i in range(len(data.files)-1)]
        return samples, labels
