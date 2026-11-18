import tonic
import numpy as np

# This subclass of SHD will pool the events from 700x1x2 to 350x1x2
class DVSGesture(tonic.datasets.DVSGesture):
    sensor_size = (32, 32, 2)

    def __init__(self, *args, **kwargs):
        DVSGesture.sensor_size = (32, 32, 2) # TODO: this is probably not needed
        super().__init__(*args, **kwargs)
        self.sensor_size = (32, 32, 2)
        self.pool_transform = tonic.transforms.Downsample(
            spatial_factor=0.25
        )

    def __getitem__(self, index):
        """
        (copied from tonic implementation)
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = np.load(self.data[index])
        events[:, 3] *= 1000  # convert from ms to us
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
        events = self.pool_transform(events)
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target