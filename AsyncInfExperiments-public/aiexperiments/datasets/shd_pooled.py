import tonic
import os
import h5py
from tonic.io import make_structured_array

# This subclass of SHD will pool the events from 700x1x2 to 350x1x2
class SHD(tonic.datasets.SHD):
    pool_size = 128 # changed this to 128 from 350, signed weights BSS2, have to change if retraining Roels pretrained models
    sensor_size = (pool_size, 1, 2)

    def __init__(self, *args, **kwargs):
        SHD.sensor_size = (SHD.pool_size, 1, 2) # TODO: this is probably not needed
        super().__init__(*args, **kwargs)
        self.sensor_size = (SHD.pool_size, 1, 2)
        self.pool_transform = tonic.transforms.Downsample(
            sensor_size=tonic.datasets.SHD.sensor_size, target_size=(SHD.pool_size, 1)
        )

    def __getitem__(self, index): # copied from tonic.datasets.SHD, but with pool_transform
        file = h5py.File(os.path.join(self.location_on_system, self.data_filename), "r")
        # adding artificial polarity of 1 and convert to microseconds
        events = make_structured_array(
            file["spikes/times"][index] * 1e6,
            file["spikes/units"][index],
            1,
            dtype=self.dtype,
        )
        target = file["labels"][index].astype("int64") #needed to change that to int64
        events = self.pool_transform(events)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target