from typing import List, Union
import numpy as np
import tonic
import torch
from aiexperiments.inference.event_utils import EventWithValue

class TimeSliceTransform:
    """
    Transform to slice the time dimension of a tensor.
    """

    def __init__(self, time_steps):
        self.time_steps = time_steps

    def __call__(self, x: torch.Tensor):
        assert len(x.shape) == 4 or len(x.shape) == 3, "Input must be a 3D or 4D tensor"
        if x.shape[0] < self.time_steps:
            if len(x.shape) == 4:
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            self.time_steps - x.shape[0],
                            x.shape[1],
                            x.shape[2],
                            x.shape[3],
                            dtype=x.dtype,
                        ),
                    ]
                )
            else:
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            self.time_steps - x.shape[0], x.shape[1], x.shape[2], dtype=x.dtype
                        ),
                    ]
                )
        elif x.shape[0] > self.time_steps:
            x = x[: self.time_steps]
        return x
    
class TMaxFilter:
    # t_max is on top of t_min, which is not necessarily 0
    # t_max is given in the same unit as the events, but should be a multiple of timestep_size (ceiled)
    def __init__(self, t_max):
        self.t_max = t_max

    def __call__(self, events: np.ndarray):
        filtered_events = np.array([], dtype=events.dtype)
        if len(events) > 0:
            t_min = events["t"].min()
            for event in events:
                if self.t_max is None or event["t"] < t_min+self.t_max:
                    filtered_events = np.append(filtered_events, event)
        return filtered_events


class ToTensorFrameWithTimestep(tonic.transforms.ToFrame): # 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, event_count=1, include_incomplete=True, **kwargs)

    def __call__(self, events: np.ndarray):
        timesteps = []
        filtered_events = np.array([], dtype=events.dtype)
        if len(events) > 0:
            timesteps.append(0)
            filtered_events = np.append(filtered_events, events[0])
            timestamps = [event["t"] for event in events]
            for i in range(1, len(timestamps)):
                dt = timestamps[i] - timestamps[i - 1]
                timesteps.append(dt)
                filtered_events = np.append(filtered_events, events[i])

        return (
            torch.from_numpy(super().__call__(filtered_events)),
            torch.tensor(timesteps),
        )
    
class NoPaddingCollate:
    def __call__(self, batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)
        return [data, target]

class PadTensorsWithTimestep:  # most is copied from tonic PadTensors, added timesteps_output
    def __init__(self, batch_first: bool = True):
        self.batch_first = batch_first

    def __call__(self, batch):
        samples_output = []
        targets_output = []
        timesteps_output = []

        max_length = max([sample.shape[0] for (sample, _), _ in batch])
        for (sample, timesteps), target in batch:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if not isinstance(timesteps, torch.Tensor):
                timesteps = torch.tensor(timesteps)

            # pad sample
            padding_tensor = torch.zeros(
                max_length - sample.shape[0], *sample.shape[1:], device=sample.device
            )
            sample = torch.cat((sample, padding_tensor))

            # pad timesteps
            padding_tensor = torch.zeros(
                max_length - timesteps.shape[0], device=timesteps.device
            )
            timesteps = torch.cat((timesteps, padding_tensor))

            timesteps_output.append(timesteps)
            samples_output.append(sample)
            targets_output.append(target)
        return (
            torch.stack(samples_output, 0 if self.batch_first else 1),
            torch.stack(timesteps_output, 0 if self.batch_first else 1),
            torch.tensor(targets_output, device=target.device),
        )
    

def generate_time_windows(events_time: np.ndarray, timestep_size: int):
    t_start = events_time.min()
    t_end = events_time.max()+timestep_size
    n_slices = int(np.ceil(((t_end - t_start) - timestep_size) / timestep_size)) + 1

    window_start_times = np.arange(n_slices) * timestep_size + t_start
    window_end_times = window_start_times + timestep_size
    time_windows = list(zip(window_start_times, window_end_times))
    return time_windows

class ToFrame():
    def __init__(self, sensor_size, timestep_size):
        self.sensor_size = sensor_size
        assert len(sensor_size) == 3
        self.timestep_size = max(timestep_size, 1)

    def __call__(self, events: np.ndarray):
        events_time = events['t']
        time_windows = generate_time_windows(events_time, self.timestep_size)
        num_windows = len(time_windows)

        has_y = 'y' in events.dtype.names
        # p, y, x or p, x
        frame_shape = (self.sensor_size[2], self.sensor_size[1], self.sensor_size[0]) if has_y else (self.sensor_size[2], self.sensor_size[0])
        fix_polarity = self.sensor_size[2] == 1
        all_frames = np.zeros((num_windows,) + frame_shape, dtype=np.int16)

        for i, (start, end) in enumerate(time_windows):
            mask = (events_time >= start) & (events_time < end)
            window_events = events[mask]
            if fix_polarity:
                polarities = np.zeros_like(window_events['p'], dtype=np.int64)
            else:
                polarities = window_events['p'].astype(np.int64)
            if has_y:
                np.add.at(all_frames[i], (polarities, window_events['y'], window_events['x']), 1)
            else:
                np.add.at(all_frames[i], (polarities, window_events['x']), 1)
        return all_frames
    

def convert_tensor_to_named_tuple(
    events: Union[torch.Tensor, np.ndarray], # after tonic.transforms.NumpyAsType(int)
    x_max,
    y_max,
    timestep_size,
) -> List[EventWithValue]:
    if timestep_size == 0:
        timestep_size = 1
    if isinstance(events, torch.Tensor):
        events = events.numpy()
    # input shape: [n_events, 4] where 4 = [x, y, time, polarity]
    if events.shape[1] == 4:
        events_indexed = (
            events[:, 0] + events[:, 1] * x_max + events[:, 3] * x_max * y_max
        )
        events_time = events[:, 2]
    elif (
        events.shape[1] == 3
    ):  # list of arrays with shape: [n_events, 3] where 3 = [time, index, polarity]
        events_indexed = events[:, 1] + events[:, 2] * x_max
        events_time = events[:, 0]
    else:
        raise Exception("Unexpected shape of events tensor")
    time_windows = generate_time_windows(events_time, timestep_size)

    output_events = []
    for i in range(len(events_time)):
        for start, end in time_windows:
            if start <= events_time[i] < end:
                output_events.append(
                    EventWithValue(
                        events_indexed[i], start // timestep_size * timestep_size, 1
                    )
                )
                break
    return output_events