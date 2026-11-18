import time
from typing import Union
import numpy as np
import tonic
import torch
from aiexperiments.datasets.circle_dataset import CircleDataset
from aiexperiments.datasets.dvs_gesture_pooled import DVSGesture
from aiexperiments.datasets.shd_pooled import SHD
from aiexperiments.datasets.transforms import TMaxFilter, TimeSliceTransform, ToFrame, ToTensorFrameWithTimestep, convert_tensor_to_named_tuple


def test_TMaxFilter_ToTensorFrameWithTimestep():
    dataset = SHD(save_to="./data")
    t_max = 50000
    t_max_transform = TMaxFilter(t_max)
    frame_transform = ToTensorFrameWithTimestep(
        sensor_size=dataset.sensor_size
    )
    transform = tonic.transforms.Compose([t_max_transform, frame_transform])
    sample = dataset[0][0]
    tensor, timesteps = transform(sample)
    event_count = len([event for event in sample if event["t"] < t_max])
    assert tensor.shape == (event_count, 2, SHD.pool_size)

    
def test_ToFrame():
    ## Test if functional with CircleDataset
    circle_dataset = CircleDataset()
    circle_transform = ToFrame(circle_dataset.sensor_size, 1)
    circle_tonic_transform = tonic.transforms.ToFrame(
        sensor_size=circle_dataset.sensor_size,
        time_window=1,
        include_incomplete=True,
    )
    circle_sample = circle_dataset[0][0]
    circle_frame = circle_transform(circle_sample)
    circle_tonic_frame = circle_tonic_transform(circle_sample)
    assert np.array_equal(circle_frame[:-1], circle_tonic_frame) # last frame is bugged in Tonic

    ## Test if functional with SHD
    shd_dataset = SHD(save_to="./data")
    shd_transform = ToFrame(shd_dataset.sensor_size, 1)
    shd_tonic_transform = tonic.transforms.ToFrame(
        sensor_size=shd_dataset.sensor_size,
        time_window=1,
        include_incomplete=True,
    )
    shd_sample = shd_dataset[0][0]
    starttime = time.time()
    shd_frame = shd_transform(shd_sample)
    shd_time = time.time() - starttime
    starttime = time.time()
    shd_tonic_frame = shd_tonic_transform(shd_sample)
    shd_tonic_time = time.time() - starttime
    assert np.array_equal(shd_frame[:-1], shd_tonic_frame)
    print(f"shd_frame: {shd_time}, shd_tonic_frame: {shd_tonic_time}")


def test_convert_tensor_to_named_tuple_compared_to_frame_circle_ts0():
    _test_convert_tensor_to_named_tuple_compared_to_frame_ts0(CircleDataset(), 40, 10000)

def test_convert_tensor_to_named_tuple_compared_to_frame_shd_ts0():
    _test_convert_tensor_to_named_tuple_compared_to_frame_ts0(SHD(save_to="./data"), 100e3, 2)

def _test_convert_tensor_to_named_tuple_compared_to_frame_ts0(dataset: Union[CircleDataset, SHD], t_max, test_n_samples):
    np_transform = tonic.transforms.Compose(
        [TMaxFilter(t_max), tonic.transforms.NumpyAsType(int)]
    )
    ts_frame_transform = tonic.transforms.Compose(
        [
            TMaxFilter(t_max),
            ToTensorFrameWithTimestep(
                sensor_size=dataset.sensor_size
            )
        ]
    )
    for i_sample in range(test_n_samples):
        # apply transforms
        sample = dataset[i_sample][0]
        np_sample = np_transform(sample)
        framed_sample, timesteps = ts_frame_transform(sample)
        converted_sample = convert_tensor_to_named_tuple(
            np_sample, dataset.sensor_size[0], dataset.sensor_size[1], 0
        )
        # check
        t_start = sample["t"].min()
        current_t = t_start
        event_index = -1
        for i_frame, frame in enumerate(framed_sample):
            n_events_in_frame = int(frame.sum().item())
            assert n_events_in_frame == 1, f"{n_events_in_frame} != 1 for sample {i_sample}"
            event_index += n_events_in_frame
            current_t += timesteps[i_frame]
            assert (
                converted_sample[event_index].time == current_t
            ), f"{converted_sample[event_index].time} != {current_t} for sample {i_sample}"
        assert (
            len(converted_sample) == event_index + 1
        ), f"{len(converted_sample)} != {event_index+1} for sample {i_sample}"


def test_convert_tensor_to_named_tuple_compared_to_frame_circle():
    _test_convert_tensor_to_named_tuple_compared_to_frame(CircleDataset(), [1, 3, 10, 40], 50, 10000)
    _test_convert_tensor_to_named_tuple_compared_to_frame(CircleDataset(), [1, 3], 22, 10000)
    
def test_convert_tensor_to_named_tuple_compared_to_frame_shd():
    _test_convert_tensor_to_named_tuple_compared_to_frame(SHD(save_to="./data"), [50000, 10000, 3000], 100000, 100)

def _test_convert_tensor_to_named_tuple_compared_to_frame(dataset: Union[CircleDataset, SHD, DVSGesture], test_ts_sizes, t_max, test_n_samples):
    for ts_size in test_ts_sizes:
        np_transform = tonic.transforms.Compose(
            [TMaxFilter(int(np.ceil(t_max / ts_size)) * ts_size), tonic.transforms.NumpyAsType(int)]
        )
        frame_transform = tonic.transforms.Compose(
            [
                ToFrame(
                    sensor_size=dataset.sensor_size,
                    timestep_size=ts_size,
                ),
                torch.from_numpy,
                TimeSliceTransform(int(np.ceil(t_max / ts_size))),
            ]
        )
        for i_sample in range(test_n_samples):
            # apply transforms
            sample = dataset[i_sample][0]
            np_sample = np_transform(sample)
            framed_sample = frame_transform(sample)
            converted_sample = convert_tensor_to_named_tuple(
                np_sample, dataset.sensor_size[0], dataset.sensor_size[1], ts_size
            )
            # check
            t_start = sample["t"].min() // ts_size * ts_size
            event_index = -1
            for i_frame, frame in enumerate(framed_sample):
                n_events_in_frame = int(frame.sum().item())
                event_index += n_events_in_frame
                if n_events_in_frame > 0:
                    t_frame = i_frame * ts_size + t_start
                    print(t_frame, converted_sample[event_index].time)
                    assert (
                        converted_sample[event_index].time == t_frame
                    ), f"{converted_sample[event_index].time} != {t_frame} for sample {i_sample}"
            assert (
                len(converted_sample) == event_index + 1
            ), f"{len(converted_sample)} != {event_index+1} for sample {i_sample}"


def test_convert_tensor_to_named_tuple_ts0():
    test_n_samples = 10000
    dataset = CircleDataset()
    np_transform = tonic.transforms.NumpyAsType(int)
    for i_sample in range(test_n_samples):
        sample = dataset[i_sample][0]
        np_sample = np_transform(sample)
        converted_sample = convert_tensor_to_named_tuple(
            np_sample, dataset.sensor_size[0], dataset.sensor_size[1], 0
        )
        assert len(converted_sample) == len(
            np_sample
        ), f"{len(converted_sample)} != {len(np_sample)} for sample {i_sample}"
