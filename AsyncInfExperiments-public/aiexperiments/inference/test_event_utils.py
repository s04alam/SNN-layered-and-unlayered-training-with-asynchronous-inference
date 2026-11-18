import torch

from aiexperiments.inference.event_utils import EventWithValue, classification_events_to_output_recordings, classification_events_to_array, event_tensor_to_fixed_timestep_tensor, membrane_potential_recordings_to_output_recordings

def test_event_tensor_to_fixed_timestep_tensor():
    event_tensor = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [0.0, 0.0]]], dtype=torch.float32
    )
    timesteps = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
    timestep_size = 1
    max_n_timesteps = 3
    only_keep_last = False
    output = event_tensor_to_fixed_timestep_tensor(
        event_tensor, timesteps, timestep_size, max_n_timesteps, only_keep_last
    )
    expected_output = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [0.0, 0.0]]]
    )
    assert torch.equal(output, expected_output), f"{output} != {expected_output}"

    event_tensor = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[4.0, 5.0], [0.0, 0.0]]], dtype=torch.float32
    )
    timesteps = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
    timestep_size = 2
    max_n_timesteps = 3
    only_keep_last = False
    output = event_tensor_to_fixed_timestep_tensor(
        event_tensor, timesteps, timestep_size, max_n_timesteps, only_keep_last
    )
    expected_output = torch.tensor(
        [[[5.0, 7.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    assert torch.equal(output, expected_output), f"{output} != {expected_output}"

    event_tensor = torch.tensor(
        [[[1.0, 2.0], [0.0, 0.0]], [[4.0, 5.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    timesteps = torch.tensor([[1, 1], [1, 0], [0, 0]], dtype=torch.float32)
    timestep_size = 2
    max_n_timesteps = 3
    only_keep_last = False
    output = event_tensor_to_fixed_timestep_tensor(
        event_tensor, timesteps, timestep_size, max_n_timesteps, only_keep_last
    )
    expected_output = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[4.0, 5.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    assert torch.equal(output, expected_output), f"{output} != {expected_output}"

    event_tensor = torch.tensor(
        [[[1.0, 2.0], [0.0, 0.0]], [[4.0, 5.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    timesteps = torch.tensor([[1, 1], [1, 0], [0, 0]], dtype=torch.float32)
    timestep_size = 2
    max_n_timesteps = 3
    only_keep_last = True
    output = event_tensor_to_fixed_timestep_tensor(
        event_tensor, timesteps, timestep_size, max_n_timesteps, only_keep_last
    )
    expected_output = torch.tensor(
        [[[1.0, 2.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
    )
    assert torch.equal(output, expected_output), f"{output} != {expected_output}"



def test_classification_events_to_array():
    test_events = [
        EventWithValue(0, 0, 1),
        EventWithValue(1, 0, 1),
        EventWithValue(1, 1, 1),
    ]
    max_n_timesteps = 3
    n_classes = 2
    timestep_size = 1
    tensor = classification_events_to_array(
        test_events, timestep_size, max_n_timesteps, n_classes
    )
    assert tensor.shape == (max_n_timesteps + 1, 1, n_classes)
    assert tensor[0, 0, 0] == 1
    assert tensor[0, 0, 1] == 1
    assert tensor[1, 0, 0] == 0
    assert tensor[1, 0, 1] == 1
    assert tensor[2, 0, 0] == 0
    assert tensor[2, 0, 1] == 0

    test_events = [EventWithValue(1, 1, 1)]
    max_n_timesteps = 3
    n_classes = 2
    timestep_size = 1
    tensor = classification_events_to_array(
        test_events, timestep_size, max_n_timesteps, n_classes
    )
    assert tensor.sum(axis=0).argmax(axis=1) == 1

    test_events = [
        EventWithValue(0, 0, 1),
        EventWithValue(1, 0, 1),
        EventWithValue(1, 99999, 1),
    ]
    max_n_timesteps = 33
    n_classes = 2
    timestep_size = 3000
    tensor = classification_events_to_array(
        test_events, timestep_size, max_n_timesteps, n_classes
    )
    assert tensor[0, 0, 0] == 1
    assert tensor[0, 0, 1] == 1
    assert tensor[33, 0, 0] == 0
    assert tensor[33, 0, 1] == 1


def test_classification_events_to_output_recordings():
    events = [
        EventWithValue(0, 0, 1),
        EventWithValue(0, 1, 1),
        EventWithValue(1, 0, 1),
        EventWithValue(1, 1, 1),
    ]
    timestep_size = 1
    max_n_timesteps = 2
    n_neurons = 2
    output_recordings = classification_events_to_output_recordings(
        events, timestep_size, max_n_timesteps, n_neurons
    )
    assert len(output_recordings) == max_n_timesteps + 1
    assert output_recordings[0].shape == (1, 2)
    assert output_recordings[0][0][0] == 1
    assert output_recordings[0][0][1] == 1
    assert output_recordings[1][0][0] == 1
    assert output_recordings[1][0][1] == 1
    events = []
    output_recordings = classification_events_to_output_recordings(
        events, timestep_size, max_n_timesteps, n_neurons
    )
    assert len(output_recordings) == max_n_timesteps + 1


def test_membrane_potential_recordings_to_output_recordings():
    ts = [[0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8], [0, 1, 2, 3, 4, 8]]
    ys = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 11, 12, 13, 15, 16]]
    timestep_size = 1
    max_n_timesteps = 20
    output_recordings = membrane_potential_recordings_to_output_recordings(
        ts, ys, timestep_size, max_n_timesteps
    )
    assert len(output_recordings) == max_n_timesteps + 1
    assert output_recordings[0].shape == (1, 2)
    assert output_recordings[8][0][0] == 10
    assert output_recordings[8][0][1] == 16
    ts = []
    ys = []
    output_recordings = membrane_potential_recordings_to_output_recordings(
        ts, ys, timestep_size, max_n_timesteps
    )
    assert len(output_recordings) == max_n_timesteps + 1
    ts = [[0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8], [0, 1, 2, 3, 4, 8]]
    ys = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 11, 12, 13, 15, 16]]
    timestep_size = 5
    max_n_timesteps = 4
    output_recordings = membrane_potential_recordings_to_output_recordings(
        ts, ys, timestep_size, max_n_timesteps
    )
    assert len(output_recordings) == max_n_timesteps + 1
    assert output_recordings[0].shape == (1, 2)
    assert output_recordings[0][0][0] == 5
    assert output_recordings[0][0][1] == 15
