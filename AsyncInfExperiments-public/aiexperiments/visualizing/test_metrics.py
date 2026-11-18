import torch
import numpy as np
from aiexperiments.constants import LossFunctionType
import torch.nn.functional as F

from aiexperiments.visualizing.metrics import ce_last_nonzero_membrane_loss, compute_latency, compute_normalized_order_overlap, last_nonzero_membrane_acc_fn


def test_ce_last_nonzero_membrane_loss():
    # Test 1: Case with nonzero values
    mem_out = torch.tensor(
        [
            [[0.2, 0.1, 0.3], [0.1, 0.4, 0.2]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([2, 1])
    loss_fn = ce_last_nonzero_membrane_loss()
    loss = loss_fn(mem_out, targets)
    expected_loss = F.cross_entropy(
        torch.tensor([[0.2, 0.1, 0.3], [0.1, 0.4, 0.2]]), torch.tensor([2, 1])
    )
    assert torch.isclose(
        loss, expected_loss
    ), f"Expected {expected_loss}, but got {loss}"

    # Test 2: Case with no nonzero values
    mem_out = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([2, 1])
    loss = loss_fn(mem_out, targets)
    expected_loss = F.cross_entropy(
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), torch.tensor([2, 1])
    )
    assert torch.isclose(
        loss, expected_loss
    ), f"Expected {expected_loss}, but got {loss}"

    # Test 3: Case with nonzero values at different timesteps
    mem_out = torch.tensor(
        [
            [[0.2, 0.1, 0.3], [0.1, 0.4, 0.2]],
            [[0.1, 0.2, 0.0], [0.0, 0.3, 0.1]],
            [[0.2, 0.1, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([2, 1])
    loss = loss_fn(mem_out, targets)
    expected_loss = F.cross_entropy(
        torch.tensor([[0.2, 0.1, 0.0], [0.0, 0.3, 0.1]]), torch.tensor([2, 1])
    )
    assert torch.isclose(
        loss, expected_loss
    ), f"Expected {expected_loss}, but got {loss}"


def test_last_nonzero_membrane_acc_fn():
    # Test 1: Case with nonzero values
    mem_out = torch.tensor(
        [
            [[0.2, 0.1, 0.3], [0.1, 0.4, 0.2]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([2, 1])
    acc = last_nonzero_membrane_acc_fn(mem_out, targets)
    expected_acc = torch.tensor(1.0)  # Both samples have correct predictions
    assert torch.isclose(acc, expected_acc), f"Expected {expected_acc}, but got {acc}"

    # Test 2: Case with no nonzero values
    mem_out = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([2, 1])
    acc = last_nonzero_membrane_acc_fn(mem_out, targets)
    expected_acc = torch.tensor(0.0)  # No samples have correct predictions
    assert torch.isclose(acc, expected_acc), f"Expected {expected_acc}, but got {acc}"

    # Test 3: Case with nonzero values at different timesteps
    mem_out = torch.tensor(
        [
            [[0.2, 0.1, 0.3], [0.1, 0.4, 0.2]],
            [[0.1, 0.2, 0.0], [0.0, 0.3, 0.1]],
            [[0.0, 0.1, 0.2], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([2, 1])
    acc = last_nonzero_membrane_acc_fn(mem_out, targets)
    expected_acc = torch.tensor(1.0)  # Both samples have correct predictions
    assert torch.isclose(acc, expected_acc), f"Expected {expected_acc}, but got {acc}"

def test_compute_normalized_order_overlap():
    no_overlap_data = {
        0: torch.tensor([1.0, 0, 0]),
        1: torch.tensor([0, 1.0, 0]),
        2: torch.tensor([0, 0, 1.0]),
    }
    metric = compute_normalized_order_overlap(no_overlap_data)
    expected_metric = np.array([0.0, 0.0, 0.0])
    assert np.allclose(
        metric, expected_metric
    ), f"Expected {expected_metric}, but got {metric}"
    two_neuron_half_overlap_data = {
        0: torch.tensor([0.5, 0.5, 0]),
        1: torch.tensor([0, 1.0, 0]),
        2: torch.tensor([0, 0, 1.0]),
    }
    metric = compute_normalized_order_overlap(two_neuron_half_overlap_data)
    expected_metric = np.array([0.0, 1 / 3 * 0.5, 0.0])
    assert np.allclose(
        metric, expected_metric
    ), f"Expected {expected_metric}, but got {metric}"
    three_neuron_half_overlap_data = {
        0: torch.tensor([0.5, 0.5, 0]),
        1: torch.tensor([0, 1.0, 0]),
        2: torch.tensor([0, 0.5, 0.5]),
    }
    metric = compute_normalized_order_overlap(three_neuron_half_overlap_data)
    expected_metric = np.array([0.0, 1 / 3 * (0.5 + 0.5 + 0.5), 0.0])
    assert np.allclose(
        metric, expected_metric
    ), f"Expected {expected_metric}, but got {metric}"
    even_overlap_data = {
        0: torch.tensor([1 / 3, 1 / 3, 1 / 3]),
        1: torch.tensor([1 / 3, 1 / 3, 1 / 3]),
        2: torch.tensor([1 / 3, 1 / 3, 1 / 3]),
    }
    metric = compute_normalized_order_overlap(even_overlap_data)
    expected_metric = np.array(
        [
            1 / 3 * (1 / 3 + 1 / 3 + 1 / 3),
            1 / 3 * (1 / 3 + 1 / 3 + 1 / 3),
            1 / 3 * (1 / 3 + 1 / 3 + 1 / 3),
        ]
    )
    assert np.allclose(
        metric, expected_metric
    ), f"Expected {expected_metric}, but got {metric}"
    full_overlap_data = {
        0: torch.tensor([0, 1.0, 0]),
        1: torch.tensor([0, 1.0, 0]),
        2: torch.tensor([0, 1.0, 0]),
    }
    metric = compute_normalized_order_overlap(full_overlap_data)
    expected_metric = np.array([0.0, 1.0, 0.0])
    assert np.allclose(
        metric, expected_metric
    ), f"Expected {expected_metric}, but got {metric}"


def test_compute_latency():
    output_recordings = torch.tensor(
        [[[0.2, 0.1], [0.1, 0.4]], [[0.1, 0.3], [0.4, 0.2]]]
    )
    labels = torch.tensor([1, 0])
    true_acc = 0.5
    inference_timestep_size = 1
    latency_timestep_size = 1
    acc_fn_type = LossFunctionType.RATE
    latency = compute_latency(
        output_recordings,
        #labels,
        #true_acc,
        inference_timestep_size,
        latency_timestep_size,
        #acc_fn_type,
    )
    print(latency)
