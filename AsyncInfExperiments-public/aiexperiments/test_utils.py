import torch

from aiexperiments.utils import reduce_bins

def test_reduce_bins():
    # Test case 1: Normal case with 4 bins reduced to 2 bins
    bin_counts = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    new_bin_counts = reduce_bins(bin_counts, 2)
    assert torch.equal(new_bin_counts, torch.tensor([3, 7], dtype=torch.float32))

    # Test case 2: Edge case with 1 bin reduced to 1 bin
    bin_counts = torch.tensor([5], dtype=torch.float32)
    new_bin_counts = reduce_bins(bin_counts, 1)
    assert torch.equal(new_bin_counts, torch.tensor([5], dtype=torch.float32))

    # Test case 3: Edge case with 0 bins reduced to 0 bins
    bin_counts = torch.tensor([], dtype=torch.float32)
    new_bin_counts = reduce_bins(bin_counts, 0)
    assert torch.equal(new_bin_counts, torch.tensor([], dtype=torch.float32))

    # Test case 4: Normal case with 3 bins reduced to 2 bins
    bin_counts = torch.tensor([2, 4, 6], dtype=torch.float32)
    new_bin_counts = reduce_bins(bin_counts, 2)
    assert torch.equal(new_bin_counts, torch.tensor([6, 6], dtype=torch.float32))

    # Test case 5: Normal case with 5 bins reduced to 3 bins
    bin_counts = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    new_bin_counts = reduce_bins(bin_counts, 3)
    assert torch.equal(new_bin_counts, torch.tensor([3, 7, 5], dtype=torch.float32))

    # Test case 6: Normal case with 6 bins reduced to 4 bins
    bin_counts = torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.float32)
    new_bin_counts = reduce_bins(bin_counts, 4)
    assert torch.equal(new_bin_counts, torch.tensor([2, 2, 5, 3], dtype=torch.float32))

    # Test case 7: Normal case with 10 bins reduced to 2 bins
    bin_counts = torch.tensor([1, 1, 1, 1, 2, 3, 1, 1, 1, 1], dtype=torch.float32)
    new_bin_counts = reduce_bins(bin_counts, 2)
    assert torch.equal(new_bin_counts, torch.tensor([6, 7], dtype=torch.float32))


