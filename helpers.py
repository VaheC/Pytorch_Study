import torch

from torch.utils.data import random_split, WeightedRandomSampler


def index_splitter(n, splits, seed=13):

    idx = torch.arange(n)

    splits_tensor = torch.as_tensor(splits)

    multiplier = n / splits_tensor.sum()

    splits_tensor = (multiplier * splits_tensor).long()

    diff = n - splits_tensor.sum()

    splits_tensor[0] += diff

    torch.manual_seed(seed)

    return random_split(idx, splits_tensor)

def make_balanced_sampler(y):
    _, counts = y.unique(return_counts=True)
    weights = 1.0 / counts.float()
    sample_weights = weights[y.squeeze().long()]
    generator = torch.Generator()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        generator=generator,
        replacement=True
    )

    return sampler