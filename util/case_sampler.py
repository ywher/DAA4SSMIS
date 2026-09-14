"""Distribute complete volumes for evaluation without padding or duplicates."""

from torch.utils.data import Sampler
import torch.distributed as dist

from util.cases import group_cases


class CaseSampler(Sampler):
    def __init__(self, dataset, rank=None, world_size=None):
        rank = dist.get_rank() if rank is None else rank
        world_size = dist.get_world_size() if world_size is None else world_size
        groups = list(group_cases(dataset.ids).values())
        self.indices = [
            index for indices in groups[rank::world_size] for index in indices
        ]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
