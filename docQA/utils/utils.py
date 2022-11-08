import torch
import inspect
import random
import numpy as np
from torch import Tensor, device


def copy_class_parameters(class_a, class_b):
    for param in list(set(dir(class_b)) - set(dir(class_a))):
        value = getattr(class_b, param)
        if inspect.ismethod(value):
            continue

        setattr(class_a, param, value)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def batch_to_device(batch, target_device: device):
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch
