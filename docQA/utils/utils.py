import torch
import inspect
import random
import numpy as np


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


__all__ = [
    'copy_class_parameters',
    'seed_worker'
]
