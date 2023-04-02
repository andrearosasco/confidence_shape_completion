import os
import random
import numpy
import torch


def make_reproducible(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    numpy.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class get_init_fn:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        seed = self.seed + worker_id + 1

        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
