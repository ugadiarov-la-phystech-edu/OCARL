import torch
from torch.utils.data import Dataset


class Dummy(Dataset):
    def __init__(self, num_samples, image_size, to_tensor=True, allow_resize=True):
        self._obs_size = tuple(image_size)
        self._to_tensor = to_tensor
        self._num_samples = num_samples

    def __getitem__(self, index):
        return torch.ones(3, *self._obs_size, dtype=torch.float32, device='cpu')

    def __len__(self):
        return self._num_samples
