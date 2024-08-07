import h5py
import torch
import numpy as np
import torchvision.transforms
from torch.utils.data import Dataset


class Hdf5Dataset(Dataset):
    def __init__(self, path, mode, image_size, to_tensor=True, allow_resize=True):
        assert mode in ('train', 'val')
        mode2split = {'train': 'TrainingSet', 'val': 'ValidationSet'}
        self._mode = mode
        self._data = h5py.File(path, 'r')[mode2split[self._mode]]
        self._allow_resize = allow_resize
        self._obs_size = tuple(image_size)
        actual_shape = self._data["obss"].shape[1:3]
        self._need_resize = self._obs_size != actual_shape
        assert self._allow_resize or not self._need_resize, f'Expected shape={self._obs_size}. Actual shape={actual_shape}'

        self._to_tensor = to_tensor
        self._num_samples = self._data["obss"].shape[0]
        if self._need_resize:
            self._resize_transform = torchvision.transforms.Resize(self._obs_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

    def __getitem__(self, index):
        image = self._data['obss'][index] / 255.
        if self._to_tensor:
            image = np.transpose(image, [2, 0, 1])  # (c, h, w)
            image = torch.tensor(image, dtype=torch.float32)
            if self._need_resize:
                image = self._resize_transform(image)

        return image

    def __len__(self):
        return self._num_samples
