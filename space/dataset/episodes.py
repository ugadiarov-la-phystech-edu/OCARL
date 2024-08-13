import os

import torch
import numpy as np
import torchvision.transforms
from torch.utils.data import Dataset
from PIL import Image


class EpisodesDataset(Dataset):
    def __init__(self, path, prefix_path, image_size, to_tensor=True, allow_resize=True):
        self._info_path = os.path.join(path, 'info.npy')
        self._prefix_path = prefix_path
        self._allow_resize = allow_resize
        self._obs_size = tuple(image_size)
        self._info = np.load(self._info_path).item()
        self._episode2offset = [0]
        self._index2episode = []
        for episode_id in sorted(self._info.keys()):
            self._index2episode.extend([episode_id] * len(self._info[episode_id]['obs']))
            self._episode2offset.append(self._episode2offset[-1] + len(self._info[episode_id]['obs']))

        print('Dataset size:', self._episode2offset[-1])

        sample = self._read_image(os.path.join(self._prefix_path, self._info_path[0]['obs'][0]))
        actual_shape = sample.shape[:2]
        print('Actual shape:', actual_shape, 'Expected shape:', self._obs_size, flush=True)
        self._need_resize = self._obs_size != actual_shape
        assert self._allow_resize or not self._need_resize, f'Expected shape={self._obs_size}. Actual shape={actual_shape}'

        self._to_tensor = to_tensor
        if self._need_resize:
            self._resize_transform = torchvision.transforms.Resize(self._obs_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

    @staticmethod
    def _read_image(path):
        with Image.open(path) as image:
            return np.array(image)

    def __getitem__(self, index):
        episode_id = self._index2episode[index]
        obs_id = index - self._episode2offset[episode_id]
        path = os.path.join(self._prefix_path, self._info[episode_id]['obs'][obs_id])
        image = self._read_image(path) / 255.
        if self._to_tensor:
            image = np.transpose(image, [2, 0, 1])  # (c, h, w)
            image = torch.tensor(image, dtype=torch.float32)
            if self._need_resize:
                image = self._resize_transform(image)

        return image

    def __len__(self):
        return self._episode2offset[-1]
