import argparse

import h5py
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from space_wrapper import SpaceWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--space_config_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--obs_size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--save_path', type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    space_wrapper = SpaceWrapper(args.space_config_path)
    space_wrapper.fg.requires_grad_(False)
    batch_size = args.batch_size
    key = 'TrainingSet' if args.split == 'train' else 'ValidationSplit'
    train_data = h5py.File(args.dataset_path, 'r')[key]['obss']

    transform = torchvision.transforms.Resize((args.obs_size, args.obs_size), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    threshold = args.threshold
    size = train_data.shape[0]
    pres_scores = []
    pres_glimpses = []
    for i in tqdm(range(0, size, batch_size)):
        batch = torch.as_tensor(train_data[i: i + batch_size], dtype=torch.float32, device='cuda') / 255
        batch = transform(batch.permute((0, 3, 1, 2)))
        res = space_wrapper.fg(batch, 1000000, glimpse_only=True)
        z_pres, glimpse, z_shift = res['z_pres'], res['glimpse'], res['z_shift']
        scores = z_pres[z_pres > threshold].detach().cpu().numpy()
        glimpses = glimpse[z_pres > threshold].detach().cpu().numpy()
        # pres_scores.append(scores)
        pres_glimpses.append(glimpses)

    # pres_scores = np.concatenate(pres_scores)
    pres_glimpses = np.concatenate(pres_glimpses)
    np.save(args.save_path, pres_glimpses)

    # plt.hist(pres_scores, bins=100)
    # plt.title('histogram')
    # plt.show()
