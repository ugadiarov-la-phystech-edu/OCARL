import argparse

import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from space.dataset import EpisodesDataset, Hdf5Dataset
from space_wrapper import SpaceWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--space_config_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, choices=['hdf5', 'episodes'], required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--prefix_path', type=str, required=False)
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--obs_size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, required=False)
    parser.add_argument('--wandb_group', type=str, default='Test group')
    parser.add_argument('--wandb_run_name', type=str, default='run-0')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    space_wrapper = SpaceWrapper(args.space_config_path, ckpt_path=args.checkpoint_path)
    space_wrapper.fg.requires_grad_(False)
    batch_size = args.batch_size
    if args.dataset_type == 'hdf5':
        dataset = Hdf5Dataset(path=args.dataset_path, image_size=(args.obs_size, args.obs_size), mode=args.split,
                              to_tensor=True)
    elif args.dataset_type == 'episodes':
        assert args.prefix_path is not None
        dataset = EpisodesDataset(args.dataset_path, prefix_path=args.prefix_path,
                                  image_size=(args.obs_size, args.obs_size), to_tensor=True)
    else:
        assert False

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    pres_scores = []
    pres_glimpses = []
    threshold = args.threshold
    for batch in tqdm(dataloader):
        batch = batch.to(space_wrapper.device)
        res = space_wrapper.fg(batch, 1000000, glimpse_only=True)
        z_pres, glimpse, z_shift = res['z_pres'], res['glimpse'], res['z_shift']
        scores = z_pres[z_pres > threshold].detach().cpu().numpy()
        glimpses = glimpse[z_pres > threshold].detach().cpu().numpy()
        pres_scores.append(scores)
        pres_glimpses.append(glimpses)

    pres_scores = np.concatenate(pres_scores)
    pres_glimpses = np.concatenate(pres_glimpses)
    np.save(args.save_path, pres_glimpses)

    if args.wandb_project:
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name,
            resume='never',
            sync_tensorboard=True,
        )

        table = wandb.Table(data=pres_scores.reshape((-1, 1)), columns=['scores'])
        wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_run_name)
        wandb.log({'histogram': wandb.plot.histogram(table, "scores", title="Presence score")})
        wandb.finish()
