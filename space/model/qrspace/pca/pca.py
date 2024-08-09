import argparse

import torch
from sklearn.cluster import KMeans
import joblib
import wandb
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import numpy as np
from scipy import sparse
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os


def pred_cat(x, pca, kmeans):
    z = pca.transform(x.reshape((x.shape[0], -1)))
    cat = kmeans.predict(z)
    return cat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, default=64)
    parser.add_argument('--sample_size', type=int, default=2048)
    parser.add_argument('--num_categories', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_pca_components', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_size', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='cat_pred.model')
    parser.add_argument('--wandb_project', type=str, required=False)
    parser.add_argument('--wandb_group', type=str, default='Test group')
    parser.add_argument('--wandb_run_name', type=str, default='run-0')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    N = args.num_examples
    M = args.sample_size
    C = args.num_categories

    colors = [[255, 0, 0], [255, 255, 0], [255, 153, 18], [255, 127, 80], [255, 192, 203], [255, 0, 255], [0, 255, 0],
              [0, 255, 255], [8, 46, 84], [0, 199, 140], [0, 0, 255], [160, 32, 240], [218, 112, 214]]
    colors = torch.Tensor(colors) / 255.
    logdir = args.log_dir
    data = np.load(args.data_path, mmap_mode='r')
    if args.data_size > 0:
        data = data[:args.data_size]

    X = data.reshape((data.shape[0], -1))
    transformer = IncrementalPCA(n_components=args.num_pca_components, batch_size=args.batch_size)
    X = sparse.csr_matrix(X)
    X_transformed = transformer.fit_transform(X)
    kmeans = KMeans(init="k-means++", n_clusters=C, max_iter=1000)
    kmeans = kmeans.fit(X_transformed)

    joblib.dump(dict(pca=transformer, kmeans=kmeans), args.save_path)
    labels = kmeans.predict(X_transformed[:M])

    os.makedirs(logdir, exist_ok=True)
    if args.wandb_project is not None:
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name,
            resume='never',
            sync_tensorboard=True,
        )

    writer = SummaryWriter(log_dir=args.log_dir, flush_secs=10)

    writer.add_embedding(torch.as_tensor(X_transformed[:M]), metadata=labels, label_img=torch.as_tensor(data[:M]),
                         global_step=0)

    imgs = data[np.random.choice(len(data), N)]
    cat = pred_cat(imgs, transformer, kmeans)
    imgs[:, :, :3] = colors[cat].reshape(-1, 3, 1, 1)
    grid = make_grid(torch.as_tensor(imgs), 8)
    writer.add_image('CatInfo', grid, 0)

    for i in range(C):
        cat_imgs = data[:M][labels == i]
        cat_imgs = cat_imgs[:min(N, len(cat_imgs))]
        grid = make_grid(torch.as_tensor(cat_imgs), 8)
        writer.add_image(f'CatImg/Cat{i}', grid, global_step=0)

    wandb.finish()
