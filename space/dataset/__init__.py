from .atari import Atari
from .hdf5 import Hdf5Dataset
from .obj3d import Obj3D
from torch.utils.data import DataLoader


__all__ = ['get_dataset', 'get_dataloader']

from .procgen import CustomDataset


def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    if cfg.dataset == 'ATARI':
        mode = 'validation' if mode == 'val' else mode
        return Atari(cfg.dataset_roots.ATARI, mode, gamelist=cfg.gamelist)
    elif cfg.dataset == 'OBJ3D_SMALL':
        return Obj3D(cfg.dataset_roots.OBJ3D_SMALL, mode)
    elif cfg.dataset == 'OBJ3D_LARGE':
        return Obj3D(cfg.dataset_roots.OBJ3D_LARGE, mode)
    elif 'hdf5' in cfg.dataset:
        return Hdf5Dataset(cfg.dataset_roots[cfg.dataset], mode=mode, image_size=cfg.arch.img_shape, to_tensor=cfg.get('to_tensor', True))
    elif 'custom' in cfg.dataset:
        return CustomDataset(cfg.dataset_roots[cfg.dataset], image_size=cfg.arch.img_shape, to_tensor=cfg.get('to_tensor', True), gamelist=cfg.gamelist)

def get_dataloader(cfg, mode):
    assert mode in ['train', 'val', 'test']
    
    batch_size = getattr(cfg, mode).batch_size
    shuffle = True
    num_workers = getattr(cfg, mode).num_workers
    
    dataset = get_dataset(cfg, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    
    return dataloader
    
