__all__ = ['get_vislogger']

from .space_vis import SpaceVis
def get_vislogger(cfg):
    return SpaceVis(cfg.train.do_visualize_categories)
    return None
