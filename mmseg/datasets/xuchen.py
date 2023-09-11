from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp 
classes =("1","2","3","4","5","6","7","8","9","10","11","12")
image_palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                 [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0]]
@DATASETS.register_module()
class XuchenDataset(BaseSegDataset):
    METAINFO = dict(
        classes=classes,
        palette=image_palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',  **kwargs)
        # assert osp.exists(self.img_dir) 