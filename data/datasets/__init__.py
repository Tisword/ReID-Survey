# encoding: utf-8

from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .tvpr2 import TVPR2
from .partial_ilids import PartialILIDS
from .partial_reid import PartialREID
from .dataset_loader import ImageDataset
from .dataset_loader import ImageDataSet_Mutil,ImageDataSet_Mutil_Own
from .dataset_loader import read_img_L,read_image
__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
    'tvpr2':TVPR2,
    #下面两个数据集是被遮挡的数据集，不用理会
    'partial_reid' : PartialREID,
    'partial_ilids' : PartialILIDS,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
