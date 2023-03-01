# encoding: utf-8


import os.path as osp
from PIL import Image
from torch.utils.data import Dataset

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_img_L(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process.
    用于读取深度图片
    """
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('L')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

#用于读取彩色图和深度图
class ImageDataSet_Mutil(Dataset):
    def __init__(self, dataset, transformRGB=None,transformDepth=None):
        self.dataset = dataset
        self.transform1 = transformRGB
        self.transform2= transformDepth

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path_color, img_path_depth, pid,camid= self.dataset[index]
        img_color = read_image(img_path_color)
        img_depth = read_img_L(img_path_depth)

        if self.transform1 is not None:
            img_color = self.transform1(img_color)

        if self.transform2 is not None:
            img_depth = self.transform2(img_depth)

        return img_color, img_depth, pid,camid
class ImageDataSet_Mutil_Own(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform=transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path_color, img_path_depth, pid,camid= self.dataset[index]
        img_color = read_image(img_path_color)
        img_depth = read_img_L(img_path_depth)

        if self.transform is not None:
            img_color,img_depth=self.transform(img_color,img_depth)

        return img_color, img_depth, pid,camid