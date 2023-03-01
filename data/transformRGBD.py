from PIL import Image
from torchvision.transforms.functional import hflip
import torchvision.transforms.functional as F
import torchvision.transforms as T
import math
import random

#新的随机擦除
class NewRandomErasing(object):
    """ 这个类只用来得到 x1,y1,h,w
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                # if img.size()[0] == 3:
                #     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                #     img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                #     img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                # else:
                #     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                # return img
                return (x1,y1,h,w)

        return img

class Erasing(object):
    def __init__(self,parameter,mean=(0.4914, 0.4822, 0.4465)):
        self.param=parameter
        self.mean = mean
    def __call__(self, img):
        #说明是一个元组
        if type(self.param) is tuple:
                x1,y1,h,w=self.param[0],self.param[1],self.param[2],self.param[3]
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        else :
            return img

class TransformRGBD(object):
    def __init__(self,cfg):
        #开始定义各种参数
        self.cfg=cfg
        self.t1= T.Resize(cfg.INPUT.IMG_SIZE)
        self.t2=hflip
        self.t3=T.Pad(cfg.INPUT.PADDING)
        self.t4= T.RandomCrop(cfg.INPUT.IMG_SIZE)
        self.t5= T.ToTensor()
        self.t6_rgb= T.Normalize(mean=cfg.INPUT.RGB.PIXEL_MEAN, std=cfg.INPUT.RGB.PIXEL_STD)
        self.t6_depth= T.Normalize(mean=cfg.INPUT.DEPTH.PIXEL_MEAN, std=cfg.INPUT.DEPTH.PIXEL_STD)
        self.t7=NewRandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    def __call__(self, imgRGB,imgD):
        #一步步处理
        image1,image2=self.t1(imgRGB),self.t1(imgD)
        #是否水平翻转
        if random.random() < 0.5:
            image1 = self.t2(image1)
            image2 = self.t2(image2)
        #pad
        image1,image2=self.t3(image1),self.t3(image2)
        #随机裁剪
        i, j, h, w=self.t4.get_params(image1,self.cfg.INPUT.IMG_SIZE)
        image1=F.crop(image1,i,j,h,w)
        image2=F.crop(image2,i,j,h,w)
        #转为tensor
        img1,img2=self.t5(image1),self.t5(image2)
        #normalize
        img1,img2=self.t6_rgb(img1),self.t6_depth(img2)
        #随机擦除
        param=self.t7(img1)
        t8=Erasing(param)
        img1,img2=t8(img1),t8(img2)
        return img1,img2

class TransformRGBD_Eval(object):
    def __init__(self,cfg):
        #开始定义各种参数
        self.cfg=cfg
        self.t1= T.Resize(cfg.INPUT.IMG_SIZE)
        self.t2=hflip
        self.t3=T.Pad(cfg.INPUT.PADDING)
        self.t4= T.RandomCrop(cfg.INPUT.IMG_SIZE)
        self.t5= T.ToTensor()
        self.t6_rgb= T.Normalize(mean=cfg.INPUT.RGB.PIXEL_MEAN, std=cfg.INPUT.RGB.PIXEL_STD)
        self.t6_depth= T.Normalize(mean=cfg.INPUT.DEPTH.PIXEL_MEAN, std=cfg.INPUT.DEPTH.PIXEL_STD)
        self.t7=NewRandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    def __call__(self, imgRGB,imgD):
        #一步步处理
        image1,image2=self.t1(imgRGB),self.t1(imgD)
        #是否水平翻转
        # if random.random() < 0.5:
        #     image1 = self.t2(image1)
        #     image2 = self.t2(image2)
        # #pad
        # image1,image2=self.t3(image1),self.t3(image2)
        # #随机裁剪
        # i, j, h, w=self.t4.get_params(image1,self.cfg.INPUT.IMG_SIZE)
        # image1=F.crop(image1,i,j,h,w)
        # image2=F.crop(image2,i,j,h,w)
        #转为tensor
        img1,img2=self.t5(image1),self.t5(image2)
        #normalize
        img1,img2=self.t6_rgb(img1),self.t6_depth(img2)
        #随机擦除
        # param=self.t7(img1)
        # t8=Erasing(param)
        # img1,img2=t8(img1),t8(img2)
        return img1,img2