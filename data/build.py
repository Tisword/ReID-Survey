# encoding: utf-8

import torch
from torch.utils.data import DataLoader

from .datasets import init_dataset, ImageDataset,ImageDataSet_Mutil,ImageDataSet_Mutil_Own
from .triplet_sampler import RandomIdentitySampler
from .transforms import build_transforms,build_transforms_mutil
from .transformRGBD import TransformRGBD,TransformRGBD_Eval


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def train_collate_fn_mutil(batch):

    imgs1, imgs2, pids,_,= zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs1, dim=0), torch.stack(imgs2, dim=0), pids
def val_collate_fn_mutil(batch):
    imgs1, imgs2, pids, camids, = zip(*batch)
    return torch.stack(imgs1, dim=0), torch.stack(imgs2, dim=0), pids,camids



def make_data_loader(cfg):
    transforms = build_transforms(cfg)
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_set = ImageDataset(dataset.train, transforms['train'])
    data_loader={}
    #一般都会使用，用来指定一个批次的人的类别和一个类别的图片数
    if cfg.DATALOADER.PK_SAMPLER == 'on':
        data_loader['train'] = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    else:
        #shuffle为True的话，每次出来的batch都是随机且打乱的，否则按原来的数据顺序建立批次
        data_loader['train'] = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )

    if cfg.TEST.PARTIAL_REID == 'off':
        eval_set = ImageDataset(dataset.query + dataset.gallery, transforms['eval'])
        data_loader['eval'] = DataLoader(
            eval_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
    else:
        dataset_reid = init_dataset('partial_reid', root=cfg.DATASETS.ROOT_DIR)
        dataset_ilids = init_dataset('partial_ilids', root=cfg.DATASETS.ROOT_DIR)
        eval_set_reid = ImageDataset(dataset_reid.query + dataset_reid.gallery, transforms['eval'])
        data_loader['eval_reid'] = DataLoader(
            eval_set_reid, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        eval_set_ilids = ImageDataset(dataset_ilids.query + dataset_ilids.gallery, transforms['eval'])
        data_loader['eval_ilids'] = DataLoader(
            eval_set_ilids, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
    return data_loader, len(dataset.query), num_classes

#用于制作彩色图和深度图的数据集
def make_dataloader_mutil(cfg):
    if cfg.MODEL.USEOWN_TRANSFORM == 'off':
        transforms=build_transforms_mutil(cfg)
    else:
        transforms=TransformRGBD(cfg=cfg)
        transforms_eval=TransformRGBD_Eval(cfg=cfg)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = init_dataset(cfg.DATASETS.NAMES,root=cfg.DATASETS.ROOT_DIR)
    if cfg.MODEL.USEOWN_TRANSFORM=='off':
        train_set = ImageDataSet_Mutil(dataset.train,transformRGB=transforms['train_rgb'],transformDepth=transforms['train_depth'])
    else:
        train_set = ImageDataSet_Mutil_Own(dataset.train, transform=transforms)
    num_classes = dataset.num_train_pids
    data_loader = {}
    if cfg.DATALOADER.PK_SAMPLER == 'on':
        data_loader['train'] = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn_mutil
        )
    else:
        #shuffle为True的话，每次出来的batch都是随机且打乱的，否则按原来的数据顺序建立批次
        data_loader['train'] = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn_mutil
        )
    if cfg.MODEL.USEOWN_TRANSFORM == 'off':
        eval_set = ImageDataSet_Mutil(dataset.query + dataset.gallery
                                , transformRGB=transforms['eval_rgb'],transformDepth=transforms['eval_depth'])
    else:
        eval_set = ImageDataSet_Mutil_Own(dataset.query + dataset.gallery
                                  , transform=transforms_eval)
    data_loader['eval'] = DataLoader(
        eval_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn_mutil
    )
    return data_loader,len(dataset.query),num_classes

