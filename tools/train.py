# encoding: utf-8

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import r1_mAP_mINP
from tools.test import create_supervised_evaluator,create_supervised_evaluator_mutil

global ITER
ITER = 0


def create_supervised_trainer(model, optimizer, criterion, cetner_loss_weight=0.0, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (dict - class:`torch.optim.Optimizer`): the optimizer to use
        criterion (dict - class:loss function): the loss function to use
        cetner_loss_weight (float, optional): the weight for cetner_loss_weight
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """

    def _update(engine, batch):
        model.train()
        optimizer['model'].zero_grad()

        if 'center' in optimizer.keys():
            optimizer['center'].zero_grad()

        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = criterion['total'](score, feat, target)
        loss.backward()
        optimizer['model'].step()

        if 'center' in optimizer.keys():
            for param in criterion['center'].parameters():
                param.grad.data *= (1. / cetner_loss_weight)
            optimizer['center'].step()

        # compute acc  这里完全用类别去判断准确率
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)

def create_supervised_trainer_mutil(cfg,model, optimizer, criterion, cetner_loss_weight=0.0, device=None):

    def _update(engine, batch):
        model.train()
        optimizer['model'].zero_grad()

        if 'center' in optimizer.keys():
            optimizer['center'].zero_grad()

        img_rgb,img_depth,target = batch
        img_rgb = img_rgb.to(device) if torch.cuda.device_count() >= 1 else img_rgb
        img_depth=img_depth.to(device) if torch.cuda.device_count() >=1 else img_depth
        if cfg.MODEL.NAME=='resent_nl_oneline_rgbd':
            img=torch.cat([img_rgb,img_depth],dim=1)
            score, feat = model(img)
        elif cfg.MODEL.NAME=='resent_nl_twoline_rgbd_rgb':
            img_rgbd = torch.cat([img_rgb, img_depth], dim=1)
            score, feat = model(img_rgbd, img_rgb)
        else:#其余的时候作为三通道输入
            img_depth = torch.cat([img_depth, img_depth, img_depth], dim=1)
            score, feat = model(img_rgb, img_depth)
        target = target.to(device) if torch.cuda.device_count() >= 1 else target


        #这里判断要不要使用联合损失
        if type(feat) is list or type(feat) is tuple:
            loss_list = []
            for i in range (0,len(feat)-1):
                loss_list.append(
                    criterion['triplet'](feat[i], target)[0]
                )
            loss_list.append(criterion['total'](score, feat[-1], target))
            #最后一层算分类损失和三重损失
            loss = sum(loss_list)
            # compute acc  这里完全用类别去判断准确率,这里应该使用最后一个类别输出的分数
            acc = (score.max(1)[1] == target).float().mean()
        else:
            loss = criterion['total'](score, feat, target)
            # compute acc  这里完全用类别去判断准确率
            acc = (score.max(1)[1] == target).float().mean()

        loss.backward()
        optimizer['model'].step()

        if 'center' in optimizer.keys():
            for param in criterion['center'].parameters():
                param.grad.data *= (1. / cetner_loss_weight)
            optimizer['center'].step()

        # compute acc  这里完全用类别去判断准确率
        # acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)

def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        criterion,
        num_query,
        start_epoch,
        Ismutil=False
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline")
    logger.info("Start training")
    if not Ismutil:
        trainer = create_supervised_trainer(model, optimizer, criterion, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    else:
        trainer = create_supervised_trainer_mutil(cfg,model, optimizer, criterion, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    if cfg.TEST.PARTIAL_REID == 'off':
        if not Ismutil:
            evaluator = create_supervised_evaluator(model, metrics={
                'r1_mAP_mINP': r1_mAP_mINP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
        else:
            evaluator = create_supervised_evaluator_mutil(cfg,model, metrics={
                'r1_mAP_mINP': r1_mAP_mINP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    else:
        evaluator_reid = create_supervised_evaluator(model,
                                                     metrics={'r1_mAP_mINP': r1_mAP_mINP(300, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                     device=device)
        evaluator_ilids = create_supervised_evaluator(model,
                                                      metrics={'r1_mAP_mINP': r1_mAP_mINP(119, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                      device=device)
    #这里定义了文件的保存格式和保存内容
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer['model'],
                                                                     'center_param': criterion['center'],
                                                                     'optimizer_center': optimizer['center']})
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(data_loader['train']),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(data_loader['train']) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            data_loader['train'].batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            if cfg.TEST.PARTIAL_REID == 'off':
                evaluator.run(data_loader['eval'])
                cmc, mAP, mINP = evaluator.state.metrics['r1_mAP_mINP']
                logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
                logger.info("mINP: {:.1%}".format(mINP))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            else:
                evaluator_reid.run(data_loader['eval_reid'])
                cmc, mAP, mINP = evaluator_reid.state.metrics['r1_mAP_mINP']
                logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
                logger.info("mINP: {:.1%}".format(mINP))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 3, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


                evaluator_ilids.run(data_loader['eval_ilids'])
                cmc, mAP, mINP = evaluator_ilids.state.metrics['r1_mAP_mINP']
                logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
                logger.info("mINP: {:.1%}".format(mINP))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 3, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(data_loader['train'], max_epochs=epochs)
