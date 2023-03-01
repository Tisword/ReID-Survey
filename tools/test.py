# encoding: utf-8
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine

from utils.reid_metric import r1_mAP_mINP, r1_mAP_mINP_reranking


def create_supervised_evaluator(model, metrics, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to evaluate
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def create_supervised_evaluator_mutil(cfg,model, metrics, device=None):
    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            img_rgb, img_depth, pids, camids = batch
            img_rgb = img_rgb.to(device) if torch.cuda.device_count() >= 1 else img_rgb
            img_depth = img_depth.to(device) if torch.cuda.device_count() >= 1 else img_depth
            if cfg.MODEL.NAME == 'resent_nl_oneline_rgbd':
                img = torch.cat([img_rgb, img_depth], dim=1)
                feat = model(img)
            elif cfg.MODEL.NAME == 'resent_nl_twoline_rgbd_rgb':
                img_rgbd = torch.cat([img_rgb, img_depth], dim=1)
                feat = model(img_rgbd, img_rgb)
            else:  # 其余的时候作为三通道输入
                img_depth = torch.cat([img_depth, img_depth, img_depth], dim=1)
                feat = model(img_rgb, img_depth)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def do_test(
        cfg,
        model,
        data_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'off':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP_mINP': r1_mAP_mINP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'on':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP_mINP': r1_mAP_mINP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for on or off, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(data_loader['eval'])
    cmc, mAP, mINP = evaluator.state.metrics['r1_mAP_mINP']
    logger.info('Validation Results')
    logger.info("mINP: {:.1%}".format(mINP))
    logger.info("mAP: {:.1%}".format(mAP))
    if cfg.TEST.PARTIAL_REID == 'off':
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    else:
        for r in [1, 3, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
