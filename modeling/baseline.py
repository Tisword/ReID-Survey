# encoding: utf-8

import torch
from torch import nn
import collections
from utils.Bilinearpooling import CompactBilinearPooling
from .backbones.resnet import ResNet, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.resnet_nl_rgbd import ResNetNL_RGBD
from .backbones.resnet_nl import ResNetNL
from .backbones.convnext import convnext_base
from .backbones.resnet_nl_dare import dare_resnet50
from .layer import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet, CenterLoss, GeneralizedMeanPooling, GeneralizedMeanPoolingP
from .layer import CBAMBlock_RGBD


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class visible_module(nn.Module):
    def __init__(self, model_path):
        super(visible_module, self).__init__()

        model_v = ResNet(last_stride=1,
                           block=Bottleneck,
                           layers=[3, 4, 6, 3])
        # avg pooling to global pooling
        self.visible = model_v
        self.visible.load_param(model_path)  # 这里加载预训练模型self.depth.load_param(model_path)  # 这里加载预训练模型

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        # x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class depth_module(nn.Module):
    def __init__(self, model_path):
        super(depth_module, self).__init__()

        model_t =ResNet(last_stride=1,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])

        # avg pooling to global pooling
        self.depth = model_t
        self.depth.load_param(model_path)  # 这里加载预训练模型

    def forward(self, x): #######是取resnet50的前面的一部分？
        x = self.depth.conv1(x)
        x = self.depth.bn1(x)
        # x = self.depth.relu(x)
        x = self.depth.maxpool(x)
        return x
#没有开头的那一层的Resnet
class base_resnet_nl(nn.Module):
    def __init__(self, last_stride,model_path):
        super(base_resnet_nl, self).__init__()

        self.base = ResNetNL(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3],
                               non_layers=[0, 2, 3, 0])
        self.base.load_param(model_path)  # 这里加载预训练模型
        print('Loading pretrained ImageNet model......')

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x
#尝试ConvNext的后面部分
class base_convnext(nn.Module):

    def __init__(self,a) -> None:
        super(base_convnext,self).__init__()
        #这里直接会加载
        self.base=convnext_base(pretrained=True,in_22k=False)
        print('Loading pretrained Convext model......')
    def forward(self,x):
        x=self.base.stages[0](x)
        for i in range(1,4):
            x = self.base.downsample_layers[i](x)
            x = self.base.stages[i](x)
        return x
#使用ConvNext第一个降采样层进行融合
class depth_module_v2(nn.Module):

    def __init__(self,a) -> None:
        super(depth_module_v2,self).__init__()
        self.base=convnext_base(pretrained=True)
        print('Loading pretrained Convext model depth......')
    def forward(self,x):
        x=self.base.downsample_layers[0](x)
        return x
class visible_module_v2(nn.Module):

    def __init__(self,a) -> None:
        super(visible_module_v2,self).__init__()
        self.base=convnext_base(pretrained=True)
        print('Loading pretrained Convext model visible......')
    def forward(self,x):
        x=self.base.downsample_layers[0](x)
        return x
class Baseline(nn.Module):
    in_planes = 2048
    # in_planes = 1024 #convnet使用
    # in_planes = 128 #dear_net用
    def __init__(self, num_classes, last_stride, model_path, model_name, gem_pool, pretrain_choice):
        super(Baseline, self).__init__()
        self.global_pooling_size=(8,4)
        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50_nl':
            self.base = ResNetNL(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3],
                               non_layers=[0, 2, 3, 0])
        elif model_name == 'resnet50_nl_mutil':
            self.base=base_resnet_nl(last_stride,model_path)
            self.visible_module=visible_module(model_path)
            self.depth_module=depth_module(model_path)
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)
        elif model_name =='convnext_mutil':
            self.base=base_convnext(1)
            self.visible_module = visible_module_v2(1)
            self.depth_module = depth_module_v2(1)
        elif model_name=='resnet_nl_mutil_twoline':
            self.base1=ResNetNL(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3],
                               non_layers=[0, 2, 3, 0])
            self.base2=ResNetNL(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3],
                               non_layers=[0, 2, 3, 0])
            # self.fusion = CompactBilinearPooling(self.in_planes,self.in_planes,self.in_planes, sum_pool=False)
            # self.fusion=nn.Linear(2*self.in_planes, self.in_planes) #Linear的效果不错
            # self.fusion=nn.AdaptiveAvgPool1d(1) #自适应池化，实际上没有过多的参数
            self.fusion=nn.Conv2d(2, 1, kernel_size=3, padding=1)
            self.CBAM3=CBAMBlock_RGBD(channel=1024,kernel_size=(16,8))
            self.CBAM4 = CBAMBlock_RGBD(channel=2048, kernel_size=(16, 8))
            self.base1.load_param(model_path)
            self.base2.load_param(model_path)
        elif model_name.startswith('resent_nl_dear_twoline'):
            #如果需要计算联合损失，则需要一起完成
            if model_name.endswith('sumloss'):
                self.base1=dare_resnet50(pretrained=True, Issumloss=True)
                self.base2=dare_resnet50(pretrained=True, Issumloss=True)
                self.layer3_fc = nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                )
                self.layer4_fc = nn.Sequential(
                    nn.Linear(4096, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(inplace=True),
                )
            else:
                self.base1 = dare_resnet50(pretrained=True, Issumloss=False)
                self.base2 = dare_resnet50(pretrained=True, Issumloss=False)
            self.fusion=nn.Conv2d(2, 1, kernel_size=3, padding=1)
        elif  model_name =='resent_nl_oneline_rgbd':
            #单独处理rgbd信息
            self.base=ResNetNL_RGBD(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3],
                               non_layers=[0, 2, 3, 0])
            self.base.conv1.apply(weights_init_kaiming)
        elif model_name=='resent_nl_twoline_rgbd_rgb':
            self.base1 = ResNetNL_RGBD(last_stride=last_stride,
                                      block=Bottleneck,
                                      layers=[3, 4, 6, 3],
                                      non_layers=[0, 2, 3, 0])
            self.base2=ResNetNL(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3],
                               non_layers=[0, 2, 3, 0])
            self.fusion = nn.Conv2d(2, 1, kernel_size=3, padding=1)
            self.base1.load_param(model_path)
            self.base2.load_param(model_path)
            self.base1.conv1.apply(weights_init_kaiming)

        # if pretrain_choice == 'imagenet':
        #     if model_name not in ['resnet50_nl_mutil','convnext_mutil','resnet_nl_mutil_twoline','resent_nl_dear_twoline','resent_nl_dear_twoline_sumloss']:
        #         # 如果不是多路的话就在这里加载
        #         self.base.load_param(model_path)#这里加载预训练模型
        #         print('Loading Nomal pretrained ImageNet model......')

        self.model_name=model_name
        self.num_classes = num_classes

        if gem_pool == 'on':
            print("Generalized Mean Pooling")
            self.global_pool = GeneralizedMeanPoolingP()
        else:
            print("Global Adaptive Pooling")
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    # def forward(self, x):
    #     x = self.base(x)
    #
    #     global_feat = self.global_pool(x)  # (b, 2048, 1, 1)
    #     global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
    #
    #     feat = self.bottleneck(global_feat)  # normalize for angular softmax
    #
    #     if not self.training:
    #         return feat
    #
    #     cls_score = self.classifier(feat)
    #     return cls_score, global_feat
    def forward_mutil(self, x1,x2):#尝试使用convenxt替换主干网络，可惜效果不好
        x1 = self.visible_module(x1)
        x2 = self.depth_module(x2)
        x=[]
        for i in range(0, len(x1)):
            xi = torch.einsum("abc,abc->abc", [x1[i], x2[i]])
            x.append(xi)

        x = torch.stack(x)
        x = self.base(x)
        if self.model_name == 'convnext_mutil':
            global_feat=self.base.base.norm(x.mean([-2, -1]))
            feat=global_feat
        else:
            global_feat = self.global_pool(x)  # (b, 2048, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            feat = self.bottleneck(global_feat)  # normalize for angular softmax


        if not self.training:
            return feat

        cls_score = self.classifier(feat)
        return cls_score, global_feat

#用于返回64*2048的向量
    def f(self,x, option=0):
        if option == 0:
            x = self.base1(x)
        else:
            x = self.base2(x)
        global_feat = self.global_pool(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        return global_feat

#  尝试使用自适应pooling的方式去融合，准确率有所提高(目前最高)，但是太久了
    def forward_twoline_pooling(self,x1,x2):
        #通过两条resnet_nl 再使用cat的方式连接
        feat1,feat2=self.f(x1,0),self.f(x2,1)
        # feat=torch.cat([feat1,feat2],dim=1)
        # global_feat=self.fusion(feat)
        # x=self.fusion(x1,x2)

        # x=x.permute(0, 3, 1, 2)
        # global_feat = self.global_pool(x)  # (b, 2048, 1, 1)
        # global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        feat=torch.stack((feat1,feat2),dim=0)
        feat=feat.permute(1,2,0)
        global_feat=torch.squeeze(self.fusion(feat),dim=2)
        return global_feat
        # feat = self.bottleneck(global_feat)  # normalize for angular softmax
        # if not self.training:
        #     return feat
        # cls_score = self.classifier(feat)
        # return cls_score, global_feat


    #尝试使用cmp的方式融合最后的两个向量，但是计算出的loss为nan，无法使用
    def forward_twoline_cmp(self,x1,x2):
        x1=self.base1(x1)
        x2=self.base2(x2)
        x=self.fusion(x1,x2)
        x=x.permute(0, 3, 1, 2)
        global_feat = self.global_pool(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        # feat = self.bottleneck(global_feat)  # normalize for angular softmax
        # if not self.training:
        #     return feat
        # cls_score = self.classifier(feat)
        # return cls_score, global_feat
    def forward_twoline_cat_linear(self,x1,x2):
        feat1, feat2 = self.f(x1, 0), self.f(x2, 1)
        feat=torch.cat([feat1,feat2],dim=1)
        global_feat=self.fusion(feat)
        return global_feat
    #尝试使用卷积
    def forward_twoline_conv(self,x1,x2):
        feat1,feat2=self.f(x1,0),self.f(x2,1)
        feat = torch.stack((feat1, feat2), dim=0)
        feat=torch.unsqueeze(feat,dim=0)
        global_feat=torch.squeeze(self.fusion(feat))
        return global_feat

    def forward_twoline_dear(self,x1,x2):
        feat1, feat2 = self.base1(x1), self.base2(x2)
        feat = torch.stack((feat1, feat2), dim=0)
        feat = torch.unsqueeze(feat, dim=0)
        global_feat = torch.squeeze(self.fusion(feat))
        return global_feat
    def forward_twoline_dear_sumloss(self,x1,x2):
        #先拿到原网络的每个结构输出的原本的向量
        #256,512,1024,2048
        a1,b1,c1,d1=self.base1.forward_agw(x1)
        a2,b2,c2,d2=self.base2.forward_agw(x2)
        c=torch.cat([c1,c2],dim=1)
        c = self.global_pool(c)
        c = c.view(c.shape[0], -1)
        featc=self.layer3_fc(c)

        d =torch.cat([d1,d2],dim=1)
        d=self.global_pool(d)
        d=d.view(d.shape[0],-1)
        d=self.layer4_fc(d)
        featd=self.bottleneck(d)
        if not self.training:
            return d

        cls_score = self.classifier(featd)

        return cls_score,(featc,featd)

        # return cls_score,feats
    def forward_twoline_CBAMblock(self, x1, x2):
        #使用第三条路构建中间那个分支
        x1=self.base1.forward0(x1)
        x1 = self.base1.forward1(x1)
        x1 = self.base1.forward2(x1)
        x1 = self.base1.forward3(x1)

        x2 = self.base1.forward0(x2)
        x2 = self.base1.forward1(x2)
        x2 = self.base1.forward2(x2)
        x2 = self.base1.forward3(x2)

        x1,x2=self.CBAM3(x1,x2)


        feat1=self.base1.forward4(x1)
        feat2=self.base2.forward4(x2)

        feat1 = self.global_pool(feat1)  # (b, 2048, 1, 1)
        feat1 = feat1.view(feat1.shape[0], -1)  # flatten to (bs, 2048)
        feat2 = self.global_pool(feat2)  # (b, 2048, 1, 1)
        feat2 = feat2.view(feat2.shape[0], -1)  # flatten to (bs, 2048)


        feat = torch.stack((feat1, feat2), dim=0)
        feat = torch.unsqueeze(feat, dim=0)
        global_feat = torch.squeeze(self.fusion(feat))
        return global_feat

    def forward(self,x1,x2):
        global_feat= self.forward_twoline_conv(x1, x2)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        if not self.training:
            return feat
        cls_score = self.classifier(feat)
        return cls_score, global_feat
        # return self.forward_twoline_dear_sumloss(x1,x2)
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if not isinstance(param_dict, collections.OrderedDict):
            param_dict = param_dict.state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def get_optimizer(self, cfg, criterion):
        optimizer = {}
        params = []
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        if cfg.MODEL.CENTER_LOSS == 'on':
            optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LR)
        return optimizer

    def get_creterion(self, cfg, num_classes):
        criterion = {}
        criterion['xent'] = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo

        print("Weighted Regularized Triplet:", cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET)
        if cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET == 'on':
            criterion['triplet'] = WeightedRegularizedTriplet()
        else:
            criterion['triplet'] = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

        if cfg.MODEL.CENTER_LOSS == 'on':
            criterion['center'] = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM,
                                             use_gpu=True)

        def criterion_total(score, feat, target):
            loss = criterion['xent'](score, target) + criterion['triplet'](feat, target)[0]
            if cfg.MODEL.CENTER_LOSS == 'on':
                loss = loss + cfg.SOLVER.CENTER_LOSS_WEIGHT * criterion['center'](feat, target)
            return loss

        criterion['total'] = criterion_total

        return criterion

    #cbp