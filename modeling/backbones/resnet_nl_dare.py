# encoding: utf-8
#resnet_nl 根据论文Resource Aware Person Re-identiﬁcation across Multiple Resolutions得出的网络
#注意这里要修改loss

import math

import torch
from torch import nn
from modeling.layer.non_local import Non_local

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch




model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, fc_layer1=1024, fc_layer2=128, global_pooling_size=(8,4), drop_rate=0. ,Issumloss=False ,non_layers=[0, 2, 3, 0]):
        self.Issumloss = Issumloss
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #NoLocal部分
        self.NL_1 = nn.ModuleList(
            [Non_local(256) for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512) for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048) for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])


        self.avgpool1 = nn.AvgPool2d([x*8 for x in global_pooling_size])
        self.avgpool2 = nn.AvgPool2d([x*4 for x in global_pooling_size])
        self.avgpool3 = nn.AvgPool2d([x*2 for x in global_pooling_size])
        self.avgpool4 = nn.AvgPool2d([x*1 for x in global_pooling_size])
        self.layer1_fc = nn.Sequential(
            nn.Linear(64 * block.expansion, fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer2_fc = nn.Sequential(
            nn.Linear(128 * block.expansion, fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer3_fc = nn.Sequential(
            nn.Linear(256 * block.expansion, fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )
        self.layer4_fc = nn.Sequential(
            nn.Linear(512 * block.expansion, fc_layer1),
            nn.BatchNorm1d(fc_layer1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_layer1, fc_layer2),
        )

        self.fusion_conv = nn.Conv1d(3,1,kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    #使用agw的forward方式
    def forward_agw(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)
        #Layer 1
        NL1_counter = 0
        if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        x1=x
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        x2 = x
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        x3 = x
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
        x4=x
        return x1,x2,x3,x4

    def forward(self, x):
        """x shape is (batch size, sequence, c, h, w)"""
        # x = x.view(-1, *x.size()[-3:])
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)


        # x1 = self.layer1(x)
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # x4 = self.layer4(x3)
        #Layer 1
        NL1_counter = 0
        if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        x1=x
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        x2 = x
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        x3 = x
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
        x4=x
        x1 = self.avgpool1(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.layer1_fc(x1)

        x2 = self.avgpool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.layer2_fc(x2)

        x3 = self.avgpool3(x3)
        x3 = x3.view(x3.size(0), -1)
        x3 = self.layer3_fc(x3)

        x4 = self.avgpool4(x4)
        x4 = x4.view(x4.size(0), -1)
        x4 = self.layer4_fc(x4)

        # x5 = torch.cat([x1.unsqueeze(dim=1),x2.unsqueeze(dim=1),x3.unsqueeze(dim=1),x4.unsqueeze(dim=1)],dim=1)
        x5 = torch.cat([ x2.unsqueeze(dim=1), x3.unsqueeze(dim=1), x4.unsqueeze(dim=1)], dim=1)
        x5 = self.fusion_conv(x5)
        x5 = x5.view(x5.size(0),-1)
        if self.training:
            if self.Issumloss:
                return (x1,x2,x3,x4,x5)
            else:
                return x5
        else:
                return x5





def dare_resnet50(pretrained=False, fc_layer1=1024, fc_layer2=2048, global_pooling_size=(8, 4), drop_rate=0., Issumloss=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], fc_layer1=fc_layer1, fc_layer2=fc_layer2,
                   global_pooling_size=global_pooling_size, drop_rate=drop_rate, Issumloss=Issumloss)
    if pretrained:
        model_dict = model.state_dict()
        params = model_zoo.load_url(model_urls['resnet50'])
        params = {k: v for k, v in params.items() if k in model_dict}
        model_dict.update(params)
        model.load_state_dict(model_dict)
    return model


#融合直接在一个网络中融合向量
class DareNet(nn.Module):

    def __init__(self) -> None:
        super(DareNet,self).__init__()
        fc_layer = 2048
        global_pooling_size = (8, 4)
        self.base1=dare_resnet50(pretrained=True)
        self.base2=dare_resnet50(pretrained=True)
        self.avgpool3 = nn.AvgPool2d([x * 2 for x in global_pooling_size])
        self.avgpool4 = nn.AvgPool2d([x * 1 for x in global_pooling_size])
        self.layer3_fc = nn.Sequential(
            nn.Linear(fc_layer,fc_layer),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
        )
        self.layer4_fc = nn.Sequential(
            nn.Linear(2*fc_layer, fc_layer),
            nn.BatchNorm1d(),
            nn.ReLU(inplace=True),
        )
    def forward(self,x1,x2):
        #先拿到原网络的每个结构输出的原本的向量
        #256,512,1024,2048
        a1,b1,c1,d1=self.base1.forward_agw(x1)
        a2,b2,c2,d2=self.base2.forward_agw(x2)
        #先尝试只融合最后两层，
        #先融合倒数第二层，直接用之前卷积的拼起
        c=torch.cat(c1,c2,dim=1)
        c=self.avgpool3(c)
        c=self.layer3_fc(c)


        d=torch.cat(d1,d2,dim=1)
        d=self.avgpool3(d)







