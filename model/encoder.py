#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/31 22:55
# @File     : encoder.py
# @Project  : lab
from torch import nn
from torchvision import models


class EncoderResNet(nn.Module):
    def __init__(self, embed_size, name='resnet50'):
        super(EncoderResNet, self).__init__()
        resnet = getattr(models, name)(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
