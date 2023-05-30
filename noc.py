#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/30 23:19
# @File     : noc.py
# @Project  : lab
import argparse

import yaml

import utils


def main(config):
    device = utils.reproduce(args.seed)


if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(description="Image Captioning of Novel Objects by PyTorch")
    parser.add_argument('--seed', '-s', default=config['seed'], type=int, help="Set random seed")
    parser.add_argument('--batch-size', '-b', type=int, default=config['batch_size'], help='Training batch size')
    parser.add_argument('--lr', default=config['lr'], type=float, help="Learning rate")
    parser.add_argument('--epoch', '-e', default=config['epoch'], type=int, help="Max training epochs")
    args = parser.parse_args()
    main(config)
