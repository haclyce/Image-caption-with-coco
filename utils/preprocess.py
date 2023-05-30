#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/30 23:23
# @File     : preprocess.py
# @Project  : lab
import os
import random

import numpy as np
import torch


def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if not os.path.isdir('./log'):
        os.mkdir('./log')
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')
    return 'cuda' if torch.cuda.is_available() else 'cpu'
