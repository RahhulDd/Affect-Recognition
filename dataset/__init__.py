# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:48:57 2020

@author: dubs
"""
from dataset.datasets import _DatasetForSkeleton, _DatasetForView, _DatasetForFull
from torch.utils.data import DataLoader
from dataset.base_dataset import get_meanpose
import numpy as np


def get_dataloader(phase, config, batch_size=64, num_workers=4):
    assert config.name is not None
    if config.name == 'skeleton':
        dataset = _DatasetForSkeleton(phase, config)
    elif config.name == 'view':
        dataset = _DatasetForView(phase, config)
    else:
        dataset = _DatasetForFull(phase, config)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, worker_init_fn=lambda _: np.random.seed())
    # if phase == 'Train':
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
    #                             num_workers=num_workers, worker_init_fn=lambda _: np.random.seed())
    # else:
    #     dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return dataloader
