# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:52:50 2020

@author: dubs
"""
from dataset.datasets import get_dataloader
from common import config
#from model import get_autoencoder
from functional.utils import cycle
#from agent import get_training_agent
from functional.visualization import visulize_motion_in_training
import torch
import os
from collections import OrderedDict
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse

torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, choices=['skeleton', 'view', 'full'], required=True,
                        help='which structure to use')
    # parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    parser.add_argument('--disable_triplet', action='store_true', default=False, help="disable triplet loss")
    parser.add_argument('--use_footvel_loss', action='store_true', default=False, help="use use footvel loss")
    parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
    args = parser.parse_args()

    config.initialize(args)
    
    train_loader = get_dataloader('train', config, config.batch_size, config.num_workers)
    mean_pose, std_pose = train_loader.dataset.mean_pose, train_loader.dataset.std_pose
    val_loader = get_dataloader('test', config, config.batch_size, config.num_workers)
    val_loader = cycle(val_loader)
    
    return train_loader,val_loader
