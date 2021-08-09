# ResNet + Vision Transformer Block + Scale + Stochastic Depth + Kinetic Regularization
# Structure
#   - device: cpu or gpu, no distributed
#   - recording: results/
#   - data
#   - model
#   - train
#   - test

import os
import time
import datetime

import torch
import torch.nn as nn
from torch import optim

from data.loader.dataloaders_LAP import dataloaders

import argparse

def main(cfg):
    ### [device]
    print('-'*50)
    print('[Device] choosing devices ...')
    if cfg.device == "cuda":
        if torch.cuda.is_available():
            print('[Device] GPU: ', torch.cuda.get_device_name(0))
        else:
            print('[Device] no GPU available, use cpu')
            cfg.device = "cpu"
    else:
        print('[Device] use cpu')
    device = cfg.device
    
    ### [data]
    print('-'*50)
    print('[Dataset] preparing dataset: ' + cfg.dataset, '...')
    trainloader, valloader, testloader, datashape, nclasses, mean, std = dataloaders(cfg.dataset, cfg.train_batch_size, trainsize = 1, valsize = 0, testsize = 1)
    print('[Dataset] number of training images: ', len(trainloader.dataset), ' test images: ', len(testloader.dataset))
    print('[Dataset] data shape: ', datashape, ' num of classes: ', nclasses)
    print('[Dataset] batch_size: ', cfg.train_batch_size)

    ### [model]
    print('-'*50)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--status", type=str, default="train", choices=["train", "test"])
    
    parser.add_argument("--randomseed", type=int, default=int(str(time.time()).split(".")[-1][::-1]))
    
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist","cifar10","cifar100","tinyimagenet"])
    
    parser.add_argument("--model", type=str, default="cait", choices=["resnet","cait"])
    parser.add_argument("--model_patch_size", type=int, default=4)
    parser.add_argument("--model_embed_dim", type=int, default=128)
    parser.add_argument("--model_num_layers", type=int, default=32)
    parser.add_argument("--model_heads", type=int, default=16)
    
    
    parser.add_argument("--train_batch_size", type=int, default=40)
    parser.add_argument("--train_num_epochs", type=int, default=200)
    parser.add_argument("--train_dropout_rate", type=float, default=0.0)
    parser.add_argument("--train_drop_path_rate", type=float, default=0.2)
    parser.add_argument("--train_initial_scale", type=float, default=1e-4)
    
    parser.add_argument("--optim_loss", type=str, default="cross_entropy")
    parser.add_argument("--optim_alg", type=str, default="adam")
    parser.add_argument("--optim_lr", type=float, default=1e-3)
    
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="./results/")

    cfg = parser.parse_args()
    main(cfg)
