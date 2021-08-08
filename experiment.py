# ResNet + Vision Transformer Block + Scale + Stochastic Depth + Kinetic Regularization
# Structure
#   - device: cpu or gpu
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

import argparse

def main(cfg):
    print(cfg.status)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--status", type=str, default="train", choices=["train", "test"])
    
    parser.add_argument("--randomseed", type=int, default=int(str(time.time()).split(".")[-1][::-1]))
    
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist","cifar10"])
    
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
    
    parser.add_argument("optim_loss", type=str, default="cross_entropy")
    parser.add_argument("optim_alg", type=str, default="adam")
    parser.add_argument("optim_lr", type=float, default=1e-3)
    
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="./results/")

    cfg = parser.parse_args()
    main(cfg)