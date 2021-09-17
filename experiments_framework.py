######
# dataset, dataloader, data aug
    # name, batch size
# model, load checkpoint
    # model parameters, size
# trainer
    # optim, scheduler (warmup)
# evaluation
# entry to distributed training
# logger, SummaryWriter, checkpoint
######

import argparse
import logging
import os
import torch
import torch.nn as nn
import datetime
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from models.copied_networks.cct import CCT
from tensorboardX import SummaryWriter

def set_logger():
    logging.getLogger().setLevel(logging.INFO)

def set_parser():
    parser = argparse.ArgumentParser(description='PyTorch framework for NNs')
    parser.add_argument("--randomseed", type=int, default=9999)
    parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10","cifar100"])
    parser.add_argument("--datafolder", type=str, default='./data/dataset/cifar100')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_aug", type=bool, default=True)
    args = parser.parse_args()
    return args

def set_device(args):
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu': args.num_gpus = 0
    args.distributed = args.num_gpus > 1
    args.local_rank = -1
    if args.distributed:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return args, device

def set_SummaryWriter(args):
    log_root = './tblogs/'
    log_name = ''
    date = str(datetime.datetime.now())
    log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    log_dir = os.path.join(log_root, log_name, log_base)
    if not args.distributed or args.local_rank == 1:
        writer = SummaryWriter(log_dir)
    return writer, log_dir

def data_normalize_augment(args):
    DATASETS = {
        'cifar10': {
            'num_classes': 10,
            'img_size': 32,
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2470, 0.2435, 0.2616]
        },
        'cifar100': {
            'num_classes': 100,
            'img_size': 32,
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761]
        }
    }
    args.img_size = DATASETS[args.dataset]['img_size']
    args.num_classes = DATASETS[args.dataset]['num_classes']
    args.img_mean, args.img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']
    
    normalize = [transforms.Normalize(mean=args.img_mean, std=args.img_std)]
    augmentations = []
    
    if args.data_aug:
        from utils.autoaug import CIFAR10Policy
        augmentations += [
            CIFAR10Policy()
        ]
    augmentations += [
        transforms.RandomCrop(args.img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
        ]

    augmentations = transforms.Compose(augmentations)
    return args, augmentations

def get_trainloader(args, augmentations):
    train_dataset = datasets.__dict__[args.dataset.upper()](root = args.datafolder, train = True, download = True, transform = augmentations)
    
    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_dataset)
        trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    else:
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
    return trainloader

def get_testloader(args):
    normalize = [transforms.Normalize(mean=args.img_mean, std=args.img_std)]
    val_dataset = datasets.__dict__[args.dataset.upper()](
        root=args.datafolder, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            *normalize,
        ]))
    testloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)
    return testloader
    

if __name__ == '__main__':
    
    ## set argument parser
    args = set_parser()
    print(args.dataset)
    
    ## set device
    args, device = set_device(args)
    
    ## set logger
    set_logger()
    log = not args.distributed or args.local_rank == 1
    
    logging.info(f"[Device] device: {device}, num_gpus: {args.num_gpus}, distributed: {args.distributed}") # log for all devices
    
    ## set SummaryWriter
    writer, log_dir = set_SummaryWriter(args)
    if log: logging.info(f"[SummaryWriter] Directory: {log_dir}")
    
    ## set dateset and loader
    args, augmentations = data_normalize_augment(args)
    if log: logging.info(f"[Data] Dataset: {args.dataset}, img_size: {args.img_size}, num_class: {args.num_classes}")
    
    trainloader = get_trainloader(args, augmentations)
    testloader = get_testloader(args)
    if log: logging.info(f"[DataLoader] Batch size: {args.batch_size}, augmentation: {args.data_aug}, num_workers: {args.num_workers}")
    
    ## set model
    # model = CCT(img_size=args.img_size, kernel_size=cfg.model_patch_size, n_input_channels=datashape[1], num_classes=nclasses, embeding_dim=cfg.model_embed_dim, num_layers=cfg.model_num_layers,num_heads=cfg.model_num_heads, mlp_ratio=cfg.model_mlp_ratio, n_conv_layers=cfg.model_conv_layer, drop_rate=cfg.train_dropout_rate, attn_drop_rate=cfg.train_attn_dropout_rate, drop_path_rate=cfg.train_drop_path_rate, layerscale = cfg.model_layerscale, positional_embedding='learnable')
    
    ## set optimizer, scheduler
    
    
    

    
    
    
    





