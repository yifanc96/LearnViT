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

from models.copied_networks.ViT import VisionTransformer
from training.trainer_ViT_kinetic_hook_distributed import trainer
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler

import argparse

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(random_seed)
    # random.seed(random_seed)
    
def main(cfg):
    print('-'*50)
    print('[Randomseed]', cfg.randomseed)
    set_random_seeds(random_seed=cfg.randomseed)
    ### [device]
    print('-'*50)
    print('[Device] choosing devices ...')
    if cfg.device == "cuda":
        if torch.cuda.is_available():
            torch.distributed.init_process_group(backend="nccl")
            local_rank = cfg.local_rank
            if local_rank ==0: print('[Device] GPU: ', torch.cuda.get_device_name(0))
            nGPUs = torch.cuda.device_count()
            if local_rank ==0: print('[Device] number of GPUs: ', nGPUs)
        else:
            print('[Device] no GPU available, use cpu')
            cfg.device = "cpu"
    else:
        print('[Device] use cpu')
    device = torch.device("cuda:{}".format(local_rank))
    

    
    ### [data]
    if local_rank ==0: print('-'*50)
    if local_rank ==0: print('[Dataset] preparing dataset: ' + cfg.dataset, '...')
    
    datafolder = './data/dataset/cifar10'
    datashape, img_mean, img_std, nclasses = [1, 3, 32, 32], [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784], 10
    img_size = datashape[-1]
    
    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]
    augmentations = []
    from utils.autoaug import CIFAR10Policy
    augmentations += [
        CIFAR10Policy()
    ]
    augmentations += [
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
        ]

    augmentations = transforms.Compose(augmentations)
    
    train_dataset = datasets.CIFAR10(root = datafolder, train = True, download = True, transform = augmentations)

    val_dataset = datasets.CIFAR10(
        root=datafolder, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize,
        ]))

    testloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.train_batch_size, shuffle=False,
        num_workers=2)
    
    if nGPUs > 1:
        train_sampler = DistributedSampler(dataset=train_dataset)
        trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cfg.train_batch_size, sampler=train_sampler, num_workers=2)
    else:
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=2)
        
    
    if local_rank ==0: print('[Dataset] number of training images: ', len(trainloader.dataset), ' test images: ', len(testloader.dataset))
    if local_rank ==0: print('[Dataset] data shape: ', datashape, ' num of classes: ', nclasses)
    if local_rank ==0: print('[Dataset] batch_size: ', cfg.train_batch_size)

    ### [model]
    if local_rank ==0: print('-'*50)
    if local_rank ==0: print('[Model] constructing model ' + cfg.model, '...')
    model = VisionTransformer(img_size=datashape[-1], patch_size=cfg.model_patch_size, in_chans=datashape[1], num_classes=nclasses, embed_dim=cfg.model_embed_dim, depth=cfg.model_num_layers,
                 num_heads=cfg.model_num_heads, mlp_ratio=cfg.model_mlp_ratio,
                 drop_rate=cfg.train_dropout_rate, attn_drop_rate=cfg.train_attn_dropout_rate, drop_path_rate=cfg.train_drop_path_rate)
    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if local_rank ==0: print(f'[Model] patch size: {cfg.model_patch_size}, embed dim: {cfg.model_embed_dim}, depth: {cfg.model_num_layers}, num of heads: {cfg.model_num_heads}, mlp ratio: {cfg.model_mlp_ratio}, dropout rate: {cfg.train_dropout_rate}, attn_drop_rate: {cfg.train_attn_dropout_rate}')
    
    if local_rank ==0: print(f'[Model] stochastic depth rule: linear, start from 0 to {cfg.train_drop_path_rate}')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if local_rank ==0: print(f'[Model] num of trainable parameters: {total_params}')
    
    ### [training]
    if local_rank ==0: print('-'*50)
    if cfg.optim_loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss().to(device)
        if local_rank ==0: print(f'[Train] criterion is {cfg.optim_loss}')
    if cfg.optim_alg == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.optim_lr)
        if local_rank ==0: print(f'[Train] algorithm: {cfg.optim_alg}, learning rate: {cfg.optim_lr}')
    if local_rank ==0: print(f'[Train] nepochs: {cfg.train_num_epochs}')
    save_name = 'checkpoint_' + cfg.dataset +'_' + cfg.model + '-' + str(datetime.date.today()) + '.pt'
    save_path = os.path.join(cfg.save_path,save_name)
    if local_rank ==0: print(f'[Train] kinetic lambda: {cfg.train_kinetic_lambda}')
    trainer(model, trainloader, local_rank, device, optimizer, criterion, cfg.train_num_epochs, cfg.save_epochs, save_path, test_dataloader=testloader, kinetic_lambda = cfg.train_kinetic_lambda)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--status", type=str, default="train", choices=["train", "test"])
    
    parser.add_argument("--randomseed", type=int, default=9999)
    
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--local_rank", type=int, default = 0, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist","cifar10","cifar100"])
    
    parser.add_argument("--model", type=str, default="ViT_kinetic", choices=["ViT_kinetic"])
    parser.add_argument("--model_patch_size", type=int, default=4)
    parser.add_argument("--model_embed_dim", type=int, default=128)
    parser.add_argument("--model_num_layers", type=int, default=20)
    parser.add_argument("--model_num_heads", type=int, default=16)
    parser.add_argument("--model_mlp_ratio", type=float, default=4.)
    
    parser.add_argument("--train_batch_size", type=int, default=100)
    parser.add_argument("--train_num_epochs", type=int, default=200)
    parser.add_argument("--train_attn_dropout_rate", type=float, default=0.0)
    parser.add_argument("--train_dropout_rate", type=float, default=0.0)
    parser.add_argument("--train_drop_path_rate", type=float, default=0.5)
    parser.add_argument("--train_kinetic_lambda", type=float, default = 1.0)
    
    parser.add_argument("--optim_loss", type=str, default="cross_entropy")
    parser.add_argument("--optim_alg", type=str, default="adam")
    parser.add_argument("--optim_lr", type=float, default=1e-3)
    
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="./results/")

    cfg = parser.parse_args()
    main(cfg)

