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
import torchvision
import torchvision.transforms as transforms

from data.loader.dataloaders_LAP import dataloaders
from training.trainer_ViT import trainer

import timm
import argparse

def main(cfg):
    print('-'*50)
    print('[Config]', cfg)
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
    datafolder = './data/dataset/cifar10'
    datashape, mean, std, nclasses = (1, 3, 32, 32), (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784), 10
    transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    data_aug = [transforms.RandomCrop(32, padding = 4), transforms.Resize(224),transforms.RandomHorizontalFlip()]
    
    trainset = torchvision.datasets.CIFAR10(root = datafolder, train = True, download = True, transform = transforms.Compose(data_aug + transform))
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root = datafolder, train = False, download = True, transform = transforms.Compose([transforms.Resize(224)]+transform))
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=2)
    
    print('[Dataset] number of training images: ', len(trainloader.dataset), ' test images: ', len(testloader.dataset))
    print('[Dataset] data shape: ', datashape, ' num of classes: ', nclasses)
    print('[Dataset] batch_size: ', cfg.train_batch_size)

    ### [model]
    print('-'*50)
    print('[Model] constructing model ' + cfg.model, '...')
    model = timm.create_model("vit_small_patch16_224", pretrained=True)
    model.head = nn.Linear(model.head.in_features, 10)
    print(model)
    
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Model] num of trainable parameters: {total_params}')
    
    ### [training]
    print('-'*50)
    if cfg.optim_loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss().to(device)
        print(f'[Train] criterion is {cfg.optim_loss}')
    if cfg.optim_alg == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.optim_lr)
        print(f'[Train] algorithm: {cfg.optim_alg}, learning rate: {cfg.optim_lr}')
    print(f'[Train] nepochs: {cfg.train_num_epochs}')
    save_name = 'checkpoint_' + cfg.dataset +'_' + cfg.model + '-' + str(datetime.date.today()) + '.pt'
    save_path = os.path.join(cfg.save_path,save_name)
    trainer(model, trainloader, device, optimizer, criterion, cfg.train_num_epochs, cfg.save_epochs, save_path, test_dataloader=testloader)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--status", type=str, default="train", choices=["train", "test"])
    
    parser.add_argument("--randomseed", type=int, default=int(str(time.time()).split(".")[-1][::-1]))
    
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist","cifar10","cifar100"])
    
    parser.add_argument("--model", type=str, default="ViT_pretrained", choices=["ViT_kinetic"])
    parser.add_argument("--model_patch_size", type=int, default=4)
    parser.add_argument("--model_embed_dim", type=int, default=128)
    parser.add_argument("--model_num_layers", type=int, default=20)
    parser.add_argument("--model_num_heads", type=int, default=16)
    parser.add_argument("--model_mlp_ratio", type=float, default=4.)
    
    parser.add_argument("--train_batch_size", type=int, default=40)
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

