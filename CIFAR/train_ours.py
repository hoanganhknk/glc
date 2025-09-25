# -*- coding: utf-8 -*-

import argparse
import os
import time
import math
import json
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from resnet import resnet32
import numpy as np
from load_corrupted_data import CIFAR10, CIFAR100
from PIL import Image
import socket

# note: nosgdr, schedule, and epochs are highly related settings

parser = argparse.ArgumentParser(description='Trains WideResNet on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Positional arguments
parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
# Optimization options
parser.add_argument('--nosgdr', default=False, action='store_true', help='Turn off SGDR.')
parser.add_argument('--epochs', '-e', type=int, default=75, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--gold_fraction', '-gf', type=float, default=0.1, help='What fraction of the data should be trusted?')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0.3, help='The label corruption probability.')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif', help='Type of corruption ("unif" or "flip").')
parser.add_argument('--adjust', '-a', action='store_true', help='Adjust the C_hat estimate with base-rate information.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--test_bs', type=int, default=128)
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs. Use when SGDR is off.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--nonlinearity', type=str, default='relu', help='Nonlinearity (relu, elu, gelu).')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# i/o
parser.add_argument('--log', type=str, default='./', help='Log folder.')
# random seed
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

print()
print("This is on machine:", socket.gethostname())
print()
print(args)
print()

# Device
use_cuda = args.ngpu > 0 and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# Init logger
os.makedirs(args.log, exist_ok=True)
log = open(os.path.join(args.log, args.dataset + '_log.txt'), 'w')
state = {k: v for k, v in args._get_kwargs()}
state['tt'] = 0      # SGDR variable
state['init_learning_rate'] = args.learning_rate
log.write(json.dumps(state) + '\n')

# Init dataset
os.makedirs(args.data_path, exist_ok=True)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data_gold = CIFAR10(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True)
    train_data_silver = CIFAR10(
        args.data_path, True, False, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices, seed=args.seed)
    train_data_gold_deterministic = CIFAR10(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=test_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices)
    test_data = CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10

elif args.dataset == 'cifar100':
    train_data_gold = CIFAR100(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True)
    train_data_silver = CIFAR100(
        args.data_path, True, False, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices, seed=args.seed)
    train_data_gold_deterministic = CIFAR100(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=test_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices)
    test_data = CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, transform):
        # data_tensor: numpy array (N, H, W, C)
        # target_tensor: torch.LongTensor (N,)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data_tensor[index], self.target_tensor[index].item()
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)  # tensor (C,H,W)
        return img, target  # target as int for cleaner collation

    def __len__(self):
        return 50000

pin_memory = use_cuda
train_silver_loader = torch.utils.data.DataLoader(
    train_data_silver, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=pin_memory)
train_gold_deterministic_loader = torch.utils.data.DataLoader(
    train_data_gold_deterministic, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=pin_memory)
train_all_loader = torch.utils.data.DataLoader(
    TensorDataset(np.vstack((train_data_gold.train_data, train_data_silver.train_data)),
                  torch.from_numpy(np.array(train_data_gold.train_labels + train_data_silver.train_labels)).long(),
                  train_transform),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=pin_memory)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=pin_memory)

# Init checkpoints
os.makedirs(args.save, exist_ok=True)

net = resnet32(num_classes=num_classes)
print(net)

if use_cuda and args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

net = net.to(device)

if use_cuda:
    torch.cuda.manual_seed(args.seed)

optimizer = torch.optim.SGD(net.parameters(), state['init_learning_rate'],
                            momentum=args.momentum, weight_decay=args.decay, nesterov=True)

# saving so we can start again from these same weights when applying the correction
init_model_path = os.path.join(
    args.save, f"{args.dataset}_{args.gold_fraction}{args.corruption_prob}{args.corruption_type}_init.pytorch")
torch.save(net.state_dict(), init_model_path)

# Restore model (optional)
start_epoch = 0
# if args.load != '':
#     for i in range(args.epochs-1,-1,-1):
#         model_name = os.path.join(args.load, args.dataset + '_model_epoch' + str(i) + '.pytorch')
#         if os.path.isfile(model_name):
#             net.load_state_dict(torch.load(model_name, map_location=device))
#             start_epoch = i+1
#             print('Model restored! Epoch:', i)
#             break
#     if start_epoch == 0:
#         raise RuntimeError("could not resume")

if use_cuda:
    cudnn.benchmark = True  # fire on all cylinders

def train_phase1():
    net.train()
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(train_silver_loader):
        data = data.to(device)
        target = target.to(device)

        # forward
        output = net(data)

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + loss.item() * 0.2

        # cyclic learning rate (SGDR-like)
        if not args.nosgdr:
            dt = math.pi / float(args.epochs)
            state['tt'] += float(dt) / (len(train_silver_loader.dataset) / float(args.batch_size))
            if state['tt'] >= math.pi - 0.05:
                state['tt'] = math.pi - 0.05
            curT = math.pi/2.0 + state['tt']
            new_lr = args.learning_rate * (1.0 + math.sin(curT)) / 2.0  # lr_min = 0, lr_max = lr
            state['learning_rate'] = new_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

    state['train_loss'] = loss_avg

@torch.no_grad()
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)

        # forward
        output = net(data)
        loss = F.cross_entropy(output, target)

        # accuracy
        pred = output.argmax(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # test loss average
        loss_avg += loss.item()

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / float(total)

# Main loop phase 1
for epoch in range(start_epoch, args.epochs):
    # If you want step schedule instead of SGDR, uncomment this block:
    # if epoch < 150:
    #     state['learning_rate'] = state['init_learning_rate']
    # elif epoch < 225:
    #     state['learning_rate'] = state['init_learning_rate'] * args.gamma
    # else:
    #     state['learning_rate'] = state['init_learning_rate'] * (args.gamma ** 2)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = state['learning_rate']

    state['epoch'] = epoch

    begin_epoch = time.time()
    train_phase1()
    print('Epoch', epoch, '| Time Spent:', round(time.time() - begin_epoch, 2))

    test()

    # torch.save(net.state_dict(), os.path.join(args.save, args.dataset + '_model_epoch' + str(epoch) + '.pytorch'))
    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)

print('\nNow retraining with correction\n')

@torch.no_grad()
def get_C_hat_transpose():
    probs = []
    net.eval()
    for batch_idx, (data, target) in enumerate(train_gold_deterministic_loader):
        data = data.to(device)
        # we subtract num_classes because we added num_classes to gold so we could identify which example is gold
        target = (target.to(device) - num_classes)

        output = net(data)
        pred = F.softmax(output, dim=1)
        probs.extend(list(pred.detach().cpu().numpy()))

    probs = np.array(probs, dtype=np.float32)
    C_hat = np.zeros((num_classes, num_classes), dtype=np.float32)
    # Build per-class mean over gold set
    gold_labels_np = np.array(train_data_gold.train_labels) - num_classes
    for label in range(num_classes):
        indices = np.arange(len(gold_labels_np))[np.isclose(gold_labels_np, label)]
        if len(indices) > 0:
            C_hat[label] = np.mean(probs[indices], axis=0, keepdims=True)
        else:
            # Fallback: uniform if no gold for this label (shouldn't happen)
            C_hat[label] = np.ones((1, num_classes), dtype=np.float32) / num_classes
    return C_hat.T  # transpose

C_hat_transpose = torch.from_numpy(get_C_hat_transpose()).to(device)  # (num_classes, num_classes)

# /////// Resetting the network ////////
state = {k: v for k, v in args._get_kwargs()}
state['tt'] = 0      # SGDR variable
state['init_learning_rate'] = args.learning_rate
state['learning_rate'] = state['init_learning_rate']
for param_group in optimizer.param_groups:
    param_group['lr'] = state['learning_rate']

net.load_state_dict(torch.load(init_model_path, map_location=device))

def train_phase2(C_hat_transpose: torch.Tensor):
    net.train()
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(train_all_loader):
        # data: tensor (B,C,H,W), target: tensor of ints (B,)
        data = data.to(device)
        target = target.to(device)

        gold_indices = target > (num_classes - 1)
        silver_indices = target < num_classes
        gold_len = int(gold_indices.sum().item())
        silver_len = int(silver_indices.sum().item())

        optimizer.zero_grad()

        loss_s = torch.tensor(0.0, device=device)
        if silver_len > 0:
            data_s = data[silver_indices]
            target_s = target[silver_indices]
            output_s = net(data_s)  # (bs, K)
            # index rows of C_hat_transpose by target_s
            pre1 = C_hat_transpose[target_s.long()]  # (bs, K)
            pre2 = torch.mul(F.softmax(output_s, dim=1), pre1)  # (bs, K)
            loss_s = -(torch.log(pre2.sum(1))).sum()  # scalar

        loss_g = torch.tensor(0.0, device=device)
        if gold_len > 0:
            data_g = data[gold_indices]
            target_g = (target[gold_indices] - num_classes).long()
            output_g = net(data_g)
            loss_g = F.cross_entropy(output_g, target_g, reduction='sum')

        # backward
        loss = (loss_g + loss_s) / args.batch_size
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.2 + loss.item() * 0.8

        if not args.nosgdr:
            dt = math.pi / float(args.epochs)
            state['tt'] += float(dt) / (len(train_all_loader.dataset) / float(args.batch_size))
            if state['tt'] >= math.pi - 0.05:
                state['tt'] = math.pi - 0.05
            curT = math.pi/2.0 + state['tt']
            new_lr = args.learning_rate * (1.0 + math.sin(curT)) / 2.0
            state['learning_rate'] = new_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

    state['train_loss'] = loss_avg

# Main loop phase 2
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train_phase2(C_hat_transpose)
    print('Epoch', epoch, '| Time Spent:', round(time.time() - begin_epoch, 2))

    test()

    # torch.save(net.state_dict(), os.path.join(args.save, args.dataset + '_model_epoch' + str(epoch) + '.pytorch'))
    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)

log.close()

try:
    os.remove(init_model_path)
except Exception:
    pass
