import argparse
import sys
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam

from resnet import build_resnet_32x32

from advertorch.attacks import LinfPGDAttack


def cal_parameters(model):
    cnt = 0
    for para in model.parameters():
        cnt += para.numel()
    return cnt


def get_dataset(dataset='mnist', train=True):
    if dataset == 'mnist':
        dataset = datasets.MNIST('data/MNIST', train=train, download=True,
                                 transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                   ]))
    elif dataset == 'fashion':
        dataset = datasets.FashionMNIST('data/FashionMNIST', train=train, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize((32, 32)),
                                            transforms.ToTensor(),
                                        ]))
    elif dataset == 'cifar10':
        if train:
            transform = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                        ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        dataset = datasets.CIFAR10('data/CIFAR10', train=train, download=True, transform=transform)
    else:
        print('dataset {} is not available'.format(dataset))

    return dataset


if __name__ == "__main__":
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--inference", action="store_true",
                        help="Used in inference mode")
    parser.add_argument("--fgsm_attack", action="store_true",
                        help="Perform FGSM attack")
    parser.add_argument("--noise_attack", action="store_true",
                        help="Perform noise attack")
    parser.add_argument("--log_dir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Problem (mnist/fashion/cifar10")
    parser.add_argument("--n_classes", type=int,
                        default=10, help="number of classes of dataset.")
    parser.add_argument("--data_dir", type=str, default='data',
                        help="Location of data")

    # Optimization hyperparams:
    parser.add_argument("--n_batch_train", type=int,
                        default=128, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=200, help="Minibatch size")
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Total number of training epochs")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=32, help="Image size")
    parser.add_argument("--mi_units", type=int,
                        default=256, help="output size of 1x1 conv network for mutual information estimation")
    parser.add_argument("--rep_size", type=int,
                        default=128, help="size of the global representation from encoder")
    parser.add_argument("--encoder_name", type=str, default='resnet25',
                        help="encoder name: resnet#")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Ablation
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    hps.device = torch.device("cuda" if use_cuda else "cpu")

    if hps.problem == 'cifar10':
        hps.image_channel = 3
    elif hps.problem == 'mnist':
        hps.image_channel = 1

    n_encoder_layers = int(hps.encoder_name.strip('resnet'))
    model = build_resnet_32x32(n=n_encoder_layers,
                               fc_size=hps.n_classes,
                               image_channel=hps.image_channel
                               ).to(hps.device)

    optimizer = Adam(model.parameters(), lr=hps.lr)

    print('==>  # Model parameters: {}.'.format(cal_parameters(model)))

    checkpoint_path = os.path.join(hps.log_dir, '{}_{}.pth'.format(hps.encoder_name, hps.problem))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=False)

    dataset = get_dataset(dataset=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    model.eval()
    test_clnloss = 0
    clncorrect = 0
    test_advloss = 0
    advcorrect = 0

    for clndata, target in test_loader:
        clndata, target = clndata.to(hps.device), target.to(hps.device)
        with torch.no_grad():
            output = model(clndata)
        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()

        advdata = adversary.perturb(clndata, target)
        with torch.no_grad():
            output = model(advdata)
        test_advloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        advcorrect += pred.eq(target.view_as(pred)).sum().item()

    test_clnloss /= len(test_loader.dataset)
    print('\nTest set: avg cln loss: {:.4f},'
          ' cln acc: {}/{} ({:.0f}%)\n'.format(
        test_clnloss, clncorrect, len(test_loader.dataset),
        100. * clncorrect / len(test_loader.dataset)))

    test_advloss /= len(test_loader.dataset)
    print('Test set: avg adv loss: {:.4f},'
          ' adv acc: {}/{} ({:.0f}%)\n'.format(
        test_advloss, advcorrect, len(test_loader.dataset),
        100. * advcorrect / len(test_loader.dataset)))