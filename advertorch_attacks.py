import argparse
import sys
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim import Adam

from resnet import build_resnet_32x32
from sdim import SDIM

from advertorch.attacks import LinfPGDAttack, L2PGDAttack, CarliniWagnerL2Attack


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
    parser.add_argument("--attack", type=str, default='pgdinf',
                        help="Location of data")

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
                        default=64, help="Minibatch size")
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

    prefix = ''
    if hps.encoder_name.startswith('sdim_'):
        prefix = 'sdim_'
        hps.encoder_name = hps.encoder_name.strip('sdim_')
        model = SDIM(rep_size=hps.rep_size,
                     mi_units=hps.mi_units,
                     encoder_name=hps.encoder_name,
                     image_channel=hps.image_channel
                     ).to(hps.device)

        checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}.pth'.format(hps.encoder_name, hps.problem, hps.rep_size))
        model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    else:
        n_encoder_layers = int(hps.encoder_name.strip('resnet'))
        model = build_resnet_32x32(n=n_encoder_layers,
                                   fc_size=hps.n_classes,
                                   image_channel=hps.image_channel
                                   ).to(hps.device)

        checkpoint_path = os.path.join(hps.log_dir, '{}_{}.pth'.format(hps.encoder_name, hps.problem))
        model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    print('Model name: {}'.format(hps.encoder_name))
    print('==>  # Model parameters: {}.'.format(cal_parameters(model)))

    dataset = get_dataset(dataset=hps.problem, train=False)
    hps.n_batch_test = 1
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    image_dir = 'images'
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    from cw_attack import cw
    model.eval()

    for batch_id, (x, y) in enumerate(test_loader):
        adv_example, noise, adv_logits = cw(model, x, y, targeted=False, max_iter=2000, learning_rate=2e-3)
        save_image(x, os.path.join(image_dir, 'original{}.png'.format(batch_id)))
        save_image(adv_example, os.path.join(image_dir, 'adv{}.png'.format(batch_id)))
        save_image(noise, os.path.join(image_dir, 'noise{}.png'.format(batch_id)))
        print('logits: ', model(x).detach().numpy())
        print('adv logits: ', adv_logits.detach().numpy())
        if batch_id == 0:
            break
    exit(0)

    if hps.attack == 'pgdinf':
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=False)
    elif hps.attack == 'pgd2':
        adversary = L2PGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=False)
    elif hps.attack == 'cw':
        adversary = CarliniWagnerL2Attack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            num_classes=10,
            clip_min=0.0,
            clip_max=1.0,
            targeted=False)



    model.eval()
    test_clnloss = 0
    clncorrect = 0
    test_advloss = 0
    advcorrect = 0

    for batch_id, (clndata, target) in enumerate(test_loader):
        clndata, target = clndata.to(hps.device), target.to(hps.device)
        path = os.path.join(image_dir, 'original_{}.png'.format(batch_id))
        save_image(clndata, path, normalize=True)

        with torch.no_grad():
            output = model(clndata)
        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()

        advdata = adversary.perturb(clndata, target)
        path = os.path.join(image_dir, '{}perturbed_{}.png'.format(prefix, batch_id))
        save_image(advdata, path, normalize=True)

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