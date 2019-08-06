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

from advertorch.attacks import LinfPGDAttack, L2PGDAttack, CarliniWagnerL2Attack, GradientSignAttack

from utils import get_dataset, cal_parameters


def attack_run(model, adversary, hps):
    model.eval()
    dataset = get_dataset(data_name=hps.problem, train=False)
    # hps.n_batch_test = 1
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    model.eval()
    test_clnloss = 0
    clncorrect = 0
    test_advloss = 0
    advcorrect = 0

    attack_path = os.path.join(hps.attack_dir, hps.attack)
    if not os.path.exists(attack_path):
        os.mkdir(attack_path)

    for batch_id, (clndata, target) in enumerate(test_loader):
        # Note that images are scaled to [-1.0, 1.0]
        clndata, target = clndata.to(hps.device), target.to(hps.device)
        path = os.path.join(attack_path, 'original_{}.png'.format(batch_id))
        save_image(clndata, path, normalize=True)

        with torch.no_grad():
            output = model(clndata)

        #print('original logits ', output.detach().cpu().numpy())
        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        #print('pred: ', pred)
        clncorrect += pred.eq(target.view_as(pred)).sum().item()

        advdata = adversary.perturb(clndata, target)
        path = os.path.join(attack_path, '{}perturbed_{}.png'.format(prefix, batch_id))
        save_image(advdata, path, normalize=True)

        with torch.no_grad():
            output = model(advdata)
        #print('adv logits ', output.detach().cpu().numpy())

        test_advloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        #print('pred: ', pred)
        advcorrect += pred.eq(target.view_as(pred)).sum().item()

        #if batch_id == 2:
        #    exit(0)

    test_clnloss /= len(test_loader.dataset)
    print('Test set: avg cln loss: {:.4f},'
          ' cln acc: {}/{}'.format(
        test_clnloss, clncorrect, len(test_loader.dataset)))

    test_advloss /= len(test_loader.dataset)
    print('Test set: avg adv loss: {:.4f},'
          ' adv acc: {}/{}'.format(
        test_advloss, advcorrect, len(test_loader.dataset)))

    cln_acc = clncorrect / len(test_loader.dataset)
    adv_acc = advcorrect / len(test_loader.dataset)
    return cln_acc, adv_acc


def attack_run_rejection_policy(model, adversary, hps):
    model.eval()
    # Get thresholds
    threshold_list = []
    for label_id in range(hps.n_classes):
        # No data augmentation(crop_flip=False) when getting in-distribution thresholds
        dataset = get_dataset(data_name=hps.problem, train=True, label_id=label_id, crop_flip=False)
        in_test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

        print('Inference on {}, label_id {}'.format(hps.problem, label_id))
        in_ll_list = []
        for batch_id, (x, y) in enumerate(in_test_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)
            ll = model(x)

            correct_idx = ll.argmax(dim=1) == y

            ll_, y_ = ll[correct_idx], y[correct_idx]  # choose samples are classified correctly
            in_ll_list += list(ll_[:, label_id].detach().cpu().numpy())
        
        thresh = sorted(in_ll_list)[50]
        print('len: {}, threshold (min ll): {:.4f}'.format(len(in_ll_list), thresh))
        threshold_list.append(thresh)  # class mean as threshold

    # Evaluation
    dataset = get_dataset(data_name=hps.problem, train=False)
    # hps.n_batch_test = 1
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    clncorrect = 0
    cln_reject = 0
    advcorrect = 0
    adv_reject = 0

    attack_path = os.path.join(hps.attack_dir, hps.attack)
    if not os.path.exists(attack_path):
        os.mkdir(attack_path)

    thresholds = torch.tensor(threshold_list).to(hps.device)

    for batch_id, (clndata, target) in enumerate(test_loader):
        # Note that images are scaled to [-1.0, 1.0]
        clndata, target = clndata.to(hps.device), target.to(hps.device)
        path = os.path.join(attack_path, 'original_{}.png'.format(batch_id))
        save_image(clndata, path, normalize=True)

        with torch.no_grad():
            output = model(clndata)

        # print('original logits ', output.detach().cpu().numpy())
        # test_clnloss += F.cross_entropy(
        #     output, target, reduction='sum').item()
        values, pred = output.max(dim=1)
        confidence_idx = values >= thresholds[pred]
        reject_idx = values < thresholds[pred]

        clncorrect += pred[confidence_idx].eq(target[confidence_idx]).sum().item()
        cln_reject += reject_idx.float().sum().item()

        advdata = adversary.perturb(clndata, target)
        path = os.path.join(attack_path, '{}perturbed_{}.png'.format(prefix, batch_id))
        save_image(advdata, path, normalize=True)

        with torch.no_grad():
            output = model(advdata)
        # print('adv logits ', output.detach().cpu().numpy())

        # test_advloss += F.cross_entropy(
        #     output, target, reduction='sum').item()
        values, pred = output.max(dim=1)
        confidence_idx = values >= thresholds[pred]
        reject_idx = values < thresholds[pred]

        # pred = output.max(1, keepdim=True)[1]
        advcorrect += pred[confidence_idx].eq(target[confidence_idx]).sum().item()
        adv_reject += reject_idx.float().sum().item()

        # if batch_id == 2:
        #     exit(0)

    n = len(test_loader.dataset)
    print('Test set: cln acc: {:.4f}, reject rate: {:.4f}'.format(clncorrect / n, cln_reject / n))
    print('Test set: adv acc: {:.4f}, reject success rate: {:.4f}'.format(advcorrect / n, adv_reject / n))

    cln_acc = clncorrect / len(test_loader.dataset)
    adv_acc = advcorrect / len(test_loader.dataset)
    return cln_acc, adv_acc


def fgsm_attack(model, hps):
    eps_list = [0., 0.1, 0.2, 0.4, 0.5]

    print('============== FGSM Summary ===============')

    for eps in eps_list:
        adversary = GradientSignAttack(
            model,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=eps,
            clip_min=-1.,
            clip_max=1.,
            targeted=hps.targeted
        )
        print('epsilon = {:.4f}'.format(adversary.eps))
        attack_run(model, adversary, hps)

    print('============== FGSM Summary ===============')


def linfPGD_attack(model, hps):
    eps_list = [0.1, 0.3, 0.5, 0.7]

    #hps.n_batch_test = 5
    print('============== LinfPGD Summary ===============')
    for eps in eps_list:
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=-1.0,
            clip_max=1.0, targeted=hps.targeted)
        print('epsilon = {:.4f}'.format(adversary.eps))
        #attack_run(model, adversary, hps)
        attack_run_rejection_policy(model, adversary, hps)

    print('============== LinfPGD Summary ===============')


def l2PGD_attack(model, hps):
    eps_list = [0.1, 0.3, 0.5, 0.7]

    print('============== L2PGD Summary ===============')
    for eps in eps_list:
        adversary = L2PGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=-1.0,
            clip_max=1.0, targeted=hps.targeted)
        print('epsilon = {:.4f}'.format(adversary.eps))
        attack_run(model, adversary, hps)

    print('============== L2PGD Summary ===============')


def cw_l2_attack(model, hps):
    confidence_list = [0., 10, 20, 30, 30]

    print('============== CW_l2 Summary ===============')
    for confidence in confidence_list:
        adversary = CarliniWagnerL2Attack(model,
                                          num_classes=10,
                                          confidence=confidence,
                                          clip_min=0.,
                                          clip_max=1.,
                                          max_iterations=500
                                          )
        print('confidence = {}'.format(adversary.confidence))
        hps.n_batch_test = 1
        attack_run(model, adversary, hps)

    print('============== CW_l2 Summary ===============')


if __name__ == "__main__":
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--inference", action="store_true",
                        help="Used in inference mode")
    parser.add_argument("--log_dir", type=str,
                        default='./logs', help="Location to save logs")
    parser.add_argument("--attack_dir", type=str,
                        default='./attack_logs', help="Location to save logs")

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
                        default=16, help="Minibatch size")
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

    # Attack parameters
    parser.add_argument("--targeted", action="store_true",
                        help="whether perform targeted attack")
    parser.add_argument("--attack", type=str, default='pgdinf',
                        help="Location of data")

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

    if not os.path.exists(hps.log_dir):
        os.mkdir(hps.log_dir)

    if not os.path.exists(hps.attack_dir):
        os.mkdir(hps.attack_dir)
    # from cw_attack import cw
    # model.eval()
    #
    # for batch_id, (x, y) in enumerate(test_loader):
    #     x = x.to(hps.device)
    #     y = y.to(hps.device)
    #     adv_example, noise, adv_logits = cw(model, x, y, targeted=False, max_iter=2000, learning_rate=2e-3)
    #     save_image(x, os.path.join(image_dir, 'original{}.png'.format(batch_id)))
    #     save_image(adv_example, os.path.join(image_dir, 'adv{}.png'.format(batch_id)))
    #     save_image(noise, os.path.join(image_dir, 'noise{}.png'.format(batch_id)))
    #     print('logits: ', model(x).detach().numpy())
    #     print('adv logits: ', adv_logits.detach().numpy())
    #     if batch_id == 0:
    #         break
    # exit(0)

    if hps.attack == 'pgdinf':
        linfPGD_attack(model, hps)
    elif hps.attack == 'pgd2':
        l2PGD_attack(model, hps)
    elif hps.attack == 'cw':
        cw_l2_attack(model, hps)
    elif hps.attack == 'fgsm':
        fgsm_attack(model, hps)

