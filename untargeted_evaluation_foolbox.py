import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from resnet import build_resnet_32x32
from sdim import SDIM

from utils import get_dataset, cal_parameters

from foolbox.attacks import BoundaryAttack, SpatialAttack, DeepFoolL2Attack, LocalSearchAttack
import numpy as np

import foolbox


def attack_run_rejection_policy(model, hps):
    """
    An attack run with rejection policy.
    :param model: Pytorch model.
    :param adversary: Advertorch adversary.
    :param hps: hyperparameters
    :return:
    """
    model.eval()

    # Get thresholds
    threshold_list1 = []
    threshold_list2 = []
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

        thresh_idx = int(0.01 * len(in_ll_list))
        thresh1 = sorted(in_ll_list)[thresh_idx]
        thresh_idx = int(0.02 * len(in_ll_list))
        thresh2 = sorted(in_ll_list)[thresh_idx]
        threshold_list1.append(thresh1)  # class mean as threshold
        threshold_list2.append(thresh2)  # class mean as threshold
        print('1st & 2nd percentile thresholds: {:.3f}, {:.3f}'.format(thresh1, thresh2))

    # Evaluation
    n_eval = 0   # total number of correct classified samples by clean classifier
    n_successful_adv = 0  # total number of successful adversarial examples generated
    n_rejected_adv1 = 0   # total number of successfully rejected (successful) adversarial examples, <= n_successful_adv
    n_rejected_adv2 = 0   # total number of successfully rejected (successful) adversarial examples, <= n_successful_adv

    attack_path = os.path.join(hps.attack_dir, hps.attack)
    if not os.path.exists(attack_path):
        os.mkdir(attack_path)

    thresholds1 = torch.tensor(threshold_list1).to(hps.device)
    thresholds2 = torch.tensor(threshold_list2).to(hps.device)

    l2_distortion_list = []

    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1.), num_classes=10)

    hps.n_batch_test = 1
    dataset = get_dataset(data_name=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    for batch_id, (x, y) in enumerate(test_loader):
        # Note that images are scaled to [0., 1.0]
        x, y = x.to(hps.device), y.to(hps.device)
        with torch.no_grad():
            output = model(x)

        pred = output.argmax(dim=1)
        if pred != y:
            continue

        n_eval += 1

        img, label = x[0], y[0]
        if hps.attack == 'boundary':
            attack = BoundaryAttack(fmodel)
            adv_x = attack(img.cpu().numpy(), label.cpu().numpy(), log_every_n_steps=10000)
        elif hps.attack == 'deepfool':
            attack = DeepFoolL2Attack(fmodel)
            adv_x = attack(img.cpu().numpy(), label.cpu().numpy())
        elif hps.attack == 'local':
            attack = LocalSearchAttack(fmodel)
            adv_x = attack(img.cpu().numpy(), label.cpu().numpy())
        elif hps.attack == 'spatial':
            attack = SpatialAttack(fmodel)
            adv_x = attack(img.cpu().numpy(), label.cpu().numpy())
        elif hps.attack == 'jsma':
            attack = SpatialAttack(fmodel)
            adv_x = attack(img.cpu().numpy(), label.cpu().numpy())
        else:
            raise ValueError('param attack {} not available.'.format(hps.attack))

        adv_x = torch.tensor(adv_x).unsqueeze(dim=0).to(hps.device)

        with torch.no_grad():
            output = model(adv_x)

        logit, pred = output.max(dim=1)

        if pred != label:
            n_successful_adv += 1

        diff = adv_x - x
        l2_distortion = diff.norm(p=2, dim=-1).mean().item()  # mean l2 distortion
        l2_distortion_list.append(l2_distortion)

        if logit < thresholds1[pred]:
            n_rejected_adv1 += 1

        if logit < thresholds2[pred]:
            n_rejected_adv2 += 1

        if batch_id == 100:
            print('Evaluating on {}-th batch ...'.format(batch_id))
            break  # only one batch

    reject_rate1 = n_rejected_adv1 / n_successful_adv
    reject_rate2 = n_rejected_adv2 / n_successful_adv
    success_adv_rate = n_successful_adv / n_eval
    print('success rate of adv examples generation: {}/{}={:.4f}'.format(n_successful_adv, n_eval, success_adv_rate))
    print('Mean L2 distortion of Adv Examples: {:.4f}'.format(np.mean(l2_distortion_list)))
    print('1st percentile, reject success rate: {}/{}={:.4f}'.format(n_rejected_adv1, n_successful_adv, reject_rate1))
    print('2nd percentile, reject success rate: {}/{}={:.4f}'.format(n_rejected_adv2, n_successful_adv, reject_rate2))


# def boundary_attack(model, hps):
#
#     print('============== Boundary Summary ===============')
#     adversary = BoundaryAttack(model, targeted=True)
#     attack_run_rejection_policy(model, adversary, hps)
#
#     print('============== Boundary Summary ===============')


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
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
                        default=100, help="Minibatch size")
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
                        default=64, help="size of the global representation from encoder")
    parser.add_argument("--encoder_name", type=str, default='resnet25',
                        help="encoder name: resnet#")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Inference hyperparams:
    parser.add_argument("--percentile", type=float, default=0.01,
                        help="percentile value for inference with rejection.")
    parser.add_argument("--cw_confidence", type=int, default=0,
                        help="confidence for CW attack.")

    # Attack parameters
    parser.add_argument("--targeted", action="store_true",
                        help="whether perform targeted attack")
    parser.add_argument("--attack", type=str, default='boundary',
                        help="Location of data")

    # Ablation
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    hps.device = torch.device("cuda" if use_cuda else "cpu")

    if hps.problem == 'cifar10':
        hps.image_channel = 3
    elif hps.problem == 'svhn':
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

        checkpoint_path = os.path.join(hps.log_dir,
                                       'sdim_{}_{}_d{}.pth'.format(hps.encoder_name, hps.problem, hps.rep_size))
        model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        mu, log_sigma = model.class_conditional.class_embed.weight.split(dim=1, split_size=hps.rep_size)
        sigma = torch.exp(log_sigma).mean(dim=1)
        print("sigma: ", sigma.cpu().detach().numpy())
        mu_list = mu.split(dim=0, split_size=1)
        import itertools
        distances = [(a - b).norm(p=2) for (a, b) in itertools.combinations(mu_list, 2)]
        print('mean distances of centers: ', np.mean(distances))

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

    # if hps.attack == 'boundary':
    #     local_search_attack(model, hps)
    # elif hps.attack == 'cw':
    #     cw_l2_attack(model, hps)
    attack_run_rejection_policy(model, hps)

