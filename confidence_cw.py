# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim

from advertorch.utils import calc_l2distsq
from advertorch.utils import tanh_rescale
from advertorch.utils import torch_arctanh
from advertorch.utils import clamp
from advertorch.utils import to_one_hot
from advertorch.utils import replicate_input

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import is_successful


CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
NUM_CHECKS = 10


class CarliniWagnerL2Attack(Attack, LabelMixin):
    """
    The Carlini and Wagner L2 Attack, https://arxiv.org/abs/1608.04644
    :param predict: forward pass function.
    :param num_classes: number of clasess.
    :param threshold: target threshold.
    :param confidence: confidence of the adversarial examples.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm
    :param binary_search_steps: number of binary search times to find the
        optimum
    :param max_iterations: the maximum number of iterations
    :param abort_early: if set to true, abort early if getting stuck in local
        min
    :param initial_const: initial value of the constant c
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function
    """

    def __init__(self, predict, num_classes, c=1, confidence=0,
                 targeted=False, learning_rate=0.01,
                 binary_search_steps=9, max_iterations=10000,
                 abort_early=True, initial_const=1e-3,
                 clip_min=0., clip_max=1., loss_fn=None, ):
        """Carlini Wagner L2 Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently do not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        loss_fn = None

        super(CarliniWagnerL2Attack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.c = c
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.confidence = confidence
        self.initial_const = initial_const
        self.num_classes = num_classes
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        self.targeted = targeted

    def _loss_fn(self, output, y_onehot, l2distsq, const):
        # TODO: move this out of the class and make this the default loss_fn
        #   after having targeted tests implemented
        real = (y_onehot * output).sum(dim=1)

        # TODO: make loss modular, write a loss class
        other = ((1.0 - y_onehot) * output - (y_onehot * TARGET_MULT)
                 ).max(1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if self.targeted:
            adv_loss = clamp(other - real + self.confidence, min=0.)
            threshold_loss = clamp(self.threshold - real, min=0.)
            #threshold_loss = clamp(real - self.threshold, min=0.)
        else:
            adv_loss = clamp(real - other + self.confidence, min=0.)
            #threshold_loss = clamp(self.threshold - other, min=0.)
        l2dist = (l2distsq).sum()
        # const = 0.001
        adv_loss = torch.sum(const * adv_loss)
        threshold_loss = torch.sum(const * threshold_loss)
        # print('const ', const)

        loss = l2dist + adv_loss  #+ threshold_loss
        #loss = loss1 + loss2 + threshold_loss
        return loss, l2dist, adv_loss, threshold_loss

    def _get_arctanh_x(self, x):
        result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
                       min=self.clip_min, max=self.clip_max) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)
        x = replicate_input(x)
        # batch_size = len(x)

        final_advs = x
        x_atanh = self._get_arctanh_x(x)
        y_onehot = to_one_hot(y, self.num_classes).float()

        delta = nn.Parameter(torch.zeros_like(x))
        optimizer = optim.Adam([delta], lr=self.learning_rate)
        prevloss = PREV_LOSS_INIT

        for ii in range(self.max_iterations):
            # loss, l2distsq, output, adv_img = \
            #     self._forward_and_update_delta(
            #         optimizer, x_atanh, delta, y_onehot, self.c)

            optimizer.zero_grad()
            adv = tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)
            transimgs_rescale = tanh_rescale(x_atanh, self.clip_min, self.clip_max)
            output = self.predict(adv)
            l2distsq = calc_l2distsq(adv, transimgs_rescale)
            loss, l2dist, adv_loss, threshold_loss = self._loss_fn(output, y_onehot, l2distsq, self.c)
            loss.backward()
            optimizer.step()

            if ii % 1000 == 1:
                print('step: {}, dis: {:.2f}, loss1: {:.2f}, threshold_loss: {:.2f}'.format(ii, l2dist.item(), adv_loss.item(),
                                                                              threshold_loss.item()))

            if self.abort_early:
                if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                    if loss > prevloss * ONE_MINUS_EPS:
                        break
                    prevloss = loss

            final_advs = adv.data
        return final_advs


def targeted_cw(model, adversary, hps):
    """
       An attack run with rejection policy.
       :param model: Pytorch model.
       :param adversary: Advertorch adversary.
       :param hps: hyperparameters
       :return:
       """
    model.eval()
    dataset = get_dataset(data_name=hps.problem, train=False, label_id=0)
    hps.n_batch_test = 1
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    import os
    for batch_id, (x, y) in enumerate(test_loader):
        # Note that images are scaled to [0., 1.0]
        x, y = x.to(hps.device), y.to(hps.device)

        y_targets = torch.arange(hps.n_classes).long().to(hps.device)

        for i in range(hps.n_classes):
            yy = y_targets[i]
            if yy != y:
                adv_x = adversary.perturb(x, y_targets)
            else:
                adv_x = x

            with torch.no_grad():
                output = model(adv_x)

            path = os.path.join(hps.attack_dir, 'targeted_cw_{}_{}_{}.png'.format(
                hps.problem, hps.cw_confidence, yy.cpu().item()))
            save_image(adv_x, path)

            print('target: {}, logits: {}'.format(y.cpu().item(), output.tolist()))

        break


def attack_run_rejection_policy(model, adversary, hps):
    """
    An attack run with rejection policy.
    :param model: Pytorch model.
    :param adversary: Advertorch adversary.
    :param hps: hyperparameters
    :return:
    """
    model.eval()

    # # Get thresholds
    # threshold_list = []
    # for label_id in range(hps.n_classes):
    #     # No data augmentation(crop_flip=False) when getting in-distribution thresholds
    #     dataset = get_dataset(data_name=hps.problem, train=True, label_id=label_id, crop_flip=False)
    #     in_test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)
    #
    #     print('Inference on {}, label_id {}'.format(hps.problem, label_id))
    #     in_ll_list = []
    #     for batch_id, (x, y) in enumerate(in_test_loader):
    #         x = x.to(hps.device)
    #         y = y.to(hps.device)
    #         ll = model(x)
    #
    #         correct_idx = ll.argmax(dim=1) == y
    #
    #         ll_, y_ = ll[correct_idx], y[correct_idx]  # choose samples are classified correctly
    #         in_ll_list += list(ll_[:, label_id].detach().cpu().numpy())
    #
    #     thresh_idx = int(hps.percentile * len(in_ll_list))
    #     thresh = sorted(in_ll_list)[thresh_idx]
    #     print('threshold_idx/total_size: {}/{}, threshold: {:.3f}'.format(thresh_idx, len(in_ll_list), thresh))
    #     threshold_list.append(thresh)  # class mean as threshold
    #
    # Evaluation
    dataset = get_dataset(data_name=hps.problem, train=False)
    hps.n_batch_test = 1
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    # n_correct = 0   # total number of correct classified samples by clean classifier
    # n_successful_adv = 0  # total number of successful adversarial examples generated
    # n_rejected_adv = 0   # total number of successfully rejected (successful) adversarial examples, <= n_successful_adv
    #
    # attack_path = os.path.join(hps.attack_dir, hps.attack)
    # if not os.path.exists(attack_path):
    #     os.mkdir(attack_path)
    #
    # thresholds = torch.tensor(threshold_list).to(hps.device)

    # print('all thresholds ', threshold_list)
    for batch_id, (x, y) in enumerate(test_loader):
        # Note that images are scaled to [0., 1.0]
        x, y = x.to(hps.device), y.to(hps.device)
        with torch.no_grad():
            output = model(x)

        save_image(x, 'original.png')

        pred = output.argmax(dim=1)
        print('real label: {}, pred: {}'.format(y.item(), pred.item()))

        y = y - 7
        #print('target label: {}'.format(y.item()))
        adversary.threshold = 600   #  threshold_list[y.item()]

        adv_x = adversary.perturb(x, y)
        with torch.no_grad():
            output = model(adv_x)

        save_image(adv_x, 'threshold_cw.png')

        print('outputs ', output)
        values, pred = output.max(dim=1)
        print('adv pred: {}, log_like: {:.4f}'.format(pred.item(), values.item()))
        break


if __name__ == '__main__':
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


    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--targeted_cw", action="store_true",
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
    parser.add_argument("--cw_confidence", type=float, default=0,
                        help="confidence for CW attack.")

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

    adversary = CarliniWagnerL2Attack(model, confidence=hps.cw_confidence,
                                      c=1, num_classes=10, clip_min=0., clip_max=1., max_iterations=5000, targeted=True)

    if hps.targeted_cw:
        targeted_cw(model, adversary, hps)
    else:
        attack_run_rejection_policy(model, adversary, hps)


