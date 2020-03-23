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


class ThresholdCarliniWagnerL2Attack(Attack, LabelMixin):
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

    def __init__(self, predict, num_classes, threshold, confidence=0,
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

        super(ThresholdCarliniWagnerL2Attack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.threshold = threshold
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
            loss1 = clamp(other - real + self.confidence, min=0.)
            threshold_loss = clamp(self.threshold - real, min=0.)
            #threshold_loss = clamp(real - self.threshold, min=0.)
        else:
            loss1 = clamp(real - other + self.confidence, min=0.)
            threshold_loss = clamp(self.threshold - other, min=0.)
        loss2 = (l2distsq).sum()
        # const = 0.001
        loss1 = torch.sum(const * loss1)
        threshold_loss = torch.sum(const * threshold_loss)
        # print('const ', const)
        print('dis: {:.2f}, loss1: {:.2f}, threshold_loss: {:.2f}'.format(loss2.item(), loss1.item(), threshold_loss.item()))
        loss = loss2 + threshold_loss
        #loss = loss1 + loss2 + threshold_loss
        return loss

    def _is_successful(self, output, label, is_logits):
        # determine success, see if confidence-adjusted logits give the right
        #   label

        if is_logits:
            output = output.detach().clone()
            if self.targeted:
                output[torch.arange(len(label)).long(),
                       label] -= self.confidence
            else:
                output[torch.arange(len(label)).long(),
                       label] += self.confidence
            pred = torch.argmax(output, dim=1)
        else:
            pred = output
            if pred == INVALID_LABEL:
                return pred.new_zeros(pred.shape).byte()

        return is_successful(pred, label, self.targeted)

    def _forward_and_update_delta(
            self, optimizer, x_atanh, delta, y_onehot, loss_coeffs):

        optimizer.zero_grad()
        adv = tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)
        transimgs_rescale = tanh_rescale(x_atanh, self.clip_min, self.clip_max)
        output = self.predict(adv)
        l2distsq = calc_l2distsq(adv, transimgs_rescale)
        loss = self._loss_fn(output, y_onehot, l2distsq, loss_coeffs)
        loss.backward()
        optimizer.step()

        return loss.item(), l2distsq.data, output.data, adv.data

    def _get_arctanh_x(self, x):
        result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
                       min=self.clip_min, max=self.clip_max) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)

    def _update_if_smaller_dist_succeed(
            self, adv_img, labs, output, l2distsq, batch_size,
            cur_l2distsqs, cur_labels,
            final_l2distsqs, final_labels, final_advs):

        target_label = labs
        output_logits = output
        _, output_label = torch.max(output_logits, 1)

        mask = (l2distsq < cur_l2distsqs) & self._is_successful(
            output_logits, target_label, True)

        cur_l2distsqs[mask] = l2distsq[mask]  # redundant
        cur_labels[mask] = output_label[mask]

        mask = (l2distsq < final_l2distsqs) & self._is_successful(
            output_logits, target_label, True)
        final_l2distsqs[mask] = l2distsq[mask]
        final_labels[mask] = output_label[mask]
        final_advs[mask] = adv_img[mask]

    def _update_loss_coeffs(
            self, labs, cur_labels, batch_size, loss_coeffs,
            coeff_upper_bound, coeff_lower_bound):

        # TODO: remove for loop, not significant, since only called during each
        # binary search step
        for ii in range(batch_size):
            cur_labels[ii] = int(cur_labels[ii])
            if self._is_successful(cur_labels[ii], labs[ii], False):
                coeff_upper_bound[ii] = min(
                    coeff_upper_bound[ii], loss_coeffs[ii])

                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (
                        coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
            else:
                coeff_lower_bound[ii] = max(
                    coeff_lower_bound[ii], loss_coeffs[ii])
                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (
                        coeff_lower_bound[ii] + coeff_upper_bound[ii]) / 2
                else:
                    loss_coeffs[ii] *= 10

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)
        x = replicate_input(x)
        batch_size = len(x)
        coeff_lower_bound = x.new_zeros(batch_size)
        coeff_upper_bound = x.new_ones(batch_size) * CARLINI_COEFF_UPPER
        loss_coeffs = torch.ones_like(y).float() * self.initial_const
        final_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = x
        x_atanh = self._get_arctanh_x(x)
        y_onehot = to_one_hot(y, self.num_classes).float()

        final_l2distsqs = torch.FloatTensor(final_l2distsqs).to(x.device)
        final_labels = torch.LongTensor(final_labels).to(x.device)

        # Start binary search
        for outer_step in range(self.binary_search_steps):
            delta = nn.Parameter(torch.zeros_like(x))
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            cur_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size
            cur_l2distsqs = torch.FloatTensor(cur_l2distsqs).to(x.device)
            cur_labels = torch.LongTensor(cur_labels).to(x.device)
            prevloss = PREV_LOSS_INIT

            if (self.repeat and outer_step) == self.binary_search_steps - 1:
                loss_coeffs = coeff_upper_bound
            for ii in range(self.max_iterations):
                loss, l2distsq, output, adv_img = \
                    self._forward_and_update_delta(
                        optimizer, x_atanh, delta, y_onehot, loss_coeffs)
                if self.abort_early:
                    if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                        if loss > prevloss * ONE_MINUS_EPS:
                            break
                        prevloss = loss

                self._update_if_smaller_dist_succeed(
                    adv_img, y, output, l2distsq, batch_size,
                    cur_l2distsqs, cur_labels,
                    final_l2distsqs, final_labels, final_advs)

            self._update_loss_coeffs(
                y, cur_labels, batch_size,
                loss_coeffs, coeff_upper_bound, coeff_lower_bound)

        return final_advs


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
        # correct_idx = pred == y
        # x, y = x[correct_idx], y[correct_idx]  # Only evaluate on the correct classified samples by clean classifier.
        # n_correct += correct_idx.sum().item()

        # print('threshold: {:.4f}'.format(threshold_list[y.item()]))

        y = y - 1
        #print('target label: {}'.format(y.item()))
        adversary.threshold = 500 #  threshold_list[y.item()]

        adv_x = adversary.perturb(x, y)
        with torch.no_grad():
            output = model(adv_x)

        save_image(adv_x, 'threshold_cw.png')

        print('outputs ', output)
        values, pred = output.max(dim=1)
        print('adv pred: {}, log_like: {:.4f}'.format(pred.item(), values.item()))
        break
        # successful_idx = pred != y   # idx of successful adversarial examples.
        # values, pred = output[successful_idx].max(dim=1)
        # confidence_idx = values >= thresholds[pred]
        # reject_idx = values < thresholds[pred]  # idx of successfully rejected samples.

        # adv_correct += pred[confidence_idx].eq(y[confidence_idx]).sum().item()
        # n_successful_adv += successful_idx.float().sum().item()
        # n_rejected_adv += reject_idx.float().sum().item()
        #
        # if batch_id % 10 == 0:
        #     print('Evaluating on {}-th batch ...'.format(batch_id + 1))

    # n = len(test_loader.dataset)
    # reject_rate = n_rejected_adv / n_successful_adv
    # success_adv_rate = n_successful_adv / n_correct
    # print('Test set, clean classification accuracy: {}/{}={:.4f}'.format(n_correct, n, n_correct / n))
    # print('success rate of adv examples generation: {}/{}={:.4f}'.format(n_successful_adv, n_correct, success_adv_rate))
    # print('reject success rate: {}/{}={:.4f}'.format(n_rejected_adv, n_successful_adv, reject_rate))


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

    adversary = ThresholdCarliniWagnerL2Attack(model, threshold=-1,
                                               confidence=0,
                                                num_classes=10,
                                                clip_min=0.,
                                                clip_max=1.,
                                                max_iterations=5000,
                                                targeted=True
                                                )

    attack_run_rejection_policy(model, adversary, hps)

