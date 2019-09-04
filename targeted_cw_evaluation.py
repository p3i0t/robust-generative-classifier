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

from advertorch.attacks import CarliniWagnerL2Attack


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

        largest_logits = []
        pred_list = []
        for i in range(hps.n_classes):
            if i != y:
                y_cur = torch.LongTensor([i]).to(hps.device)
                adv_x = adversary.perturb(x, y_cur)
            else:
                y_cur = y
                adv_x = x

            with torch.no_grad():
                output = model(adv_x)

            path = os.path.join(hps.attack_dir, 'targeted_cw_{}_{}_{}.png'.format(
                hps.problem, hps.cw_confidence, y_cur.cpu().item()))
            save_image(adv_x, path)

            ll, pred = output.max(dim=-1)
            largest_logits.append(ll.cpu().item())
            pred_list.append(pred.cpu().item())

            out_str = ' '.join('{:.1f}'.format(logit) for logit in output[0].tolist())
            print('target: {}, logits: {}'.format(y_cur.cpu().item(), out_str))

        out_str = ' & '.join('{:.1f}'.format(logit) for logit in largest_logits)
        print('confidence: {}, largest_logits: {}'.format(hps.cw_confidence, out_str))

        pred_str = ' & '.join('{}'.format(pred) for pred in pred_list)
        print('adv predictions: {}'.format(pred_str))

        break


if __name__ == '__main__':
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
    parser.add_argument("--cw_confidence", type=int, default=0,
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
                                      num_classes=10, clip_min=0., clip_max=1., max_iterations=10000, targeted=True)

    targeted_cw(model, adversary, hps)


