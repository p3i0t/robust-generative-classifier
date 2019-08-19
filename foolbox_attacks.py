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

import foolbox

from utils import get_dataset, cal_parameters


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
    parser.add_argument("--attack", type=str, default='deepfool',
                        help="attack type")

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
    #
    # if hps.attack == 'pgdinf':
    #     linfPGD_attack(model, hps)
    # elif hps.attack == 'pgd2':
    #     l2PGD_attack(model, hps)
    # elif hps.attack == 'cw':
    #     cw_l2_attack(model, hps)
    # elif hps.attack == 'fgsm':
    #     fgsm_attack(model, hps)

    model.eval()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1.), num_classes=10)

    dataset = get_dataset(data_name=hps.problem, train=False, label_id=0)
    # hps.n_batch_test = 1
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    for batch_id, (x, y) in enumerate(test_loader):
        # Note that images are scaled to [0., 1.0]
        x, y = x.to(hps.device), y.to(hps.device)

        if hps.attack == 'deepfool':
            attack = foolbox.attacks.DeepFoolL2Attack(fmodel)
        elif hps.attack == 'cw':
            attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)
        elif hps.attack == 'boundary':
            attack = foolbox.attacks.BoundaryAttack(fmodel)
        elif hps.attack == 'jsma':
            attack = foolbox.attacks.SaliencyMapAttack(fmodel)
        else:
            raise ValueError('param attack {} not available.'.format(hps.attack))

        img, label = x[0], y[0]

        adversarial = attack(img.cpu().numpy(), label.cpu().numpy())
        #adversarial = attack(img.cpu().numpy(), label.cpu().numpy(), confidence=500, max_iterations=1000)

        ll = model(img.unsqueeze(dim=0).to(hps.device))

        result_str = ' & '.join('{:.1f}'.format(ll) for ll in ll[0].tolist())
        print('original log_likes: ', result_str)

        path = os.path.join(hps.attack_dir, '{}_{}_original.png'.format(hps.problem, hps.attack))
        save_image(img, path)

        adv = torch.tensor(adversarial)
        ll = model(adv.unsqueeze(dim=0).to(hps.device))

        result_str = ' & '.join('{:.1f}'.format(ll) for ll in ll[0].tolist())
        print('adv log_likes: ', result_str)

        path = os.path.join(hps.attack_dir, '{}_{}_adv.png'.format(hps.problem, hps.attack))
        save_image(adv, path)

        classification_label = int(np.argmax(fmodel.predictions(img.cpu().numpy())))
        adversarial_label = int(np.argmax(fmodel.predictions(adversarial)))

        print("source label: " + str(int(label)) + ", adversarial_label: " + str(
            adversarial_label) + ", classification_label: " + str(classification_label))
        break

