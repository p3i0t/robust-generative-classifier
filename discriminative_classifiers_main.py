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

from utils import get_dataset, cal_parameters


def train(model, optimizer, hps):
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    # Create log dir
    logdir = os.path.abspath(hps.log_dir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    dataset = get_dataset(data_name=hps.problem, train=True)
    train_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_train, shuffle=True)

    dataset = get_dataset(data_name=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    min_loss = 1e3

    for epoch in range(1, hps.epochs + 1):
        model.train()
        loss_list = []
        acc_list = []

        for batch_id, (x, y) in enumerate(train_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.nll_loss(F.log_softmax(logits, dim=1), y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            acc = (logits.argmax(dim=1) == y).float().mean()
            acc_list.append(acc.item())

        print('===> Epoch: {}'.format(epoch))
        print('loss: {:.4f}, train accuracy: {:.4f}'.format(np.mean(loss_list), np.mean(acc_list)))
        if np.mean(loss_list) < min_loss:
            min_loss = np.mean(loss_list)
            torch.save(model.state_dict(),
                       os.path.join(hps.log_dir, '{}_{}.pth'.format(hps.encoder_name, hps.problem)))

        model.eval()
        # Evaluate accuracy on test set.
        if epoch > 10:
            acc_list = []
            for batch_id, (x, y) in enumerate(test_loader):
                x = x.to(hps.device)
                y = y.to(hps.device)

                preds = model(x).argmax(dim=1)
                acc = (preds == y).float().mean()
                acc_list.append(acc.item())
            print('Test accuracy: {:.3f}'.format(np.mean(acc_list)))


def inference(model, hps):
    model.eval()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    checkpoint_path = os.path.join(hps.log_dir, '{}_{}.pth'.format(hps.encoder_name, hps.problem))
                                                                            
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    dataset = get_dataset(data_name=hps.problem, train=True)
    # test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=True)

    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)

        preds = model(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        acc_list.append(acc.item())

    print('Train accuracy: {:.4f}'.format(np.mean(acc_list)))

    dataset = get_dataset(data_name=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)

        preds = model(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        acc_list.append(acc.item())

    print('Test accuracy: {:.4f}'.format(np.mean(acc_list)))


def noise_ood_inference(model, hps):
    model.eval()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}.pth'.format(model.encoder_name,
                                                                            hps.problem,
                                                                            hps.rep_size))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

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
            outs = model(x)
            if hps.use_prob:
                outs = F.softmax(outs, dim=-1)

            correct_idx = outs.argmax(dim=1) == y

            outs_, y_ = outs[correct_idx], y[correct_idx]  # choose samples are classified correctly
            in_ll_list += list(outs_[:, label_id].detach().cpu().numpy())

        thresh_idx = int(hps.percentile * len(in_ll_list))
        thresh = sorted(in_ll_list)[thresh_idx]
        print('threshold_idx/total_size: {}/{}, threshold: {:.3f}'.format(thresh_idx, len(in_ll_list), thresh))
        threshold_list.append(thresh)  # class mean as threshold

    shape = x.size()

    batch_size = 100
    n_batches = 100

    reject_acc_dict = dict([(str(label_id), []) for label_id in range(hps.n_classes)])
    # Noise as out-distribution samples
    for batch_id in range(n_batches):
        noises = torch.randn((batch_size, shape[1], shape[2], shape[3])).uniform_(0., 1.).to(hps.device) # sample noise
        outs = model(noises)
        if hps.use_prob:
            outs = F.softmax(outs, dim=-1)

        for label_id in range(hps.n_classes):
            # samples whose ll lower than threshold will be successfully rejected.
            acc = (outs[:, label_id] < threshold_list[label_id]).float().mean().item()
            reject_acc_dict[str(label_id)].append(acc)

    print('==================== Noise OOD Summary ====================')
    print('In-distribution dataset: {}, Out-distribution dataset: Noise ~ Uniform[0, 1]'.format(hps.problem))
    rate_list = []
    for label_id in range(hps.n_classes):
        acc = np.mean(reject_acc_dict[str(label_id)])
        rate_list.append(acc)
        print('Label id: {}, reject success rate: {:.4f}'.format(label_id, acc))

    print('Mean reject success rate: {:.4f}'.format(np.mean(rate_list)))
    print('===========================================================')

    reject_acc_dict = dict([(str(label_id), []) for label_id in range(hps.n_classes)])
    # Noise as out-distribution samples
    for batch_id in range(n_batches):
        noises = 0.5 + torch.randn((batch_size, shape[1], shape[2], shape[3])).clamp_(min=-0.5, max=0.5).to(hps.device)  # sample noise
        ll = model(noises)

        for label_id in range(hps.n_classes):
            # samples whose ll lower than threshold will be successfully rejected.
            acc = (ll[:, label_id] < threshold_list[label_id]).float().mean().item()
            reject_acc_dict[str(label_id)].append(acc)

    print('==================== Noise OOD Summary ====================')
    print('In-distribution dataset: {}, Out-distribution dataset: Noise ~ Normal(0.5, 1) clamped to [0, 1]'.format(hps.problem))
    rate_list = []
    for label_id in range(hps.n_classes):
        acc = np.mean(reject_acc_dict[str(label_id)])
        rate_list.append(acc)
        print('Label id: {}, reject success rate: {:.4f}'.format(label_id, acc))

    print('Mean reject success rate: {:.4f}'.format(np.mean(rate_list)))
    print('===========================================================')


if __name__ == "__main__":
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--inference", action="store_true",
                        help="Used in inference mode")
    parser.add_argument("--noise_ood", action="store_true",
                        help="Perform noise as OoD detection")
    parser.add_argument("--use_prob", action="store_true",
                        help="Perform noise as OoD detection")
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
    elif hps.problem == 'svhn':
        hps.image_channel = 3
    elif hps.problem == 'mnist':
        hps.image_channel = 1
    elif hps.problem == 'fashion':
        hps.image_channel = 1

    n_encoder_layers = int(hps.encoder_name.strip('resnet'))
    model = build_resnet_32x32(n=n_encoder_layers,
                               fc_size=hps.n_classes,
                               image_channel=hps.image_channel
                               ).to(hps.device)

    optimizer = Adam(model.parameters(), lr=hps.lr)

    print('==>  # Model parameters: {}.'.format(cal_parameters(model)))

    if hps.fgsm_attack:
        fgsm_evaluation(model, hps)
    # elif hps.noise_attack:
    #     noise_attack(model, hps)
    elif hps.inference:
        inference(model, hps)
    else:
        train(model, optimizer, hps)
