import argparse
import sys
import os

import numpy as np

# from models.dim import DIM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from tensorboardX import SummaryWriter

from sdim import SDIM
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

    writer = SummaryWriter()
    global_step = 0
    min_loss = 1e3
    for epoch in range(1, hps.epochs+1):
        model.train()
        loss_list = []
        mi_list = []
        nll_list = []
        margin_list = []

        for batch_id, (x, y) in enumerate(train_loader):
            global_step += 1
            x = x.to(hps.device)
            y = y.to(hps.device)

            optimizer.zero_grad()

            loss, mi_loss, nll_loss, ll_margin = model.eval_losses(x, y)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), global_step)
            writer.add_scalar('mi', mi_loss.item(), global_step)
            writer.add_scalar('nll', nll_loss.item(), global_step)
            writer.add_scalar('margin', ll_margin.item(), global_step)

            loss_list.append(loss.item())
            mi_list.append(mi_loss.item())
            nll_list.append(nll_loss.item())
            margin_list.append(ll_margin.item())

        print('===> Epoch: {}'.format(epoch + 1))
        print('loss: {:.4f}, mi: {:.4f}, nll: {:.4f}, ll_margin: {:.4f}'.format(
            np.mean(loss_list),
            np.mean(mi_list),
            np.mean(nll_list),
            np.mean(margin_list)
        ))
        if np.mean(loss_list) < min_loss:
            min_loss = np.mean(loss_list)
            checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}.pth'.format(model.encoder_name,
                                                                                    hps.problem,
                                                                                    hps.rep_size))
            torch.save(model.state_dict(), checkpoint_path)

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

    checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}.pth'.format(model.encoder_name,
                                                                            hps.problem,
                                                                            hps.rep_size))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    dataset = get_dataset(data_name=hps.problem, train=True)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=True)

    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)

        preds = model(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        acc_list.append(acc.item())

    print('Train accuracy: {:.4f}'.format(np.mean(acc_list)))

    global_acc_list = []
    for label_id in range(hps.n_classes):
        dataset = get_dataset(data_name=hps.problem, train=False, label_id=label_id)
        test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

        acc_list = []
        for batch_id, (x, y) in enumerate(test_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)

            preds = model(x).argmax(dim=1)
            acc = (preds == y).float().mean()
            acc_list.append(acc.item())

        acc = np.mean(acc_list)
        global_acc_list.append(acc)
        print('Class label {}, Test accuracy: {:.4f}'.format(label_id, acc))
    print('Test accracy: {:.4f}'.format(np.mean(global_acc_list)))


def inference_rejection(model, hps):
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}.pth'.format(model.encoder_name,
                                                                            hps.problem,
                                                                            hps.rep_size))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
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

        thresh_idx = int(hps.percentile * len(in_ll_list))
        thresh = sorted(in_ll_list)[thresh_idx]
        print('threshold_idx/total_size: {}/{}, threshold: {:.3f}'.format(thresh_idx, len(in_ll_list), thresh))
        threshold_list.append(thresh)  # class mean as threshold

    # Evaluation
    dataset = get_dataset(data_name=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    n_correct = 0
    n_false = 0
    n_reject = 0

    thresholds = torch.tensor(threshold_list).to(hps.device)
    result_str = ' & '.join('{:.1f}'.format(ll) for ll in threshold_list)
    print('thresholds: ', result_str)

    for batch_id, (x, target) in enumerate(test_loader):
        # Note that images are scaled to [-1.0, 1.0]
        x, target = x.to(hps.device), target.to(hps.device)

        with torch.no_grad():
            log_lik = model(x)

        values, pred = log_lik.max(dim=1)
        confidence_idx = values >= thresholds[pred]  # the predictions you have confidence in.
        reject_idx = values < thresholds[pred]       # the ones rejected.

        n_correct += pred[confidence_idx].eq(target[confidence_idx]).sum().item()
        n_false += (pred[confidence_idx] != target[confidence_idx]).sum().item()
        n_reject += reject_idx.float().sum().item()

    n = len(test_loader.dataset)
    acc = n_correct / n
    false_rate = n_false / n
    reject_rate = n_reject / n

    acc_remain = acc / (acc + false_rate)

    print('Test set:\n acc: {:.4f}, false rate: {:.4f}, reject rate: {:.4f}'.format(acc, false_rate, reject_rate))
    print('acc on remain set: {:.4f}'.format(acc_remain))
    return acc, reject_rate, acc_remain


def noise_attack(model, hps):
    model.eval()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}.pth'.format(model.encoder_name,
                                                                            hps.problem,
                                                                            hps.rep_size))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    dataset = get_dataset(data_name=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    epsilon = 1e-2
    for batch_id, (x, y) in enumerate(test_loader):
        if batch_id == 10:
           break 
        print('Example ', batch_id + 1)
        x = x.to(hps.device)
        #print('x bound ', x.min(), x.max())
        #noise = torch.randn(x.size()).to(hps.device) * epsilon
        y = y.to(hps.device)
        ll = model(x)
        print('Label: {}, predict: {}, ll list: {}'.format(y.item(), ll.argmax().item(), ll.cpu().detach().numpy()))
    
    x = torch.zeros(x.size()).to(hps.device) 
    for eps in range(10 + 1):
        ll = model(x + eps/10)
        print('x full of {:.3f}, predict: {}, ll list: {}'.format(eps, ll.argmax().item(), ll.cpu().detach().numpy()))


def ood_inference(model, hps):
    model.eval()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}.pth'.format(model.encoder_name,
                                                                            hps.problem,
                                                                            hps.rep_size))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    if hps.problem == 'fashion':
        out_problem = 'mnist'
    elif hps.problem == 'cifar10':
        out_problem = 'svhn'

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

        thresh_idx = int(hps.percentile * len(in_ll_list))
        thresh = sorted(in_ll_list)[thresh_idx]
        print('threshold_idx/total_size: {}/{}, threshold: {:.3f}'.format(thresh_idx, len(in_ll_list), thresh))
        threshold_list.append(thresh)  # class mean as threshold

    print('Inference on {}'.format(out_problem))
    # eval on whole test set
    dataset = get_dataset(data_name=out_problem, train=False)
    out_test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    reject_acc_dict = dict([(str(label_id), [])for label_id in range(hps.n_classes)])

    for batch_id, (x, _) in enumerate(out_test_loader):
        x = x.to(hps.device)
        ll = model(x)
        for label_id in range(hps.n_classes):
            # samples whose ll lower than threshold will be successfully rejected.
            acc = (ll[:, label_id] < threshold_list[label_id]).float().mean().item()
            reject_acc_dict[str(label_id)].append(acc)

    print('==================== OOD Summary ====================')
    print('In-distribution dataset: {}, Out-distribution dataset: {}'.format(hps.problem, out_problem))
    rate_list = []
    for label_id in range(hps.n_classes):
        acc = np.mean(reject_acc_dict[str(label_id)])
        rate_list.append(acc)
        print('Label id: {}, reject success rate: {:.4f}'.format(label_id, acc))

    print('Mean reject success rate: {:.4f}'.format(np.mean(rate_list)))
    print('=====================================================')
    # ll_checkpoint = {'fashion': in_ll_list, 'mnist': out_ll_list}
    # torch.save(ll_checkpoint, 'ood_sdim_{}_{}_d{}.pth'.format(model.encoder_name, hps.problem, hps.rep_size))


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
            ll = model(x)

            correct_idx = ll.argmax(dim=1) == y

            ll_, y_ = ll[correct_idx], y[correct_idx]  # choose samples are classified correctly
            in_ll_list += list(ll_[:, label_id].detach().cpu().numpy())

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
        ll = model(noises)

        for label_id in range(hps.n_classes):
            # samples whose ll lower than threshold will be successfully rejected.
            acc = (ll[:, label_id] < threshold_list[label_id]).float().mean().item()
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
    parser.add_argument("--rejection_inference", action="store_true",
                        help="Used in inference mode with rejection")
    parser.add_argument("--ood_inference", action="store_true",
                        help="Used in ood inference mode")
    parser.add_argument("--noise_ood_inference", action="store_true",
                        help="Used in noise ood inference mode")
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

    # Inference hyperparams:
    parser.add_argument("--percentile", type=float, default=0.01,
                        help="percentile value for inference with rejection.")

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
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
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

    model = SDIM(rep_size=hps.rep_size,
                 mi_units=hps.mi_units,
                 encoder_name=hps.encoder_name,
                 image_channel=hps.image_channel
                 ).to(hps.device)
    optimizer = Adam(model.parameters(), lr=hps.lr)

    print('==>  # Model parameters: {}.'.format(cal_parameters(model)))

    if hps.noise_attack:
        noise_attack(model, hps)
    elif hps.inference:
        inference(model, hps)
    elif hps.ood_inference:
        ood_inference(model, hps)
    elif hps.rejection_inference:
        inference_rejection(model, hps)
    elif hps.noise_ood_inference:
        noise_ood_inference(model, hps)
    else:
        train(model, optimizer, hps)
