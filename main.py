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


def train(model, optimizer, hps):
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    # Create log dir
    logdir = os.path.abspath(hps.log_dir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    dataset = get_dataset(dataset=hps.problem, train=True)
    train_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_train, shuffle=True)
    
    dataset = get_dataset(dataset=hps.problem, train=False)
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

    dataset = get_dataset(dataset=hps.problem, train=True)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=True)

    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)

        preds = model(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        acc_list.append(acc.item())

    print('Train accuracy: {:.4f}'.format(np.mean(acc_list)))

    dataset = get_dataset(dataset=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)

        preds = model(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        acc_list.append(acc.item())

    print('Test accuracy: {:.4f}'.format(np.mean(acc_list)))


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def fgsm_evaluation(model, hps):
    # helper function for particular epsilon
    def test(model, hps, epsilon):
        model.eval()
        # Accuracy counter
        correct = 0
        adv_examples = []

        dataset = get_dataset(dataset=hps.problem, train=False)
        test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        # Loop over all examples in test set
        for x, y in test_loader:
            x = x.to(hps.device)
            y = y.to(hps.device)
            # Set requires_grad attribute of tensor. Important for Attack
            x.requires_grad = True

            # Forward pass the data through the model
            output = model(x, log_softmax=True)
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != y.item():
                continue

            # Calculate the loss
            loss = F.nll_loss(output, y)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = x.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(x, epsilon, data_grad)

            # Re-classify the perturbed image
            output = model(perturbed_data, log_softmax=True)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == y.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        # Calculate final accuracy for this epsilon
        final_acc = correct / float(len(test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples

    epsilons = [0., 0.05, .1, 0.15, .2, 0.25, .3]

    print("load pre-trained model")
    checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}.pth'.format(model.encoder_name,
                                                                            hps.problem,
                                                                            hps.rep_size))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, hps, eps)
        accuracies.append(acc)
        examples.append(ex)

    fgsm_checkpoint = dict(zip(epsilons, accuracies))
    checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}_fgsm.pth'.format(model.encoder_name,
                                                                                 hps.problem,
                                                                                 hps.rep_size))
    torch.save(fgsm_checkpoint, checkpoint_path)


def noise_attack(model, hps):
    model.eval()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}.pth'.format(model.encoder_name,
                                                                            hps.problem,
                                                                            hps.rep_size))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    dataset = get_dataset(dataset=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    epsilon = 1e-2
    for batch_id, (x, y) in enumerate(test_loader):
        if batch_id == 10:
            exit(0)
        print('Example ', batch_id + 1)
        x = x.to(hps.device)
        print('x bound ', x.min(), x.max())
        #noise = torch.randn(x.size()).to(hps.device) * epsilon
        y = y.to(hps.device)
        ll = model(x)
        print('Label: {}, predict: {}, ll list: {}'.format(y.item(), ll.argmax().item(), ll.cpu().detach().numpy()))


def ood_inference(model, hps):
    model.eval()
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    checkpoint_path = os.path.join(hps.log_dir, 'sdim_{}_{}_d{}.pth'.format(model.encoder_name,
                                                                            hps.problem,
                                                                            hps.rep_size))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    dataset = get_dataset(dataset=hps.problem, train=False)
    in_test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    print('Inference on {}'.format(hps.problem))
    in_ll_list = []
    for batch_id, (x, y) in enumerate(in_test_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)
        ll = model(x)

        one_hots = torch.eye(len(ll[0]))[y].to(x.device)
        ll = (ll * one_hots).sum(dim=1)  # (b, 1)
        in_ll_list += list(ll.detach().cpu().numpy())

    print('Log-likelihood, maximum {}, minimum {}'.format(max(in_ll_list), min(in_ll_list)))

    if hps.problem == 'fashion':
        out_problem = 'mnist'

    print('Inference on {}'.format(out_problem))
    dataset = get_dataset(dataset=out_problem, train=False)
    out_test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    out_ll_list = []
    for batch_id, (x, y) in enumerate(out_test_loader):
        x = x.to(hps.device)
        ll = model(x).max(dim=1)[0]  # get maximum

        out_ll_list += list(ll.detach().cpu().numpy())

    print('Log-likelihood, maximum {}, minimum {}'.format(max(out_ll_list), min(out_ll_list)))

    ll_checkpoint = {'fashion': in_ll_list, 'mnist': out_ll_list}
    torch.save(ll_checkpoint, 'ood_sdim_{}_{}_d{}.pth'.format(model.encoder_name, hps.problem, hps.rep_size))


if __name__ == "__main__":
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--inference", action="store_true",
                        help="Used in inference mode")
    parser.add_argument("--ood_inference", action="store_true",
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

    model = SDIM(rep_size=hps.rep_size,
                 mi_units=hps.mi_units,
                 encoder_name=hps.encoder_name,
                 image_channel=hps.image_channel
                 ).to(hps.device)
    optimizer = Adam(model.parameters(), lr=hps.lr)

    print('==>  # Model parameters: {}.'.format(cal_parameters(model)))
    if hps.fgsm_attack:
        fgsm_evaluation(model, hps)
    elif hps.noise_attack:
        noise_attack(model, hps)
    elif hps.inference:
        inference(model, hps)
    elif hps.ood_inference:
        ood_inference(model, hps)
    else:
        train(model, optimizer, hps)
