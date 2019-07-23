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
                       os.path.join(hps.log_dir, '{}_{}.pth'.format(model.encoder_name, hps.problem)))

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

    checkpoint_path = os.path.join(hps.log_dir, '{}_{}.pth'.format(model.encoder_name, hps.problem))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    dataset = get_dataset(dataset=hps.problem, train=True)
    # test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=True)

    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)

        preds = model.inference(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        acc_list.append(acc.item())

    print('Train accuracy: {:.4f}'.format(np.mean(acc_list)))

    dataset = get_dataset(dataset=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    acc_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)

        preds = model.inference(x).argmax(dim=1)
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
            output = model.inference(x, log_softmax=True)
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
            output = model.inference(perturbed_data, log_softmax=True)

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
    checkpoint_path = os.path.join(hps.log_dir, '{}_{}.pth'.format(model.encoder_name, hps.problem))
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, hps, eps)
        accuracies.append(acc)
        examples.append(ex)

    fgsm_checkpoint = dict(zip(epsilons, accuracies))
    checkpoint_path = os.path.join(hps.log_dir, '{}_{}_fgsm.pth'.format(model.encoder_name, hps.problem))
    torch.save(fgsm_checkpoint, checkpoint_path)


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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Ablation
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    hps.device = torch.device("cuda" if use_cuda else "cpu")

    if hps.problem=='cifar10':
        hps.encoder_name = 'resnet51'
        hps.image_channel=3
    elif hps.problem=='mnist':
        hps.encoder_name = 'resnet9'
        hps.rep_size = 16
        hps.image_channel = 1

    model = build_resnet_32x32(n=9, fc_size=hps.n_classes, image_channel=hps.image_channel)
    hps.encoder_name = 'resnet9'

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