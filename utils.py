from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataset(dataset='mnist', data_dir='data', train=True, label_id=None):
    """
    Get a dataset.
    :param dataset: str, name of dataset.
    :param data_dir: str, base directory of data.
    :param train: bool, return train set if True, or test set if False.
    :param label_id: None or int, return data with particular label_id.
    :return: pytorch dataset.
    """
    if dataset == 'mnist':
        if train:
            transform = transforms.Compose([
                                            transforms.Resize((32, 32)),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))  # 1-channel, scale to [-1, 1]
                                        ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5, ))
            ])

        dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)

    elif dataset == 'fashion':
        if train:
            transform = transforms.Compose([
                                            transforms.Resize((32, 32)),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))
                                        ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        dataset = datasets.FashionMNIST(data_dir, train=train, download=True, transform=transform)

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

        dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    else:
        print('dataset {} is not available'.format(dataset))

    if label_id:
        # select samples with particular label
        idx = dataset.targets == label_id
        print('Select samples with label: {}, # samples: {}'.format(label_id, idx.float().sum().item()))
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
    return dataset


def cal_parameters(model):
    """
    Calculate the parameters of a Pytorch model.
    :param model: torch.nn.Module
    :return: number of parameters.
    """
    cnt = 0
    for para in model.parameters():
        cnt += para.numel()
    return cnt


if __name__ == '__main__':
    dataset = get_dataset(label_id=1)
    train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    for batch_id, (x, y) in enumerate(train_loader):
        print(y)
        break
