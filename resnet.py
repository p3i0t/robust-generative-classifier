'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# __all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
__all__ = ['build_resnet_32x32']


def _weights_init(m):
    # classname = m.__class__.__name__
    # # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channel//4, out_channel//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channel, self.expansion * out_channel, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * out_channel)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channel=1):
        super(ResNet, self).__init__()
        self.in_channel = 32

        multiplier = self.in_channel

        self.conv1 = nn.Conv2d(image_channel, multiplier, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(multiplier)

        # 4 stages resnet
        self.layer1 = self._make_layer(block, multiplier, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, multiplier * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, multiplier * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, multiplier * 8, num_blocks[3], stride=2)
        self.linear = nn.Sequential(nn.Linear(multiplier * 8, multiplier * 8),
                                    nn.BatchNorm1d(multiplier * 8),
                                    nn.ReLU(),
                                    nn.Linear(multiplier * 8, num_classes))

        self.apply(_weights_init)

    def _make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, return_full_list=False):
        out_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)

        out = self.layer1(out)
        out_list.append(out)

        out = self.layer2(out)
        out_list.append(out)

        out = self.layer3(out)
        out_list.append(out)

        out = self.layer4(out)
        out_list.append(out)

        out = F.avg_pool2d(out, out.size()[3])
        out_list.append(out)

        out = out.view(out.size(0), -1)
        out_list.append(out)

        out = self.linear(out)
        out_list.append(out)

        if return_full_list:
            return out_list
        else:
            return out_list[-1]


def build_resnet_32x32(n=41, fc_size=10, image_channel=3):
    assert (n - 1) % 8 == 0, '{} should be expressed in form of 6n+1'.format(n)
    block_depth = int((n - 1) / 8)
    return ResNet(BasicBlock, [block_depth]*4, num_classes=fc_size, image_channel=image_channel)

#
# def resnet20():
#     return ResNet(BasicBlock, [3, 3, 3])
#
#
# def resnet32():
#     return ResNet(BasicBlock, [5, 5, 5])
#
#
# def resnet44():
#     return ResNet(BasicBlock, [7, 7, 7])
#
#
# def resnet56():
#     return ResNet(BasicBlock, [9, 9, 9])
#
#
# def resnet110():
#     return ResNet(BasicBlock, [18, 18, 18])
#
#
# def resnet1202():
#     return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    # for net_name in __all__:
    #     if net_name.startswith('resnet'):
    #         #print(net_name)
    #         test(globals()[net_name]())
    #         print()
    import torch
    x = torch.randn(5, 3, 32, 32)
    model = build_resnet_32x32(19, fc_size=10)
    o_list = model(x, return_full_list=True)
    for o in o_list:
        print('size: ', o.size())
