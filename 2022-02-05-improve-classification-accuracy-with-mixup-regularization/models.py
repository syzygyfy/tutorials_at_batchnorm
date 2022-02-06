'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

def to_one_hot(y,num_classes):
    y_onehot = torch.FloatTensor(y.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.unsqueeze(1).data.cpu(), 1)
    return y_onehot.cuda()

def get_lambda(alpha):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

# mixup process
def manifold_mixup(input,target,mixup_alpha,num_classes):

    # generate the mixup factor from beta distribution
    lam = get_lambda(mixup_alpha)
    lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()

    # generate the permutation of batch indices
    indices = np.random.permutation(input.size(0))

    # mixup the input/manifold
    input = input*lam + input[indices]*(1-lam)

    # mixup the labels
    target_reweighted = to_one_hot(target,num_classes=num_classes)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return input, target_reweighted

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def __forward_vanilla(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, None # returning None for compatibility with the __forward_mixup
    
    def __forward_mixup(self, x, mixup_alpha, labels):
        out = x
    
        layer_mix = random.randint(0,2)
        if layer_mix == 0:
            out, labels_weighted = manifold_mixup(out,labels,mixup_alpha,self.num_classes)

        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)

        if layer_mix == 1:
            out, labels_weighted = manifold_mixup(out,labels,mixup_alpha,self.num_classes)
        
        out = self.layer2(out)

        if layer_mix == 2:
            out, labels_weighted = manifold_mixup(out,labels,mixup_alpha,self.num_classes)

        out = self.layer3(out)

        if layer_mix == 3:
            out, labels_weighted = manifold_mixup(out,labels,mixup_alpha,self.num_classes)

        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, labels_weighted

    def forward(self, x, use_mixup=False, mixup_alpha=2.0, labels = None):
        if use_mixup:
            return self.__forward_mixup(x,mixup_alpha=mixup_alpha,labels=labels)
        else:
            return self.__forward_vanilla(x)


def ResNet18(num_classes=10,in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,in_channels=in_channels)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(num_classes=10,in_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_classes,in_channels=in_channels)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
