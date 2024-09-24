#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init


class ConvNet(nn.Module):
    ''' In this network the input image is supposed to be 28x28 '''

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes#128
        self.ndf = 64

        # Input 84x84x3
        self.conv1 = nn.Conv2d(3, self.ndf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ndf)

        # Input 42x42x64
        self.conv2 = nn.Conv2d(self.ndf, int(self.ndf*1.5), kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(int(self.ndf*1.5))

        # Input 20x20x96
        self.conv3 = nn.Conv2d(int(self.ndf*1.5), self.ndf*2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ndf*2)
        self.drop_3 = nn.Dropout2d(0.4)

        # Input 10x10x128
        self.conv4 = nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ndf*4)
        self.drop_4 = nn.Dropout2d(0.5)

        # Input 5x5x256
        self.fc = nn.Linear(self.ndf*4, self.num_classes, bias=True)
        # self.bn_fc = nn.BatchNorm1d(self.num_classes)

    def forward(self, input, return_feature=False):
        e1 = F.max_pool2d(self.bn1(self.conv1(input)), 2)
        x = F.leaky_relu(e1, 0.2, inplace=True)
        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.2, inplace=True)
        e3 = F.max_pool2d(self.bn3(self.conv3(x)), 2)
        x = F.leaky_relu(e3, 0.2, inplace=True)
        x = self.drop_3(x)
        e4 = F.max_pool2d(self.bn4(self.conv4(x)), 2)
        x = F.leaky_relu(e4, 0.2, inplace=True)
        x = self.drop_4(x)
        # print('x.shape', x.shape)
        # x = x.view(-1, self.ndf*4*5*5)
        out = x.view(x.size(0), -1)
        # output = self.bn_fc(self.fc1(out))
        output = self.fc(out)

        # return [e1, e2, e3, e4, None, output]
        if return_feature:
            return output, out
        else:
            return output

