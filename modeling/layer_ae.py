import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



def squash_v1(x, axis):
    s_squared_norm = (x ** 2).sum(axis, keepdim=True)
    scale = torch.sqrt(s_squared_norm)/ (1. + s_squared_norm)
    return scale * x

def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.Conv1d(in_channels, out_channels * num_capsules, kernel_size, stride)
        torch.nn.init.xavier_normal_(self.capsules.weight)
        # torch.nn.init.kaiming_normal_(self.capsules.weight, mode='fan_in', nonlinearity='relu')
        self.out_channels = out_channels
        self.num_capsules = num_capsules

    def forward(self, x):
        batch_size = x.size(0) * x.size(1)
        x = x.view(batch_size, 1, -1)
        u = self.capsules(x).view(batch_size, self.num_capsules, self.out_channels, -1, 1)
        # u = self.capsules(x).view(batch_size, -1, self.num_capsules)
        x = squash_v1(u, axis=2)
        return x

class FlattenCaps(nn.Module):
    def __init__(self):
        super(FlattenCaps, self).__init__()
    def forward(self, x):
        x = x.view(x.size(0), x.size(2) * x.size(3) * x.size(4), -1)
        return x


class FCCaps(nn.Module):
    def __init__(self, args, output_capsule_num, input_capsule_num, in_channels, out_channels):
        super(FCCaps, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_capsule_num = input_capsule_num
        self.output_capsule_num = output_capsule_num
        self.W1 = nn.Parameter(torch.FloatTensor(input_capsule_num, in_channels, output_capsule_num * out_channels))
        torch.nn.init.xavier_normal_(self.W1)
        self.bij = nn.Parameter(torch.zeros((self.input_capsule_num, self.output_capsule_num)))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2) # [16, 512, 1, 8]
        u_hat = x.matmul(self.W1) # self.W1: [512, 8, 24], u_hat: [16, 512, 1, 24]
        u_hat = u_hat.reshape(u_hat.size(0), self.input_capsule_num, self.output_capsule_num, self.out_channels)
        # [16, 512, 3, 8]
        ci = F.softmax(self.bij, dim=1) # [512, 3]
        sj = (ci.unsqueeze(-1) * u_hat).sum(dim=1) # [16, 3, 8]
        v = squash_v1(sj, axis=-1) # [16, 3, 8]
        num_iterations = 7
        bij = self.bij.expand((batch_size, self.input_capsule_num, self.output_capsule_num)) # [16, 512, 3]
        for i in range(num_iterations):
            v = v.unsqueeze(1) # [16, 1, 3, 8]
            bij = bij + (u_hat * v).sum(-1) # [16, 512, 3]
            ci = F.softmax(bij.view(-1, self.output_capsule_num), dim=1) # [8192, 3]
            ci = ci.view(-1, self.input_capsule_num, self.output_capsule_num, 1) # [16, 512, 3, 1]
            sj = (ci * u_hat).sum(dim=1) # [16, 3, 8]
            v = squash_v1(sj, axis=-1) # [16, 3, 8]
        x = torch.sqrt((v ** 2).sum(2)) # [16, 3]
        return x
