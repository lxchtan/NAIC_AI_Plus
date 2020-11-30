#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
If you have any questions, please contact me with https://github.com/xufana7/AutoEncoder-with-pytorch
Author, Fan xu Aug 2020
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 0].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its four bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for four time.
        b, c = grad_output.shape
        grad_bit = grad_output.repeat(1, 1, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=True)

def conv2x2(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride,
                     padding=1, bias=True)

# create your own Encoder
class Encoder(nn.Module):
    B = 1

    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()
        self.conv1 = conv3x3(2, 2)
        self.conv2 = conv3x3(2, 2)
        self.fc = nn.Linear(1024, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.view(-1, 1024)
        out = self.fc(out)
        out = self.sig(out)
        self.out_ori = out
        out = self.quantize(out)
        return out


# create your own Decoder
class Decoder(nn.Module):
    B = 1

    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)

        self.conv2nums = 2
        self.conv3nums = 3
        self.conv4nums = 5
        self.conv5nums = 3

        self.multiConvs2 = nn.ModuleList()
        self.multiConvs3 = nn.ModuleList()
        self.multiConvs4 = nn.ModuleList()
        self.multiConvs5 = nn.ModuleList()

        self.fc = nn.Linear(int(feedback_bits / self.B), 1024)
        self.out_cov = conv3x3(2, 2)
        self.sig = nn.Sigmoid()

        self.multiConvs2.append(nn.Sequential(
                conv3x3(2, 64),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                conv3x3(64, 256),
                nn.BatchNorm2d(256),
                nn.ReLU()))
        self.multiConvs3.append(nn.Sequential(
                conv3x3(256, 512),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                conv3x3(512, 512),
                nn.BatchNorm2d(512),
                nn.ReLU()))
        self.multiConvs4.append(nn.Sequential(
                conv3x3(512, 1024),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                conv3x3(1024, 1024),
                nn.BatchNorm2d(1024),
                nn.ReLU()))
        self.multiConvs5.append(nn.Sequential(
                conv3x3(1024, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                conv3x3(128, 32),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                conv3x3(32, 2),
                nn.BatchNorm2d(2),
                nn.ReLU()))
                
        for _ in range(self.conv2nums):
            self.multiConvs2.append(nn.Sequential(
                conv3x3(256, 64),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                conv3x3(64, 64),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                conv3x3(64, 256),
                nn.BatchNorm2d(256),
                nn.ReLU()))
        for _ in range(self.conv3nums):
            self.multiConvs3.append(nn.Sequential(
                conv3x3(512, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                conv3x3(128, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                conv3x3(128, 512),
                nn.BatchNorm2d(512),
                nn.ReLU()))
        for _ in range(self.conv4nums):
            self.multiConvs4.append(nn.Sequential(
                conv3x3(1024, 256),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                conv3x3(256, 256),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                conv3x3(256, 1024),
                nn.BatchNorm2d(1024),
                nn.ReLU()))
        for _ in range(self.conv5nums):
            self.multiConvs5.append(nn.Sequential(
                conv3x3(2, 32),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                conv3x3(32, 32),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                conv3x3(32, 2),
                nn.BatchNorm2d(2),
                nn.ReLU()))

    def forward(self, x):
        out = self.dequantize(x)
        self.out_de = out
        out = out.view(-1, int(self.feedback_bits / self.B))
        out = self.sig(self.fc(out))
        out = out.view(-1, 2, 16, 32)

        out = self.multiConvs2[0](out)
        for i in range(1, self.conv2nums + 1):
            residual = out
            out = self.multiConvs2[i](out)
            out = residual + out

        out = self.multiConvs3[0](out)
        for i in range(1, self.conv3nums + 1):
            residual = out
            out = self.multiConvs3[i](out)
            out = residual + out

        out = self.multiConvs4[0](out)
        for i in range(1, self.conv4nums + 1):
            residual = out
            out = self.multiConvs4[i](out)
            out = residual + out

        out = self.multiConvs5[0](out)
        for i in range(1, self.conv5nums + 1):
            residual = out
            out = self.multiConvs5[i](out)
            out = residual + out

        out = self.out_cov(out)
        out = self.sig(out)
        return out


# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 128 by default.
class AutoEncoder(nn.Module):

    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]
