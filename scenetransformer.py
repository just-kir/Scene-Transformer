# -*- coding: utf-8 -*-


import os

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm as tq

#import torch.distributions as D
import torch.nn.functional as Ff

import torchvision

import numpy as np
from pprint import pprint
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#from transformers import get_cosine_schedule_with_warmup

class MLP_bn2d(nn.Module):
    def __init__(self, in_features, A, D, hidden_dim = None, softmax = False):
        super(MLP_bn2d, self).__init__()
        if hidden_dim == None:
          hidden_dim = D
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm2d(A)
        self.linear2 = nn.Linear(hidden_dim, D)
        self.bn2 = nn.BatchNorm2d(A)
        self.softmax_i = softmax
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input):  # Input is a 1D tensor
        y = Ff.relu(self.bn1(self.linear1(input)))
        #y = F.softmax(self.linear2(y), dim=1)
        y = self.bn2(self.linear2(y))
        if self.softmax_i == True:
          y = self.softmax(y)
        return y

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class SceneEncoder(nn.Module):
  def __init__(self, in_features, A = 200, T = 50, D = 256, num_heads = 4):
    super(SceneEncoder, self).__init__()
    self.A = A
    self.T = T
    self.D = D


    self.mlp_bn = MLP_bn2d(in_features, self.A, self.D)
    self.pos_enc = PositionalEncoding(self.D)

    self.att_t1 = nn.TransformerEncoderLayer(d_model=self.D, nhead=num_heads, batch_first=True)
    self.att_t2 = nn.TransformerEncoderLayer(d_model=self.D, nhead=num_heads, batch_first=True)
    self.att_t3 = nn.TransformerEncoderLayer(d_model=self.D, nhead=num_heads, batch_first=True)
    self.att_t4 = nn.TransformerEncoderLayer(d_model=self.D, nhead=num_heads, batch_first=True)

    self.att_a1 = nn.TransformerEncoderLayer(d_model=self.D, nhead=num_heads, batch_first=True)
    self.att_a2 = nn.TransformerEncoderLayer(d_model=self.D, nhead=num_heads, batch_first=True)
    self.att_a3 = nn.TransformerEncoderLayer(d_model=self.D, nhead=num_heads, batch_first=True)
    self.att_a4 = nn.TransformerEncoderLayer(d_model=self.D, nhead=num_heads, batch_first=True)


  def forward(self, input):
    B = input.shape[0]
    y = self.mlp_bn(input)
    y = self.pos_enc(y.reshape(B*self.A, self.T, self.D))

    y = self.att_t1(y)
    y = y.reshape(B, self.A, self.T, self.D).transpose(1, 2).reshape(B*self.T, self.A, self.D)
    y = self.att_a1(y)
    y = y.reshape(B, self.T, self.A, self.D).transpose(1, 2).reshape(B*self.A, self.T, self.D)
    y = self.att_t2(y)
    y = y.reshape(B, self.A, self.T, self.D).transpose(1, 2).reshape(B*self.T, self.A, self.D)
    y = self.att_a2(y)
    y = y.reshape(B, self.T, self.A, self.D).transpose(1, 2).reshape(B*self.A, self.T, self.D)
    y = self.att_t3(y)
    y = y.reshape(B, self.A, self.T, self.D).transpose(1, 2).reshape(B*self.T, self.A, self.D)
    y = self.att_a3(y)
    y = y.reshape(B, self.T, self.A, self.D).transpose(1, 2)

    #mean1 = torch.mean(y, 1, keepdim=True)
    #y = torch.cat((y, mean1), 1)
    mean2 = torch.mean(y, 2, keepdim=True)
    y = torch.cat((y, mean2), 2)

    y = y.reshape(B*self.A, self.T+1, self.D)
    y = self.att_t4(y)
    y = y.reshape(B, self.A, self.T+1, self.D).transpose(1, 2).reshape(B*(self.T+1), self.A, self.D)
    y = self.att_a4(y)
    y = y.reshape(B, self.T+1, self.A, self.D).transpose(1, 2)

    return y

class SceneDecoder(nn.Module):
  def __init__(self, F = 5, A = 200, T = 51, D = 256, num_heads = 4):
    super(SceneDecoder, self).__init__()
    self.A = A
    self.T = T
    self.D = D
    self.F = F

    self.out = 4

    self.mlp_bn1 = MLP_bn2d(D+F, A, D)
    self.mlp_bn2 = MLP_bn2d(D, A, self.out, hidden_dim = D)
    self.mlp_bn3 = MLP_bn2d(D, F, 1, hidden_dim = D, softmax=True)




    self.att_t1 = nn.TransformerEncoderLayer(d_model=D, nhead=num_heads, batch_first=True)
    self.att_t2 = nn.TransformerEncoderLayer(d_model=D, nhead=num_heads, batch_first=True)

    self.att_a1 = nn.TransformerEncoderLayer(d_model=D, nhead=num_heads, batch_first=True)
    self.att_a2 = nn.TransformerEncoderLayer(d_model=D, nhead=num_heads, batch_first=True)


  def forward(self, input):

    B = input.shape[0]
    y = input.unsqueeze(1).repeat(1, self.F, 1, 1, 1)
    one_hot = Ff.one_hot(torch.arange(0, self.F)).reshape((1, self.F, 1, 1, self.F)).repeat((B, 1, self.A, self.T, 1)).to(device)
    y = torch.cat((y, one_hot), 4)
    y = self.mlp_bn1(y.reshape(B*self.F, self.A, self.T, self.D+self.F)).reshape(B, self.F, self.A, self.T, self.D)

    y = self.att_t1(y.reshape(B*self.F*self.A, self.T, self.D)).reshape(B, self.F, self.A, self.T, self.D)
    y = y.transpose(2,3).reshape(B*self.F*self.T, self.A, self.D)
    y = self.att_a1(y).reshape(B, self.F, self.T, self.A, self.D).transpose(2, 3)

    y = self.att_t2(y.reshape(B*self.F*self.A, self.T, self.D)).reshape(B, self.F, self.A, self.T, self.D)
    y = y.transpose(2,3).reshape(B*self.F*self.T, self.A, self.D)
    y = self.att_a2(y).reshape(B, self.F, self.T, self.A, self.D).transpose(2, 3)

    y1 = y[:, :, :, self.T-1, :]
    y = y[:, :, :, :self.T-1, :]
    y = self.mlp_bn2(y.reshape(B*self.F, self.A, self.T-1, self.D)).reshape(B, self.F, self.A, self.T-1, self.out)
    y1 = self.mlp_bn3(y1).squeeze()

    return y, y1

class SceneTransformer(nn.Module):
  def __init__(self, in_features, F = 5, A = 200, T = 50, D = 256, num_heads = 4):
    super(SceneTransformer, self).__init__()
    self.encoder = SceneEncoder(in_features, A, T, D, num_heads)
    self.decoder = SceneDecoder(F, A, T+1, D, num_heads)

  def forward(self, input):
    y = self.encoder(input)
    y, y1 = self.decoder(y)
    return y, y1

# B = 4
# A = 20
# F = 3
# T = 50
# D = 128
# in_features = 10


# x = torch.randn(B, A, T, in_features)
# ST = SceneTransformer(in_features, F, A, T, D)
# y, y1 = ST(x)
# print(y.shape)
# print(y1.shape)
