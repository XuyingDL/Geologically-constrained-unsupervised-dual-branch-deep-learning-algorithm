import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv


class SpatialCNNAutoEncoder(nn.Module):
    def __init__(self, InputSize, HiddenSize, DropOut=0.2):
        super(SpatialCNNAutoEncoder, self).__init__()
        self.Conv1 = nn.Conv2d(InputSize, 16, 3)
        self.Conv2 = nn.Conv2d(16, 8, 3)
        self.Conv3 = nn.Conv2d(8, HiddenSize, 3)

        self.TransConv1 = nn.ConvTranspose2d(HiddenSize, 8, 3)
        self.TransConv2 = nn.ConvTranspose2d(8, 16, 3)
        self.TransConv3 = nn.ConvTranspose2d(16, InputSize, 3)

        self.dropout = DropOut

    def forward(self, x):
        x = F.relu(self.Conv1(x))
        F.dropout(x, self.dropout)
        x = F.relu(self.Conv2(x))
        x = self.Conv3(x)
        HiddenX = x
        x = F.relu(self.TransConv1(x))
        x = F.relu(self.TransConv2(x))
        F.dropout(x, self.dropout)
        x = self.TransConv3(x)
        return HiddenX, x


class SpatialGCNAutoEncoder(nn.Module):
    def __init__(self, InputSize, HiddenSize, DropOut=0.5):
        super(SpatialGCNAutoEncoder, self).__init__()
        self.Conv1 = GCNConv(InputSize, 16, bias=False)
        self.Conv2 = GCNConv(16, 8, bias=False)
        self.Conv3 = GCNConv(8, HiddenSize, bias=False)
        #
        self.Lin1 = nn.Linear(HiddenSize, 8)
        self.Lin2 = nn.Linear(8, 16)
        self.Lin3 = nn.Linear(16, InputSize)
        self.TransConv1 = GCNConv(HiddenSize, 8, bias=False)
        self.TransConv2 = GCNConv(8, 16, bias=False)
        self.TransConv3 = GCNConv(16, InputSize, bias=False)

        self.dropout = DropOut

    def forward(self, x, adj):
        x = F.relu(self.Conv1(x, adj))
        F.dropout(x, self.dropout)
        x = F.relu(self.Conv2(x, adj))
        F.dropout(x, self.dropout)
        x = self.Conv3(x, adj)
        HiddenX = x
        x = F.relu(2 * self.Lin1(x) - self.TransConv1(x, adj))
        F.dropout(x, self.dropout)
        x = F.relu(2 * self.Lin2(x) - self.TransConv2(x, adj))
        F.dropout(x, self.dropout)
        x = 2 * self.Lin3(x) - self.TransConv3(x, adj)
        return HiddenX, x


class SpectrumFNNAutoEncoder(nn.Module):
    def __init__(self, InputSize, HiddenSize, DropOut=0.2):
        super(SpectrumFNNAutoEncoder, self).__init__()
        self.Lin1 = nn.Linear(InputSize, HiddenSize)
        self.Lin2 = nn.Linear(16, HiddenSize)

        self.TransLin1 = nn.Linear(HiddenSize, 16)
        self.TransLin2 = nn.Linear(HiddenSize, InputSize)

        self.dropout = DropOut

    def forward(self, x):
        x = F.layer_norm(x, x.shape)
        x = self.Lin1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.Lin2(x)
        Hidden = x
        x = self.TransLin1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.TransLin2(x)
        return Hidden, x


class SpectrumLSTMAutoEncoder(nn.Module):
    def __init__(self, InputSize, HiddenSize, DropOut=0.2):
        super(SpectrumLSTMAutoEncoder, self).__init__()
        self.RNN = nn.LSTM(input_size=InputSize, num_layers=2, dropout=0.3, hidden_size=8, bidirectional=True)
        self.Projection = nn.Linear(16, InputSize)

    def forward(self, x):
        x = F.layer_norm(x, x.shape)
        Hidden = x
        x, _ = self.RNN(x)
        x = self.Projection(x)
        return Hidden, x
