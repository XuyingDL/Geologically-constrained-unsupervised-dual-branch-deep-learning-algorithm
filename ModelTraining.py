import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import var
from osgeo import gdal
import torch.nn.functional as F
import torch
import Model
from MINXINAN import ModelList

seed = 50
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = "cuda"


def Predict(Hidden, CsvName, PointXYPath):
    # XY.shape is n*2
    XY = np.load(PointXYPath)
    result = np.append(XY, np.reshape(Hidden, [Hidden.shape[0], 1]), axis=1)
    DF = pd.DataFrame(result, columns=["POINT_X", "POINT_Y", "Result"])
    DF.to_csv(CsvName)


InputSize = 39
HiddenSize = 2
WindowSize = 7
epoch = 1000


def Constrain(ConstrainPath: str, Out: torch.Tensor, InitData: torch.Tensor):
    constrain = np.load(ConstrainPath)
    constrain = torch.FloatTensor(constrain)
    Hidden = F.l1_loss(Out, InitData, reduction="none")
    Hidden = torch.sum(Hidden, dim=-1)
    Loss2 = F.mse_loss(Hidden, constrain)
    return Loss2


def DualModelTrainAndPredict(SpatialInputPath: str, SpectralInputPath: str, EdgePath: str, ConstrainPath: str,
                             PointXYPath: str):
    SpatialInputNumpy = np.load(SpatialInputPath)
    SpectralInputNumpy = np.load(SpectralInputPath)
    edge = np.load(EdgePath)
    SpatialInput = torch.FloatTensor(SpatialInputNumpy).to(device)
    SpectralInput = torch.FloatTensor(SpectralInputNumpy).to(device)
    edge = torch.LongTensor(edge).to(device)
    # If the spatial branch is CAE, the result is Dual-AE(CAE) and the spatial data must be n*39*WindowSize*WindowSize
    SpatialAutoEncoder = ModelList.SpatialGCNAutoEncoder(InputSize, HiddenSize).to(device)
    # If the spectrum branch is FNN, the result is Dual-AE(FNN)
    SpectralAutoEncoder = ModelList.SpectrumLSTMAutoEncoder(InputSize, HiddenSize).to(device)

    FusionLinear = nn.Linear(InputSize * 2, InputSize).to(device)
    FusionLinear1 = nn.Linear(InputSize, InputSize).to(device)
    Fusion1 = Model.Fusion(InputSize, 7).to(device)

    Optim = optim.Adam([
        {'params': SpatialAutoEncoder.parameters()},
        {'params': SpectralAutoEncoder.parameters()},
        {'params': FusionLinear.parameters()},
        {'params': FusionLinear1.parameters()},
        {'params': Fusion1.parameters()}],
        lr=0.005, weight_decay=0.005)
    Loss = []
    for i in range(epoch):
        Optim.zero_grad()
        # If the spatial branch is CAE, use "SpatialHidden, SpatialOut = SpatialAutoEncoder(SpatialInput)"
        SpatialHidden, SpatialOut = SpatialAutoEncoder(SpatialInput, edge)
        SpectralHidden, SpectralOut = SpectralAutoEncoder(SpectralInput)
        # If the spatial branch is CAE, use "Out = Fusion1(SpatialOut, SpectralOut)"
        Out = FusionLinear(torch.cat([SpatialOut, SpectralOut], dim=-1))
        Out = FusionLinear1(Out)
        Loss1 = F.mse_loss(Out, SpectralInput)
        Loss2 = Constrain(ConstrainPath, Out, SpectralInput)
        Loss1 += Loss2
        Loss1.backward(retain_graph=True)
        loss = [Loss1.to('cpu').detach().numpy()]
        Optim.step()
        Loss.append(loss)
        print(i, loss[0])

    SpatialHidden, SpatialOut = SpatialAutoEncoder(SpatialInput, edge)
    SpectralHidden, SpectralOut = SpectralAutoEncoder(SpectralInput)
    Out = FusionLinear(torch.cat([SpatialOut, SpectralOut], dim=-1))
    Out = FusionLinear1(Out)
    Hidden = F.l1_loss(Out, SpectralInput, reduction="none")
    Hidden = torch.sum(Hidden, dim=-1)
    Hidden = Hidden.to('cpu').detach().numpy()
    Predict(Hidden, r"Dual_AutoEncoder.csv", PointXYPath)


def SpatialModelTrainAndPredict(SpatialInputPath: str, EdgePath: str, PointXYPath: str):
    SpatialInputNumpy = np.load(SpatialInputPath)
    edge = np.load(EdgePath)
    SpatialInput = torch.FloatTensor(SpatialInputNumpy).to(device)
    edge = torch.LongTensor(edge).to(device)
    # If the model is CAE, the spatial data must be n*39*WindowSize*WindowSize
    SpatialAutoEncoder = ModelList.SpatialGCNAutoEncoder(InputSize, HiddenSize).to(device)
    Optim = optim.Adam([
        {'params': SpatialAutoEncoder.parameters()}],
        lr=0.005, weight_decay=0.005)
    Loss = []
    for i in range(epoch):
        Optim.zero_grad()
        SpatialHidden, SpatialOut = SpatialAutoEncoder(SpatialInput, edge)
        Loss1 = F.mse_loss(SpatialOut, SpatialInput)
        Loss1.backward(retain_graph=True)
        loss = [Loss1.to('cpu').detach().numpy()]
        Optim.step()
        Loss.append(loss)
        print(i, loss[0])
    SpatialHidden, SpatialOut = SpatialAutoEncoder(SpatialInput, edge)
    Hidden = F.l1_loss(SpatialOut, SpatialInput, reduction="none")
    # If the model is CAE, use "Hidden = Hidden.reshape(Hidden.shape[0], Hidden.shape[1] * Hidden.shape[2] * Hidden.shape[3])"
    Hidden = torch.sum(Hidden, dim=-1)
    Hidden = Hidden.to('cpu').detach().numpy()
    Predict(Hidden, r"Spatial_AutoEncoder.csv", PointXYPath)


def SpectrumModelTrainAndPredict(SpectralInputPath: str, PointXYPath: str):
    SpectralInputNumpy = np.load(SpectralInputPath)
    SpectralInput = torch.FloatTensor(SpectralInputNumpy).to(device)
    # If the spectrum branch is FNN, the result is Dual-AE(FNN)
    # SpectralAutoEncoder = ModelList.SpectrumLinearAutoEncoder(InputSize, HiddenSize).to(device)
    SpectralAutoEncoder = ModelList.SpectrumLSTMAutoEncoder(InputSize, HiddenSize).to(device)
    Optim = optim.Adam([
        {'params': SpectralAutoEncoder.parameters()}],
        lr=0.005, weight_decay=0.005)
    Loss = []
    for i in range(epoch):
        Optim.zero_grad()
        SpectralHidden, SpectralOut = SpectralAutoEncoder(SpectralInput)
        Loss1 = F.mse_loss(SpectralOut, SpectralInput)
        Loss1.backward(retain_graph=True)
        loss = [Loss1.to('cpu').detach().numpy()]
        Optim.step()
        Loss.append(loss)
        print(i, loss[0])

    SpectralHidden, SpectralOut = SpectralAutoEncoder(SpectralInput)
    Hidden = F.l1_loss(SpectralOut, SpectralInput, reduction="none")
    Hidden = torch.sum(Hidden, dim=-1)
    Hidden = Hidden.to('cpu').detach().numpy()
    Predict(Hidden, r"Spectrum_AutoEncoder.csv", PointXYPath)


if __name__ == "__main__":
    # All data is saved in .npy file, so the InputPath is data.npy
    DualModelTrainAndPredict(SpatialInputPath="SpatialData.npy",
                             SpectralInputPath="SpectrumData.npy",
                             EdgePath="Edge.npy",
                             ConstrainPath="Constrain.npy",
                             PointXYPath="PointXY.npy")
