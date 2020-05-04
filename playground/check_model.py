# -*- coding: utf-8 -*-

from unet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

model = UNet3D(in_channels=3, out_channels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = model.to(device)
summary(model, (3,32,112,112))