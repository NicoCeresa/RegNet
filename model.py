"""
RegNet Implimentation 

by Nico Ceresa
"""
import torch
import numpy as np
from torch import nn
from typing import List

class ConvLSTMCell(nn.Module):
    """
    Implimentation of the ConvLSTM

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_dim, hidden_dim, kernel_size, padding, bias):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.padding = padding
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.in_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias = self.bias)

        self.W_c = None
        self.W_f = None
        self.W_o = None
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def tanh(self, x):
        num = np.exp(x) - np.exp(-x)
        denom = np.exp(x) + np.exp(-x)
        return num/denom
    
    def forward(self, x, h, c):
        
        it = self.sigmoid(self.conv(x) + self.conv(h) + c * self.W_c)
        ft = self.sigmoid(self.conv(x) + self.conv(h) + c * self.W_f)
        
        Ct = np.multiply(ft, c) + np.multiply(it, self.conv(x) + self.conv(h))
        ot = self.sigmoid(self.conv(x) + self.conv(h) + Ct * self.W_o)
        
        Ht = np.multiply(ot, self.tanh(Ct))
        
        return Ht, Ct
    
        
class ConvLSTM(nn.Module):
    def __init__(self):
        pass


class RegNet(nn.Module):
    """
    RegNet Class containing init and forward methods
    """
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        pass
    
        