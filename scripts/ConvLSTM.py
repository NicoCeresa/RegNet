"""
RegNet Implimentation 

by Nico Ceresa
"""
import torch
import numpy as np
from torch import nn
from typing import List
from torch.auto_grad import Variable

device = 'cuda' if torch.cuda.is_available else 'cpu'

class ConvLSTMCell(nn.Module):
    """
    Implimentation of a single ConvLSTM cell

    Args:
       in_dim: int 
       hidden_dim: int
       kernel_size: int 
       padding: int
       bias: int
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
    
    def init_hidden(self, batch_size, img_size):
        height, width = img_size.shape
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width, device=device)),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width, device=device)))
    
    def forward(self, x, h, c):
        
        it = self.sigmoid(self.conv(x) + self.conv(h) + c * self.W_c)
        ft = self.sigmoid(self.conv(x) + self.conv(h) + c * self.W_f)
        
        Ct = np.multiply(ft, c) + np.multiply(it, self.conv(x) + self.conv(h))
        ot = self.sigmoid(self.conv(x) + self.conv(h) + Ct * self.W_o)
        
        Ht = np.multiply(ot, self.tanh(Ct))
        
        return Ht, Ct
    
        
class ConvLSTM(nn.Module):
    """
    Full ConvLSTM with multiple layers consisting of the ConvLSTMCell class

    Args:
        input_channels: List
        hidden_channels: List
        kernel_size: List
        steps=2: int
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.cell_layers = []
        for i in range(self.num_layers):
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i])
            self.cell_layers.append(cell)

            
    def forward(self, x, hidden_state=None):
        """

        Args:
            x (_type_): 5D Tensor -> (seq_len, batches, channels, height, width)
            hidden_state (_type_, optional): _description_. Defaults to None.
        """
        # hold the states (H, C) so we can easily acces prev
        states = []
        # shape (seq_len, batches, channels, height, width) -> (batches, seq_len, channels, height, width)
        curr_input = x.permute(1, 0, 2, 3, 4)
        b, sequences, _, height, width = curr_input.shape
        
        if not hidden_state:
            hidden_state = self._init_hidden(batch_size=b,
                                             img_size=(height, width))
        
        for layer in range(self.num_layers):
            h, c = hidden_state[layer]
            output_inner = []
            for i in range(sequences):
                h_c = self.cell_layers[layer]
                output_inner.append(h_c[0])
                
            states.append(h_c)
            curr_input = torch.cat(output_inner, 0).view(curr_input.size(0), *outputs_inner[0].size()) # (seq_len, batches, channels, H, W)
        return curr_input, states
        
        
    def _init_hidden(self, batch_size, img_size):
        height, width = img_size.shape
        init_hiddens = []
        for layer in range(self.num_layers):
            init_hiddens.append(self.cell_layers[layer].init_hidden_weights(batch_size=batch_size, img_size=img_size))
        return init_hiddens