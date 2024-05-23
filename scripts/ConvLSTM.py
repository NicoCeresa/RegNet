"""
RegNet Implimentation 

by Nico Ceresa
"""
import torch
import numpy as np
from torch import nn
from typing import List, Tuple
from torch.functional import Tensor
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
    def __init__(self, in_dim: int, channels: int, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.channels = channels
        self.conv = nn.Conv2d(in_channels=in_dim + channels,
                              out_channels=channels * 4,
                              kernel_size=self.kernel_size,
                              padding=kernel_size // 2,
                              bias = True)
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def tanh(self, x):
        num = np.exp(x) - np.exp(-x)
        denom = np.exp(x) + np.exp(-x)
        return num/denom
    
    def init_hidden(self, batch_size, channels, image):
        height, width = image.shape
        return (Variable(torch.zeros(batch_size, channels, height, width, device=device)),
                Variable(torch.zeros(batch_size, channels, height, width, device=device)))
    
    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        h, c = c.to(device), h.to(device)
        
        x = np.cat([x, h], dim=1)
        conv_x = self.conv(x)
        
        it, ft, ot, gt = torch.split(conv_x, self.hidden_dim, dim=1)
        
        it = self.sigmoid(it)
        ft = self.sigmoid(ft)
        ot = self.sigmoid(ot)
        gt = self.tanh(dt)
        
        C_next = it * Ct + ot * dt
        H_next = ot * self.tanh(Ct)
        
        return H_next, C_next
    
        
class ConvLSTM(nn.Module):
    """
    Full ConvLSTM with multiple layers consisting of the ConvLSTMCell class

    Args:
        input_channels: List
        hidden_channels: List
        kernel_size: List
        steps=2: int
    """
    def __init__(self, input_channels, hidden_dim, channels, kernel_size=3):
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_dim)
        self.cell_layers = []
        for i in range(self.num_layers):
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_dim[i], self.kernel_size[i])
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
            h_c = hidden_state[layer]
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