"""
RegNet Implimentation 

by Nico Ceresa

1) RNN-Regulated ResNet Module (RegNet module): The
illustration of RegNet module is shown in Fig. 3(a). Here, we
choose ConvLSTM for expounding. 𝐻
𝑡−1 denotes the earlier
output from ConvLSTM, and 𝐻𝑡
is output of the ConvLSTM
at 𝑡-th module . 𝑋𝑡_𝑖 denotes the 𝑖-th feature map at the 𝑡-th module.
The 𝑡-th RegNet(ConvLSTM) module can be expressed as
BN - Batch Normalization
* - Convolution Operation

1) Xt_2 = ReLU(BN(Wt_12 * Xt_1 + bt_12))

2) [Ht, Ct] = ReLU(BN(ConvLSTM(Xt_2, [H{t-1}, C{t-1}])))
-> need to grab [-1] index of outputs

3) Xt_3 = ReLU(BN(Wt_23 * np.cat(Xt_2, Ht)))
->Ht is the output from ConvLSTM

4) Xt_4 = BN(Wt_34 * Xt_3 + bt_34)

5) X{t+1}_1 = ReLU(Xt_1 + Xt_4)

-> 3x3 Kernels: Wt_12 & Wt_34
-> 1x1 Kernel: Wt_23

"""
import torch
import numpy as np
from torch import nn
from ConvLSTM import ConvLSTM
from typing import List, Tuple
from torch.functional import Tensor


device = 'cuda' if torch.cuda.is_available else 'cpu'

class RegNetCell(nn.Module):
    """
    RegNet Class containing init and forward methods
    """
    def __init__(self, input_channels, hidden_dim, channels, conv_lstm):
        super(RegNetCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Block 1
        self.conv1 = self.conv = nn.Conv2d(in_channels=self.input_channels,
                              out_channels=self.hidden_channels,
                              kernel_size=3,
                              padding=1,
                              stride=1,
                              bias = False)
        self.BatchNorm1 = nn.BatchNorm2d(channels)
        self.ReLU1 = nn.Relu()
        
        # Block 2
        self.conv_lstm = conv_lstm
        self.BatchNorm_lstm  = nn.BatchNorm2d(channels)
        self.ReLU_lstm  = nn.Relu()
        
        # Block 3
        self.conv3 = self.conv = nn.Conv2d(in_channels=self.input_channels,
                              out_channels=self.hidden_dim,
                              kernel_size=1,
                              stride=1,
                              bias = False)
        self.BatchNorm3 = nn.BatchNorm2d(channels)
        self.ReLU3 = nn.Relu()
        
        # Block 4
        self.conv4 = self.conv = nn.Conv2d(in_channels=self.input_channels,
                              out_channels=self.hidden_channels,
                              kernel_size=3,
                              padding=1,
                              stride=1,
                              bias = False)
        self.BatchNorm4 = nn.BatchNorm2d(channels)
        self.ReLU4 = nn.Relu()
        
        # Block 5
        self.Relu5 = nn.ReLU()
    
    def init_hidden_weights(self, batch_size, img_size):
        height, width = img_size.shape
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width, device=device)),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width, device=device)))
        
    def forward(self, x: Tensor, state:Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        H, C = hidden_state[-1]
        
        Xt_2 = self.ReLU(self.BN(self.conv(x)))
        
        H, C = self.ReLU(self.BN(self.ConvLSTM(Xt_2, [H, C]))) 
        
        X_H = np.cat([Xt_2, H], dim = 1)
        Xt_3 = self.ReLU(self.BN(self.conv1(X_H)))
        Xt_4 = self.BN(self.conv3(Xt_3))
        X_out = self.ReLU(x + Xt_4)
        """
        H, C = state
        x2 = self.conv1(x)
        x2 = self.BatchNorm1(x2)
        x2 = self.ReLU1(x2)
        
        x_convLSTM = self.conv_lstm(x2, state)
        x_convLSTM = self.BatchNorm_lstm(x_convLSTM)
        H_new, C_new = self.ReLU_lstm(x_convLSTM)
        
        x3 = torch.cat([x2, H_new], dim=1)
        x3 = self.conv3(x3)
        x3 = self.BatchNorm3(x3)
        x3 = self.ReLU3(x3)
        
        x4 = self.conv4(x3)
        x4 = self.BatchNorm4(x4)
        
        x_out = self.Relu5(x + x4)
        
        return X_out, H, C
    
class RegNetBlock(nn.Module):
    
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(RegNetBlock, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.cell_layers = []
        for i in range(self.num_layers):
            cell = RegNetCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size[i])
            self.cell_layers.append(cell)
    
    def _init_hidden_weights(self, batch_size, img_size):
        height, width = img_size.shape
        init_hiddens = []
        for layer in range(self.num_layers):
            init_hiddens.append(self.cell_layers[layer].init_hidden_weights(batch_size=batch_size, img_size=img_size))
        return init_hiddens
        
    def forward(self, x, hidden_state = None):
        states = []
        outputs = []
        # x.size = B, C, H, W
        batch_size, _, height, width = x.shape()
        if not hidden_state:
            hidden_state = self._init_hidden_weights(batch_size, (height, width))
            
        for layer in range(self.num_layers):
            X, H, C = self.cell_layers[layer]
            states.append((H, C))
            outputs.append(X)
        
        return outputs