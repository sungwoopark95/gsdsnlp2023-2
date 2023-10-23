import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from math import sqrt

# you can refer to the implementation provided by PyTorch for more information
# https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
"""
Parameters
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True

Inputs: input, hidden
input of shape (batch, input_size): tensor containing input features
hidden of shape (batch, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided.

Outputs: h'
h’ of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch
"""
class GRUCell_assignment(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell_assignment, self).__init__()
        # hyper-parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.Wir = nn.Linear(input_size, hidden_size, bias)
        self.Whr = nn.Linear(hidden_size, hidden_size, bias)
        self.Wiz = nn.Linear(input_size, hidden_size, bias)
        self.Whz = nn.Linear(hidden_size, hidden_size, bias)
        self.Win = nn.Linear(input_size, hidden_size, bias)
        self.Whn = nn.Linear(hidden_size, hidden_size, bias)
        nn.init.uniform_(self.Wir.weight.data, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
        nn.init.uniform_(self.Wiz.weight.data, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
        nn.init.uniform_(self.Win.weight.data, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
    def forward(self, inputs, hidden=False):
        if hidden is False: hidden = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        x, h = inputs, hidden

        r = torch.sigmoid(self.Wir(x) + self.Whr(h))
        z = torch.sigmoid(self.Wiz(x) + self.Whz(h))
        n = torch.tanh(self.Win(x) + r * self.Whn(h))
        h = (1-z) * n + z * h

        return h


# you can refer to the implementation provided by PyTorch for more information
# https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html

class LSTMCell_assignment(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell_assignment, self).__init__()
        # hyper-parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Input gate
        self.Wii = nn.Linear(input_size, hidden_size, bias=True)
        self.Whi = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Forget gate
        self.Wif = nn.Linear(input_size, hidden_size, bias=True)
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Cell gate
        self.Wig = nn.Linear(input_size, hidden_size, bias=True)
        self.Whg = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Output gate
        self.Wio = nn.Linear(input_size, hidden_size, bias=True)
        self.Who = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Initialize weights with Xavier uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        # # Bias Initialization
        # if bias:
        #     nn.init.zeros_(self.bi)
        #     nn.init.zeros_(self.bf)
        #     nn.init.zeros_(self.bg)
        #     nn.init.zeros_(self.bo)

    def forward(self, inputs, dec_state):
        x, h, c = inputs, dec_state[0], dec_state[1]
        ### Inputs: input, (h_0, c_0)
        ### input of shape (batch, input_size): tensor containing input features
        ### h_0 of shape (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        ### c_0 of shape (batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        ### If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.

        ### Outputs: (h_1, c_1)
        ### h_1 of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch
        ### c_1 of shape (batch, hidden_size): tensor containing the next cell state for each element in the batch
        ### YOUR CODE HERE (~6 Lines)
        ### TODO - Implement forward prop in LSTM cell.
        
        # Compute the input, forget, and output gates
        i_t = torch.sigmoid(self.Wii(x) + self.Whi(h))
        f_t = torch.sigmoid(self.Wif(x) + self.Whf(h))
        o_t = torch.sigmoid(self.Wio(x) + self.Who(h))
        
        # Compute the cell gate
        g_t = torch.tanh(self.Wig(x) + self.Whg(h))
        
        # Compute new cell and hidden states
        c = f_t * c + i_t * g_t
        h = o_t * torch.tanh(c)

        return (h, c)