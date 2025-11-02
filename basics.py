import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=6, n_layers=2):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=3, act='gelu', res=True, LN=False):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.LN = LN  
        
        pre_layers = [nn.Linear(n_input, n_hidden)]
        if LN:
            pre_layers.append(nn.LayerNorm(n_hidden))
        pre_layers.append(act())
        self.linear_pre = nn.Sequential(*pre_layers)
        
        self.linears = nn.ModuleList()
        for _ in range(n_layers):
            layers = [nn.Linear(n_hidden, n_hidden)]
            if LN:
                layers.append(nn.LayerNorm(n_hidden))
            layers.append(act())
            self.linears.append(nn.Sequential(*layers))
        
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class GAT(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=6, n_layers=4, heads=4, dropout=0.1):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        
        self.convs.append(GATConv(
            input_dim, 
            hidden_dim, 
            heads=heads, 
            dropout=dropout
        ))
        
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(
                hidden_dim * heads,  
                hidden_dim, 
                heads=heads, 
                dropout=dropout
            ))
        
        self.convs.append(GATConv(
            hidden_dim * heads, 
            output_dim, 
            heads=1,  
            dropout=dropout
        ))
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)  
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

class Encoder_GCN_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=6, n_layers=2):
        super(Encoder_GCN_decoder, self).__init__()
        self.encoder = MLP(input_dim, hidden_dim, hidden_dim, n_layers=3)
        self.gcn = GAT(hidden_dim, hidden_dim, hidden_dim, n_layers=n_layers)
        self.decoder = MLP(hidden_dim, hidden_dim//2, output_dim, n_layers=3)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = self.ln1(x)
        x = self.gcn(x, edge_index)
        x = self.ln2(x)
        x = self.decoder(x)
        return x
