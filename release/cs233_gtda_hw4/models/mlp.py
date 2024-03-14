"""
Multi-layer perceptron.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn


class MLP(nn.Module):
    """ Multi-layer perceptron. That is a k-layer deep network where each layer is a fully-connected (nn.Linear) layer, with
    (optionally) batch-norm, a non-linearity and dropout.

    Students: again, you can use this scaffold to make a generic MLP that can be used with multiple-hyper parameters
    or, opt for a perhaps simpler custom variant that just does so for HW4. For HW4 do not use batch-norm, drop-out
    or other non-requested features, for the non-bonus question.
    """

    def __init__(self, in_feat_dim, out_channels, b_norm=False, 
                 dropout_rate=0, non_linearity=nn.ReLU(inplace=True)):
        """Constructor
        :param in_feat_dim: input feature dimension
        :param out_channels: list of ints describing each the number hidden/final neurons.
        :param b_norm: True/False, or list of booleans
        :param dropout_rate: int, or list of int values
        :param non_linearity: nn.Module
        """
        super(MLP, self).__init__()
        self.in_feat_dim = in_feat_dim
        self.out_channels = out_channels
        self.b_norm = b_norm
        self.dropout_rate = dropout_rate
        self.non_linearity = non_linearity
        self.hidden_dims = [256, 384]
        self.fc1 = nn.Linear(in_feat_dim, self.hidden_dims[0])
        self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.fc3 = nn.Linear(self.hidden_dims[1], out_channels)
        # If batch-norm is requested, add it here.
        self.bn1 = nn.BatchNorm1d(self.hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(self.hidden_dims[1])
        # If dropout is requested, add it here.
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        """
        Run forward pass of MLP
        :param x: (B x in_feat_dim) point cloud
        """
        batch_size = x.size(0)
        x = self.fc1(x)
        if self.b_norm:
            x = self.bn1(x)
        x = self.non_linearity(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        if self.b_norm:
            x = self.bn2(x)
        x = self.non_linearity(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x.view(batch_size, -1, 3)

if __name__ == '__main__':
    # simple test
    mlp = MLP(10, 3 * 1024, b_norm=True)
    x = torch.rand(32, 10)
    out = mlp(x)
    print(out.size())