"""
Point-Net.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, init_feat_dim=3, conv_dims=[32, 64, 64, 128, 128], student_optional_hyper_param=None):
        """
        Students:
        You can make a generic function that instantiates a point-net with arbitrary hyper-parameters,
        or go for an implemetnations working only with the hyper-params of the HW.
        Do not use batch-norm, drop-out and other not requested features.
        Just nn.Linear/Conv1D/ReLUs and the (max) poolings.
        
        :param init_feat_dim: input point dimensionality (default 3 for xyz)
        :param conv_dims: output point dimensionality of each layer
        """
        super(PointNet, self).__init__()
        self.init_feat_dim = init_feat_dim 
        self.conv_dims = conv_dims
        self.student_optional_hyper_param = student_optional_hyper_param # Should be a dictionary

        self.pointwise_mlps = nn.ModuleList()
        input_dim = self.init_feat_dim
        for conv_dim in self.conv_dims:
            self.pointwise_mlps.append(nn.Conv1d(input_dim, conv_dim, 1))
            input_dim = conv_dim
        #self.output_layer = nn.Conv1d(2*self.conv_dims[-1], self.conv_dims[-1], 1)  
        
    def forward(self, pointclouds):
        """
        Run forward pass of the PointNet model on a given point cloud.
        :param pointclouds: (B x N x 3) point cloud
        """
        pointclouds = pointclouds.permute(0, 2, 1) # B x 3 x N
        for i,mlp in enumerate(self.pointwise_mlps):
            pointclouds = mlp(pointclouds)
            pointclouds = F.relu(pointclouds)
        # maxpooling
        global_features = torch.max(pointclouds, dim=2, keepdim=True)[0] 
        return global_features
    
class MyPointNet(nn.Module):
    def __init__(self, init_feat_dim=3, conv_dims=[32, 64, 64, 128, 128], student_optional_hyper_param=None):
        """
        Students:
        You can make a generic function that instantiates a point-net with arbitrary hyper-parameters,
        or go for an implemetnations working only with the hyper-params of the HW.
        Do not use batch-norm, drop-out and other not requested features.
        Just nn.Linear/Conv1D/ReLUs and the (max) poolings.
        
        :param init_feat_dim: input point dimensionality (default 3 for xyz)
        :param conv_dims: output point dimensionality of each layer
        """
        super(MyPointNet, self).__init__()
        self.init_feat_dim = init_feat_dim 
        self.conv_dims = conv_dims
        self.student_optional_hyper_param = student_optional_hyper_param # Should be a dictionary

        self.pointwise_mlps = nn.ModuleList()
        input_dim = self.init_feat_dim
        for conv_dim in self.conv_dims:
            self.pointwise_mlps.append(nn.Conv1d(input_dim, conv_dim, 1))
            input_dim = conv_dim
        #self.output_layer = nn.Conv1d(2*self.conv_dims[-1], self.conv_dims[-1], 1)  
        
    def forward(self, pointclouds):
        """
        Run forward pass of the PointNet model on a given point cloud.
        :param pointclouds: (B x N x 3) point cloud
        """
        pointclouds = pointclouds.permute(0, 2, 1) # B x 3 x N
        for i,mlp in enumerate(self.pointwise_mlps):
            pointclouds = mlp(pointclouds)
            if i < len(self.pointwise_mlps)-1:
                pointclouds = F.relu(pointclouds)
        # maxpooling
        global_features = torch.max(pointclouds, dim=2, keepdim=True)[0]
        global_features += torch.kthvalue(pointclouds, 2, dim=2, keepdim=True)[0]
        global_features += torch.kthvalue(pointclouds, 3, dim=2, keepdim=True)[0]
        global_features += torch.kthvalue(pointclouds, 4, dim=2, keepdim=True)[0]
        return global_features

if __name__ == '__main__':
    # simple test
    pointnet = PointNet()
    pointclouds = torch.rand(32, 2048, 3)
    global_features = pointnet(pointclouds).squeeze(-1)
    print(global_features.size())
