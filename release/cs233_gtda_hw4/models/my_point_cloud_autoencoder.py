"""
PC-AE.

The MIT License (MIT)
Originally created at 5/22/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab.
"""

import torch
from torch import nn
from ..in_out.utils import AverageMeter
from ..losses.chamfer import chamfer_loss

# In the unlikely case where you cannot use the JIT chamfer implementation (above) you can use the slower
# one that is written in pure pytorch:
# from ..losses.nn_distance import chamfer_loss
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
        return x

class MyPointCloudAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, classifier):
        """ AE constructor.
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        """
        super(MyPointCloudAutoencoder, self).__init__()
        self.encoder = encoder # Should be pointnet
        # self.decoder1 = decoders[0]
        # self.decoder2 = decoders[1]
        # self.decoder3 = decoders[2]
        # self.decoder4 = decoders[3]
        self.decoder = decoder
        self.classifier = classifier
        self.n_part_classifier = MLP(256, 4)
        self.seg_lossfn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0, 2.0]).cuda())


    def forward(self, pointclouds):
        """Forward pass of the AE
            :param pointclouds: B x N x 3
        """
        global_features = self.encoder(pointclouds).squeeze(-1)
        reconstructed = self.decoder(global_features)
        return reconstructed
    
    def count_unique_elements_per_row(self, t):
        unique_counts = torch.zeros(t.size(0)).to(t.device)
        for i in range(t.size(0)):
            unique_counts[i] = torch.unique(t[i]).size(0)
        return unique_counts
    
    def train_with_part_one_epoch(self, loader, optimizer, num_classes, device="cuda"):
        self.train()
        loss_seg_meter = AverageMeter()
        loss_recon_meter =AverageMeter()
        for data in loader:
            optimizer.zero_grad()
            points = data["point_cloud"].to(device)
            part_masks = data["part_mask"] # [B, N]
            
            part_masks = part_masks.to(device)
            
            part_masks_onehot = torch.nn.functional.one_hot(part_masks.long(), num_classes=num_classes).to(device).float()
            # Concatenate points with part_masks
            input_data = torch.cat([points, part_masks_onehot.float()], dim=-1)
            global_features = self.encoder(input_data).squeeze(-1)
            
            point_masks = self.classifier(points, global_features)
            recons = self.decoder(global_features)
            recon_loss = chamfer_loss(recons, points).mean()
            seg_loss = self.seg_lossfn(point_masks.reshape(-1,num_classes), part_masks_onehot.reshape(-1,num_classes)).mean()
            loss = recon_loss + 0.005 * seg_loss #+ pred_loss#- 0.005 * var_loss
            loss.backward()
            optimizer.step()
            loss_recon_meter.update(recon_loss.item(), points.size(0))
            loss_seg_meter.update(seg_loss.item(), points.size(0))
        return loss_recon_meter.avg, loss_seg_meter.avg

    
    @torch.no_grad()
    def embed(self, pointclouds):
        """ Extract from the input pointclouds the corresponding latent codes.
        :param pointclouds: B x N x 3
        :return: B x latent-dimension of AE
        """
        self.eval()
        return self.encoder(pointclouds).squeeze(-1)
        
    @torch.no_grad()
    def reconstruct_with_part(self, loader, device='cuda'):
        self.eval()
        point_clouds = []
        reconstructed_pcds = []
        part_masks = []
        shape_name = []
        for data in loader:
            points = data["point_cloud"].to(device)
            point_masks = data["part_mask"].to(device)
            point_masks_onehot = torch.nn.functional.one_hot(point_masks.long(), num_classes=4).to(device).float()
            point_clouds.append(points)
            shape_name = shape_name + data["model_name"]
            input_data = torch.cat([points, point_masks_onehot], dim=-1)
            recons = self.forward(input_data)
            reconstructed_pcds.append(recons)
            global_features = self.encoder(points).squeeze(-1)
            point_masks = self.classifier(points, global_features)
            point_masks = torch.argmax(point_masks, dim=2) # [B, N]
            part_masks.append(point_masks)
        return torch.vstack(point_clouds), torch.vstack(reconstructed_pcds), torch.vstack(part_masks), shape_name