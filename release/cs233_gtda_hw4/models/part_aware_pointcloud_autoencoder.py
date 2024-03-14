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


class PartAwareAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, classifier = None):
        """ AE constructor.
        :param encoder: nn.Module acting as a point-cloud encoder.
        :param decoder: nn.Module acting as a point-cloud decoder.
        """
        super(PartAwareAutoencoder, self).__init__()
        self.encoder = encoder # Should be pointnet
        self.decoder = decoder
        if classifier is not None:
            self.classifier  =classifier
            self.seg_lossfn = nn.CrossEntropyLoss()


    def forward(self, pointclouds):
        """Forward pass of the AE
            :param pointclouds: B x N x 3
        """
        global_features = self.encoder(pointclouds).squeeze(-1)
        reconstructed = self.decoder(global_features)
        return reconstructed
    
    def train_with_part_one_epoch(self, loader, optimizer, num_classes, device="cuda"):
        self.train()
        loss_seg_meter = AverageMeter()
        loss_recon_meter =AverageMeter()
        for data in loader:
            optimizer.zero_grad()
            points = data["point_cloud"].to(device)
            part_masks = data["part_mask"].to(device)
            # Convert part_masks to one-hot
            part_masks = torch.nn.functional.one_hot(part_masks.long(), 
                                                     num_classes=num_classes).to(device).float()
            global_features = self.encoder(points).squeeze(-1)
            point_masks = self.classifier(points, global_features)
            recons = self.decoder(global_features)
            recon_loss = chamfer_loss(recons, points).mean()
            seg_loss = self.seg_lossfn(point_masks.reshape(-1,num_classes), 
                                       part_masks.reshape(-1,num_classes)).mean()
            loss = recon_loss + 0.005 * seg_loss
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
            point_clouds.append(points)
            shape_name = shape_name + data["model_name"]
            recons = self.forward(points)
            reconstructed_pcds.append(recons)
            global_features = self.encoder(points).squeeze(-1)
            point_masks = self.classifier(points, global_features)
            point_masks = torch.argmax(point_masks, dim=2) # [B, N]
            part_masks.append(point_masks)
        return torch.vstack(point_clouds), torch.vstack(reconstructed_pcds), torch.vstack(part_masks), shape_name