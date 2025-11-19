import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

import os
import time
import json

from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.metrics import accuracy_score

import torch.nn.functional as F


class Hybrid_block(nn.Module):
    def __init__(self, input_channels, keep_track, kernel_sizes=[2, 4, 8, 16, 32, 64]):
        self.keep_track = keep_track
        super(Hybrid_block, self).__init__()

        self.hybrid_block = nn.ModuleList()

        for kernel_size in kernel_sizes:
            filter_ = torch.ones((input_channels, 1, kernel_size))
            indices_ = torch.arange(kernel_size)
            filter_[:, :, indices_ % 2 == 0] *= -1

            conv = nn.Conv1d(
                            in_channels=input_channels,
                            out_channels=1,
                            kernel_size=kernel_size,
                            padding='same',
                            bias=False
                            )
            
            with torch.no_grad():  # Ensure no gradients are computed during initialization
                conv.weight = nn.Parameter(filter_, requires_grad=False)  # Transpose dimensions to match PyTorch's format

            self.hybrid_block.append(conv)

            self.keep_track += 1

        for kernel_size in kernel_sizes:
            filter_ = torch.ones((input_channels, 1, kernel_size))
            indices_ = torch.arange(kernel_size)

            filter_[:, :, indices_ % 2 > 0] *= - 1

            conv = nn.Conv1d(
                            in_channels=1,
                            out_channels=1,
                            kernel_size=kernel_size,
                            padding='same',
                            bias=False
                            )
            with torch.no_grad():  # Ensure no gradients are computed during initialization
                conv.weight = nn.Parameter(filter_, requires_grad=False)  # Transpose dimensions to match PyTorch's format
            
            self.hybrid_block.append(conv)

            self.keep_track += 1

        for kernel_size in kernel_sizes[1:]:

            filter_ = torch.zeros((kernel_size + kernel_size // 2, input_channels, 1))
            xmash = torch.linspace(start=0, end=1, steps=kernel_size // 4 + 1)[1:].reshape((-1, 1, 1))

            filter_left = xmash**2
            filter_right = filter_left.flip(0)


            filter_[0 : kernel_size // 4] = -filter_left
            filter_[kernel_size // 4 : kernel_size // 2] = -filter_right
            filter_[kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left
            filter_[3 * kernel_size // 4 : kernel_size] = 2 * filter_right
            filter_[kernel_size : 5 * kernel_size // 4] = -filter_left
            filter_[5 * kernel_size // 4 :] = -filter_right

            conv = nn.Conv1d(
                            in_channels=input_channels,
                            out_channels=1,
                            kernel_size=kernel_size + kernel_size // 2,
                            padding='same',
                            bias=False
                            )
            with torch.no_grad():  # Ensure no gradients are computed during initialization
                conv.weight = nn.Parameter(filter_.permute(2, 1, 0), requires_grad=False)  # Transpose dimensions to match PyTorch's format
            
            self.hybrid_block.append(conv)
            
            self.keep_track += 1

        
        self.relu = nn.ReLU()

    def forward(self, x):
        
        conv_outputs = [conv(x) for conv in self.hybrid_block]
        x = torch.cat(conv_outputs, dim=1)
        x = self.relu(x)

        return x

class Inception_block(nn.Module):
    
    def __init__(
              self, 
              n_filters,
              kernel_size,
              dilation_rate=1, 
              stride=1,
              keep_track=0,
              use_hybrid_layer=True,
              use_multiplexing=True):
        super(Inception_block, self).__init__()

        self.use_hybrid_layer = use_hybrid_layer

        n_convs = 3

        kernel_size_s = [kernel_size // (2**i) for i in range(n_convs)]

        self.inception_layers  = nn.ModuleList()

        for i in range(len(kernel_size_s)):
            if n_filters[i] != 0:
                self.inception_layers.append(nn.Conv1d(
                                                in_channels=1,
                                                out_channels=n_filters[i],
                                                kernel_size=kernel_size_s[i],
                                                stride=stride,
                                                padding='same',
                                                dilation=dilation_rate,                                            
                                                bias=False
                                                ))

        self.hybrid = Hybrid_block(input_channels=1, keep_track=keep_track)
        
            
        self.bn = nn.BatchNorm1d(sum(n_filters) + 17)
        self.relu = nn.ReLU()

    def forward(self, x):
        input = x
        inception_outputs = []
        for conv_layer in self.inception_layers:
            inception_outputs.append(conv_layer(x))
        
        x = torch.cat(inception_outputs, 1)
        
        if self.use_hybrid_layer:
            h = self.hybrid(input)
            x = torch.cat([x, h], 1)
        
        x = self.bn(x)
        x = self.relu(x)

        return x

class FCN_block(nn.Module):
    
    def __init__(
            self,
            in_channels,
            kernel_size,
            n_filters,
            dilation_rate,
            stride=1,
    ):
        super(FCN_block, self).__init__()
        if in_channels != 0:
            self.depthwise_conv = nn.Conv1d(
                                    in_channels=in_channels, 
                                    out_channels=in_channels, 
                                    kernel_size=kernel_size, 
                                    stride=stride,
                                    padding='same', 
                                    dilation=dilation_rate, 
                                    groups=in_channels, 
                                    bias=False)

        if n_filters != 0:
            self.pointwise_conv = nn.Conv1d(
                                    in_channels=in_channels, 
                                    out_channels=n_filters, 
                                    kernel_size=1, 
                                    bias=False
                                    )

        self.bn = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply depthwise convolution
        depth_out = self.depthwise_conv(x)
        # Apply pointwise convolution
        x = self.pointwise_conv(depth_out)

        x = self.bn(x)
        x = self.relu(x)

        return x, depth_out

class LITE(nn.Module):

    def __init__(
            self,
            output_directory,
            length_TS,
            n_classes,
            batch_size=64,
            n_filters=[[32, 32, 32], 32, 32],
            kernel_size=41,
            n_epochs=1500,
            verbose=True,
            use_custom_filters=True,
            use_dialtion=True,
            use_multiplexing=True):
        
        super(LITE, self).__init__()

        self.output_directory = output_directory
        self.length_TS = length_TS
        self.n_classes = n_classes

        self.verbose = verbose
        self.n_filters = n_filters
        self.use_custom_filters = use_custom_filters
        self.use_dilation = use_dialtion
        self.use_multiplexing = use_multiplexing
        self.kernel_size = kernel_size - 1
        self.batch_size = batch_size 
        self.n_epochs = n_epochs

        self.keep_track = 0
        self.input_shape = (self.length_TS)
        self.inception = Inception_block(n_filters=self.n_filters[0], kernel_size=self.kernel_size, dilation_rate=1, keep_track=self.keep_track, use_hybrid_layer=self.use_custom_filters)
        
        self.kernel_size //= 2

        self.fcn_module1 = FCN_block(in_channels=sum(self.n_filters[0]) + 17, kernel_size=self.kernel_size // (2**0), n_filters=self.n_filters[1], dilation_rate=2)
        self.fcn_module2 = FCN_block(in_channels=self.n_filters[1], kernel_size=self.kernel_size // (2**1), n_filters=self.n_filters[2], dilation_rate=4)        

        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(self.n_filters[2], self.n_classes)

    
    def calculate_mask_layer_1(self, x):
        b, c, _ = x.shape
        tr_filters = 96
        
        l2_norm_feature = torch.norm(x[:, :tr_filters, :], p=2, dim=-1) # compute L2 norm from  each feature and each sample
        rowwise_sum = torch.count_nonzero(l2_norm_feature[:, :tr_filters], dim=1).unsqueeze(-1)  # count how many features are non zeros from the each sample
        tmp = l2_norm_feature.sum(dim=1, keepdims=True) # compute sum of all l2 norm features for each sample
        mu = tmp / (rowwise_sum + 1)  # compute the mean of the l2 norm feature maps from each sample
        
        
        mask = (l2_norm_feature > mu * 1.0).int() # create a mask for each sample based on the threshols value (mean)   
                
        active_feat_per_sample = torch.count_nonzero(mask, dim=1) # count of active features per sample
        pruning_feat_per_sample_per = (tr_filters - active_feat_per_sample) / tr_filters # ratio of pruned features
    
        ones_mask = torch.ones((b, 17), dtype=torch.bool, device=mask.device) # mask only with 1 values for custom filters
        full_mask = torch.cat([mask, ones_mask], dim=1).unsqueeze(-1) # final mask
                
        return full_mask, pruning_feat_per_sample_per, tr_filters - active_feat_per_sample.float().mean()
     
    def calculate_mask_layer_2_3(self, x):
        tr_filters = 32
        l2_norm_feature = torch.norm(x, p=2, dim=-1) # compute L2 norm from  each feature and each sample        
        
        rowwise_sum = torch.count_nonzero(l2_norm_feature, dim=1).unsqueeze(-1)  # count how many features are non zeros from the each sample
        tmp = l2_norm_feature.sum(dim=1, keepdims=True) # compute sum of all l2 norm features for each sample
        mu = tmp / (rowwise_sum + 1)  # compute the mean of the l2 norm feature maps from each sample
        # print('mu: ', mu)
        mask = (l2_norm_feature > mu * 1.0).int() # create a mask for each sample based on the threshols value (mean)        
        active_feat_per_sample = torch.count_nonzero(mask, dim=1) # count of active features per sample
        pruning_feat_per_sample_per = (tr_filters - active_feat_per_sample) / tr_filters # ratio of pruned features
    
        return mask.unsqueeze(-1), pruning_feat_per_sample_per, tr_filters - active_feat_per_sample.float().mean()
    
    
    
    def forward_sparsity(self, x):
        features = []
         
        # Inception layer        
        x = self.inception(x)
        features = [x]
        
        # FCN layer 1
        x, depth_out_2 = self.fcn_module1(x)
        features.append(x)
        
        # FCN layer 2
        x, depth_out_3 = self.fcn_module2(x)
        # pruning_mask_3, prune_ratio_3, active_feat_3 = self.calculate_mask_layer_2_3(x)
        # masked_x = x * pruning_mask_3
        features.append(x)
        
        
        x = self.avgpool1(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)

        # final_prune_ratio = (active_feat_1 + active_feat_2 + active_feat_3) / 160
        # print('Test set - final_prune_ratio: ', final_prune_ratio)
        
        return x, features, #final_prune_ratio.item(), [pruning_mask_1, pruning_mask_2, pruning_mask_3], [depth_out_2, depth_out_3]
    
    
    
    def forward_sparsity_2(self, x):
        b, c, _ = x.shape
        
        features = []
         
        # Inception layer        
        x = self.inception(x)
        pruning_mask_1, prune_ratio_1, active_feat_1 = self.calculate_mask_layer_1(x)
        # masked_x = x * pruning_mask_1
        features = [x]
        
        # FCN layer 1
        x, depth_out_2 = self.fcn_module1(x)
        pruning_mask_2, prune_ratio_2, active_feat_2 = self.calculate_mask_layer_2_3(x)
        # masked_x = x * pruning_mask_2        
        features.append(x)
        
        # FCN layer 2
        x, depth_out_3 = self.fcn_module2(x)
        pruning_mask_3, prune_ratio_3, active_feat_3 = self.calculate_mask_layer_2_3(x)
        # masked_x = x * pruning_mask_3
        features.append(x)
        
        
        x = self.avgpool1(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)

        final_prune_ratio = (active_feat_1 + active_feat_2 + active_feat_3) / 160
        # print('Test set - final_prune_ratio: ', final_prune_ratio)
        
        return x, features, final_prune_ratio.item(), [pruning_mask_1, pruning_mask_2, pruning_mask_3], [depth_out_2, depth_out_3]
    
    
    
    
    # def forward_sparsity_eval(self, x):
    #     b, c, _ = x.shape
        
    #     features = []
         
    #     # Inception layer        
    #     x = self.inception(x)
    #     pruning_mask_1, prune_ratio_1, active_feat_1 = self.calculate_mask_layer_1(x)
    #     masked_x = x * pruning_mask_1
    #     features = [x]
        
    #     # FCN layer 1
    #     x, depth_out = self.fcn_module1(masked_x)
    #     pruning_mask_2, prune_ratio_2, active_feat_2 = self.calculate_mask_layer_2_3(x)
    #     masked_x = x * pruning_mask_2        
    #     features.append(x)
        
    #     # FCN layer 2
    #     x, depth_out = self.fcn_module2(masked_x)
    #     pruning_mask_3, prune_ratio_3, active_feat_3 = self.calculate_mask_layer_2_3(x)
    #     masked_x = x * pruning_mask_3
    #     features.append(x)
        
    #     x = self.avgpool1(masked_x)
    #     x = torch.flatten(x, start_dim=1)

    #     x = self.fc1(x)

    #     final_prune_ratio = (active_feat_1 + active_feat_2 + active_feat_3) / 160
    #     # print('Test set - final_prune_ratio: ', final_prune_ratio)
        
    #     return x, features, final_prune_ratio.item(), [pruning_mask_1, pruning_mask_2, pruning_mask_3]
    
    
    
    
    def extract_features(self, x):
        
        features = []

        x = self.inception(x)
        features.append(x)
        
        x, depth_out_2 = self.fcn_module1(x)
        features.append(x)

        x, depth_out_3 = self.fcn_module2(x)
        features.append(x)
        # l2_norm_feature = torch.norm(x, p=2, dim=-1) # compute L2 norm from  each feature and each sample
        # print('l2 norm feature shape: ', l2_norm_feature.shape)
        
        x = self.avgpool1(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)

        return x, features, [depth_out_2, depth_out_3]

    def forward(self, x):
        
        x = self.inception(x)
        
        x, depth_out = self.fcn_module1(x)
        
        x, depth_out = self.fcn_module2(x)

        x = self.avgpool1(x)
        
        x = torch.flatten(x, start_dim=1)
        gap_out = x
        x = self.fc1(x)

        return x, gap_out
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




