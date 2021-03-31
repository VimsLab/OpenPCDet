from functools import partial

import spconv
import torch.nn as nn
import torch
import numpy as np


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict

def sum_duplicates(t1, t2):
    comb_feat =  torch.cat((t1.features, t2.features), dim=0)
    comb_inds =  torch.cat((t1.indices, t2.indices), dim=0)

    new_indices, inverse_mapping, count = torch.unique(comb_inds, dim=0, sorted=True, return_inverse=True, return_counts=True)
    index_of_unique =  torch.zeros(len(new_indices), dtype=torch.long, device=comb_inds.device)
    index_of_unique[inverse_mapping] = torch.arange(len(comb_inds), dtype=torch.long, device=comb_inds.device)

    new_features = comb_feat[index_of_unique]

    comb_inds[index_of_unique] = -1
    dup_indices, dup_inverse_mapping, dup_count = torch.unique(comb_inds, dim=0, sorted=True, return_inverse=True, return_counts=True)
    
    dup_index_of_unique =  torch.zeros(len(dup_indices), dtype=torch.long, device=comb_inds.device)
    dup_index_of_unique[dup_inverse_mapping] = torch.arange(len(comb_inds), dtype=torch.long, device=comb_inds.device)
    # need to remove our dummy variable (off by one)
    dup_index_of_unique = dup_index_of_unique[1:]
    dup_feat = comb_feat[dup_index_of_unique]
    
    #print("LEN ", dup_feat.size())
    #print("SC ", (count>1).sum())
    #print("LC ", (count>1).size())
    #print("LF ", (new_features).size())
    
    new_features[count>1]  += dup_feat
    return new_indices, new_features

class VoxelBackBoneHiRes(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.alpha = 0.5 # hardcoded
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        alpha = 0.5
        level = 0
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        sf = int(1/alpha**level)
        pool = spconv.SparseMaxPool3d(sf, sf)
        self.l0_conv_input = spconv.SparseSequential(
            pool,
            spconv.SubMConv3d(input_channels, 16*sf, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16*sf),
            nn.ReLU(),
        )
        block = post_act_block
        self.l0_conv1 = spconv.SparseSequential(
            block(16*sf, 16*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.l0_conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16*sf, 32*sf, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32*sf, 32*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32*sf, 32*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        self.l0_conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*sf, 64*sf, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64*sf, 64*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64*sf, 64*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        self.l0_conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*sf, 128*sf, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(128*sf, 128*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(128*sf, 128*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        level = 1
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        sf = int(1/alpha**level)
        pool = spconv.SparseMaxPool3d(sf, sf, padding=sf//2)
        self.l1_conv_input = spconv.SparseSequential(
            pool,
            spconv.SubMConv3d(input_channels, 16*sf, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16*sf),
            nn.ReLU(),
        )
        block = post_act_block
        self.l1_conv1 = spconv.SparseSequential(
            block(16*sf, 16*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.l1_conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16*sf, 32*sf, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32*sf, 32*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32*sf, 32*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        self.l1_conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*sf, 64*sf, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64*sf, 64*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64*sf, 64*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        self.l1_conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*sf, 128*sf, 3, norm_fn=norm_fn, stride=2, padding=(1, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(128*sf, 128*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(128*sf, 128*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        level = 2
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        sf = int(1/alpha**level)
        pool = spconv.SparseMaxPool3d(sf, sf,  padding=sf//2)
        self.l2_conv_input = spconv.SparseSequential(
            pool,
            spconv.SubMConv3d(input_channels, 16*sf, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16*sf),
            nn.ReLU(),
        )
        block = post_act_block
        self.l2_conv1 = spconv.SparseSequential(
            block(16*sf, 16*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )
        self.l2_conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16*sf, 32*sf, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32*sf, 32*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32*sf, 32*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        self.l2_conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32*sf, 64*sf, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64*sf, 64*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64*sf, 64*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )
        self.l2_conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64*sf, 128*sf, 3, norm_fn=norm_fn, stride=2, padding=(1, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(128*sf, 128*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(128*sf, 128*sf, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.num_point_features = 128

        I=3
        J=4

        self.obo_0_0 = spconv.SparseConv3d(16*(2**0)*(2**0),16*2**(J-1)*2**(I-1),1)
        self.obo_0_1 = spconv.SparseConv3d(16*(2**1)*(2**0),16*2**(J-1)*2**(I-1),1)
        self.obo_0_2 = spconv.SparseConv3d(16*(2**2)*(2**0),16*2**(J-1)*2**(I-1),1)
        self.obo_0_3 = spconv.SparseConv3d(16*(2**3)*(2**0),16*2**(J-1)*2**(I-1),1)
        self.obo_1_0 = spconv.SparseConv3d(16*(2**0)*(2**1),16*2**(J-1)*2**(I-1),1)
        self.obo_1_1 = spconv.SparseConv3d(16*(2**1)*(2**1),16*2**(J-1)*2**(I-1),1)
        self.obo_1_2 = spconv.SparseConv3d(16*(2**2)*(2**1),16*2**(J-1)*2**(I-1),1)
        self.obo_1_3 = spconv.SparseConv3d(16*(2**3)*(2**1),16*2**(J-1)*2**(I-1),1)
        self.obo_2_0 = spconv.SparseConv3d(16*(2**0)*(2**2),16*2**(J-1)*2**(I-1),1)
        self.obo_2_1 = spconv.SparseConv3d(16*(2**1)*(2**2),16*2**(J-1)*2**(I-1),1)
        self.obo_2_2 = spconv.SparseConv3d(16*(2**2)*(2**2),16*2**(J-1)*2**(I-1),1)
        self.obo_2_3 = spconv.SparseConv3d(16*(2**3)*(2**2),16*2**(J-1)*2**(I-1),1)


        self.pool_2 = spconv.SparseMaxPool3d(2,stride=1)

        self.final_convs_0 = spconv.SparseConv3d(16*2**(J-1)*2**(I-1),16*2**0,1)
        self.final_convs_1 = spconv.SparseConv3d(16*2**(J-1)*2**(I-1),16*2**1,1)
        self.final_convs_2 = spconv.SparseConv3d(16*2**(J-1)*2**(I-1),16*2**2,1)
        self.final_convs_3 = spconv.SparseConv3d(16*2**(J-1)*2**(I-1),16*2**3,1)

    def forward(self, batch_dict):
        #("FORWARD?")
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor0 = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x0 = self.l0_conv_input(input_sp_tensor0)
        x0_conv1 = self.l0_conv1(x0)
        x0_conv2 = self.l0_conv2(x0_conv1)
        x0_conv3 = self.l0_conv3(x0_conv2)
        x0_conv4 = self.l0_conv4(x0_conv3)
    

        input_sp_tensor1 = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x1 = self.l1_conv_input(input_sp_tensor1)
        x1_conv1 = self.l1_conv1(x1)
        x1_conv2 = self.l1_conv2(x1_conv1)
        x1_conv3 = self.l1_conv3(x1_conv2)
        x1_conv4 = self.l1_conv4(x1_conv3)

        input_sp_tensor2 = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x2 = self.l2_conv_input(input_sp_tensor2)
        x2_conv1 = self.l2_conv1(x2)
        x2_conv2 = self.l2_conv2(x2_conv1)
        x2_conv3 = self.l2_conv3(x2_conv2)
        x2_conv4 = self.l2_conv4(x2_conv3)


        pyramids = [[x0_conv1, x0_conv2, x0_conv3, x0_conv4], 
                    [x1_conv1, x1_conv2, x1_conv3, x1_conv4], 
                    [x2_conv1, x2_conv2, x2_conv3, x2_conv4]]

        obo_convs = [[self.obo_0_0, self.obo_0_1, self.obo_0_2, self.obo_0_3], 
                     [self.obo_1_0, self.obo_1_1, self.obo_1_2, self.obo_1_3], 
                     [self.obo_2_0, self.obo_2_1, self.obo_2_2, self.obo_2_3]]

        final_convs = [self.final_convs_0,  self.final_convs_1, self.final_convs_2, self.final_convs_3]
    
        I = len(pyramids)
        J = len(pyramids[0])
        for i in list(range(I))[::-1]:
            for j in list(range(J))[::-1]: 
                #print(i, j)
                base = obo_convs[i][j](pyramids[i][j])
                if i != I-1:
                    pass
                    #base = spconv.SparseConvTensor.from_dense(nn.Upsample(size=base.dense().size()[2:], mode='nearest')(pyramids[i+1][j].dense()) +  base.dense())
                    up = pyramids[i+1][j]
                    for ii,_ in enumerate(up.spatial_shape):
                        up.spatial_shape[ii] *= 2
                    up = self.pool_2(up)
                    up.indices =  up.indices * torch.tensor([1,  2, 2, 2], device=base.features.device, dtype=torch.int32)

                    #print(base.indices.size())
                    #print(torch.unique(base.indices, dim=0).size())
                    #print(up.indices.size())
                    #print(torch.unique(up.indices, dim=0).size())

                    #print (torch.max(up.indices, dim=0))
                    #print (torch.max(base.indices, dim=0))

                    #base.features =  torch.cat((base.features, up.features), dim=0)
                    #base.indices =  torch.cat((base.indices, up.indices), dim=0)

                    comb_inds, comb_feat =  sum_duplicates(base, up)
                    base.indices = comb_inds
                    base.features = comb_feat


                    #print(base.indices.size())
                    #print(torch.unique(base.indices, dim=0).size())
                if j != J-1:
                    pass
                    #base  += nn.Upsample(size=base.size()[2:], mode='nearest')(pyramids[i][j+1])
                if i == 0:
                    base = final_convs[j](base)
                pyramids[i][j] = base

        x_conv1, x_conv2, x_conv3, x_conv4  = pyramids[0]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict