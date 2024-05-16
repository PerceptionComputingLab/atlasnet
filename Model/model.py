import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math


class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(U_Network, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 3, 1, batchnorm=bn))
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn))  # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn))  # 3
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5

        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + 2, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)
        self.conv_1x1 = self.conv_block(dim, enc_nf[-1], enc_nf[-1], kernel_size=1, padding=0)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        # Get encoder activations
        x_enc = [x]
        max_pool = getattr(nn, "MaxPool{0}d".format(self.dim))
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x = max_pool(2)(x)
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        y = self.conv_1x1(x_enc[-1])
        for i in range(3):
            y = self.dec[i](y)
            # if i != 0:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(2, 16, 3, 1, 1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, 3, 1, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(256, 512, 3, 1, 1),
            nn.MaxPool3d(2),

            nn.Conv3d(512, 3012, 3, 1, 1),

            nn.AdaptiveAvgPool3d(1),

            nn.Flatten(),

            nn.Linear(3012, 128),
            nn.ReLU(),

            nn.Linear(128, 12)
        )
        self.apply(self._init_weights)

    def forward(self, x, y):
        input_value = torch.cat((x, y), dim=1)
        return self.layer(input_value)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class AffineCOMTransform(nn.Module):
    def __init__(self, device, use_com=True):
        super(AffineCOMTransform, self).__init__()

        self.translation_m = None
        self.rotation_x = None
        self.rotation_y = None
        self.rotation_z = None
        self.rotation_m = None
        self.shearing_m = None
        self.scaling_m = None

        self.id = torch.zeros((1, 3, 4)).to(device)
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        self.id[0, 2, 2] = 1
        self.device = device

        self.use_com = use_com

    def forward(self, x, affine_para):
        # Matrix that register x to its center of mass
        id_grid = F.affine_grid(self.id, x.shape, align_corners=True)

        to_center_matrix = torch.eye(4).to(self.device)
        reversed_to_center_matrix = torch.eye(4).to(self.device)
        if self.use_com:
            x_sum = torch.sum(x)
            center_mass_x = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 0]) / x_sum
            center_mass_y = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 1]) / x_sum
            center_mass_z = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 2]) / x_sum

            to_center_matrix[0, 3] = center_mass_x
            to_center_matrix[1, 3] = center_mass_y
            to_center_matrix[2, 3] = center_mass_z
            reversed_to_center_matrix[0, 3] = -center_mass_x
            reversed_to_center_matrix[1, 3] = -center_mass_y
            reversed_to_center_matrix[2, 3] = -center_mass_z

        self.translation_m = torch.eye(4).to(self.device)
        self.rotation_x = torch.eye(4).to(self.device)
        self.rotation_y = torch.eye(4).to(self.device)
        self.rotation_z = torch.eye(4).to(self.device)
        self.rotation_m = torch.eye(4).to(self.device)
        self.shearing_m = torch.eye(4).to(self.device)
        self.scaling_m = torch.eye(4).to(self.device)

        trans_xyz = affine_para[0, 0:3]
        rotate_xyz = affine_para[0, 3:6] * math.pi
        shearing_xyz = affine_para[0, 6:9] * math.pi
        scaling_xyz = 1 + (affine_para[0, 9:12] * 0.5)

        self.translation_m[0, 3] = trans_xyz[0]
        self.translation_m[1, 3] = trans_xyz[1]
        self.translation_m[2, 3] = trans_xyz[2]
        self.scaling_m[0, 0] = scaling_xyz[0]
        self.scaling_m[1, 1] = scaling_xyz[1]
        self.scaling_m[2, 2] = scaling_xyz[2]

        self.rotation_x[1, 1] = torch.cos(rotate_xyz[0])
        self.rotation_x[1, 2] = -torch.sin(rotate_xyz[0])
        self.rotation_x[2, 1] = torch.sin(rotate_xyz[0])
        self.rotation_x[2, 2] = torch.cos(rotate_xyz[0])

        self.rotation_y[0, 0] = torch.cos(rotate_xyz[1])
        self.rotation_y[0, 2] = torch.sin(rotate_xyz[1])
        self.rotation_y[2, 0] = -torch.sin(rotate_xyz[1])
        self.rotation_y[2, 2] = torch.cos(rotate_xyz[1])

        self.rotation_z[0, 0] = torch.cos(rotate_xyz[2])
        self.rotation_z[0, 1] = -torch.sin(rotate_xyz[2])
        self.rotation_z[1, 0] = torch.sin(rotate_xyz[2])
        self.rotation_z[1, 1] = torch.cos(rotate_xyz[2])

        self.rotation_m = torch.mm(torch.mm(self.rotation_z, self.rotation_y), self.rotation_x)

        self.shearing_m[0, 1] = shearing_xyz[0]
        self.shearing_m[0, 2] = shearing_xyz[1]
        self.shearing_m[1, 2] = shearing_xyz[2]

        output_affine_m = torch.mm(to_center_matrix, torch.mm(self.shearing_m, torch.mm(self.scaling_m,
                                                                                        torch.mm(self.rotation_m,
                                                                                                 torch.mm(
                                                                                                     reversed_to_center_matrix,
                                                                                                     self.translation_m)))))
        grid = F.affine_grid(output_affine_m[0:3].unsqueeze(0), x.shape, align_corners=True)
        transformed_x = F.grid_sample(x, grid, mode='bilinear', align_corners=True)

        return transformed_x, output_affine_m[0:3].unsqueeze(0)
