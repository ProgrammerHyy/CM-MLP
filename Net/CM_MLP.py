import torch
import torch.nn as nn
import torch.nn.functional as F
from Net.Res2Net_v1b import res2net101_v1b_26w_4s
from Net.CrossGatingBlock import CrossGatingBlock
from utils.axial_atten import AA_kernel
from utils.conv_layer import Conv
from utils.partial_decoder import aggregation


class CM_MLP(nn.Module):
    def __init__(self, channel=32, classes=1, resolution=512):
        super().__init__()
        self.times = channel // classes
        self.to_channel = nn.Conv2d(classes, 32, (1, 1), (1, 1), 0)
        '''------ ResNet Backbone -------'''
        self.res_net = res2net101_v1b_26w_4s(pretrained=True)
        self.out_channel = classes

        res1 = resolution / 8
        res2 = res1 / 2
        res3 = res2 / 2

        '''Receptive Field Block'''
        self.rfb1_1 = Conv(256, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb2_1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb3_1 = Conv(1024, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb4_1 = Conv(2048, 32, 3, 1, padding=1, bn_acti=True)

        '''Partial Decoder'''
        self.agg1 = aggregation(out_channel=self.out_channel)

        '''MFI part'''
        self.MFI_1 = CrossGatingBlock(16, 16, 16, [res1, res1], [8, 8], [4, 4], 0.1, upsample_y=False, train_size=resolution)
        self.MFI_2 = CrossGatingBlock(16, 16, 16, [res2, res2], [4, 4], [4, 4], 0.1, upsample_y=False, train_size=resolution)
        self.MFI_3 = CrossGatingBlock(16, 16, 16, [res3, res3], [2, 2], [2, 2], 0.1, upsample_y=False, train_size=resolution)
        '''RA part'''
        self.ra1_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, self.out_channel, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, self.out_channel, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, self.out_channel, 3, 1, padding=1, bn_acti=True)
        '''----- ACRE part -----'''
        ''' Axial Attention'''
        self.aa_kernel_1 = AA_kernel(32, 32)
        self.aa_kernel_2 = AA_kernel(32, 32)
        self.aa_kernel_3 = AA_kernel(32, 32)

        self.rfb_bn_1 = Conv(32, 32, 3, 1, 1, bn_acti=True)
        self.rfb_bn_2 = Conv(32, 32, 3, 1, 1, bn_acti=True)
        self.rfb_bn_3 = Conv(32, 32, 3, 1, 1, bn_acti=True)

        '''Feature Concat'''
        self.concat1 = Conv(64, 32, 1, 1, 0, bn_acti=True)
        self.concat2 = Conv(64, 32, 1, 1, 0, bn_acti=True)
        self.concat3 = Conv(64, 32, 1, 1, 0, bn_acti=True)
        self.cor_conv1_1 = Conv(32, 32, 3, 1, 1, bn_acti=True)
        self.cor_conv1_2 = Conv(32, 32, 3, 1, 1, bn_acti=True)
        self.cor_conv2_1 = Conv(32, 32, 3, 1, 1, bn_acti=True)
        self.cor_conv2_2 = Conv(32, 32, 3, 1, 1, bn_acti=True)
        self.cor_conv3_1 = Conv(32, 32, 3, 1, 1, bn_acti=True)
        self.cor_conv3_2 = Conv(32, 32, 3, 1, 1, bn_acti=True)

    def forward(self, x):
        x = self.res_net.conv1(x)
        x = self.res_net.bn1(x)
        x = self.res_net.relu(x)
        x = self.res_net.maxpool(x)     # bs, 64, 128, 128

        '''------ low level features ------'''
        x1 = self.res_net.layer1(x)     # bs, 256, 128, 128
        x2 = self.res_net.layer2(x1)    # bs, 512, 64, 64
        x3 = self.res_net.layer3(x2)    # bs, 1024, 32, 32
        x4 = self.res_net.layer4(x3)    # bs, 2048, 16, 16
        '''------ RFB part ------'''
        x2_rfb = self.rfb2_1(x2)    # 512 --> 32
        x3_rfb = self.rfb3_1(x3)    # 1024 --> 32
        x4_rfb = self.rfb4_1(x4)    # 2048 --> 32

        x4_rfb_1, x4_rfb_2 = torch.chunk(x4_rfb, 2, dim=1)
        x3_rfb_1, x3_rfb_2 = torch.chunk(x3_rfb, 2, dim=1)
        x2_rfb_1, x2_rfb_2 = torch.chunk(x2_rfb, 2, dim=1)

        '''------ Partial Decoder ------'''

        decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb)  # bs, 1, 128, 128
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode='bilinear', align_corners=True)

        '''------ Attention Part ------'''

        '''--- --- Attention One --- ---'''
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode='bilinear', align_corners=True)
        mfi_out_1_1, mfi_out_1_2 = self.MFI_3(x4_rfb_1, x4_rfb_2)  # 32 --> 32
        mfi_out_1 = self.rfb_bn_1(torch.cat([mfi_out_1_1, mfi_out_1_2], dim=1))

        '''--- Correlation ---'''
        aa_atten_3 = self.aa_kernel_3(mfi_out_1)    # feature map
        decoder_2_sig = torch.sigmoid(decoder_2)
        decoder_2_ra = self.to_channel(decoder_2_sig)   # mask
        aa_atten_3_o = decoder_2_ra.mul(aa_atten_3)     # feature * mask
        m_decider_2_ra = 1 - decoder_2_ra       # 1 - mask
        aa_atten_3_o_m = m_decider_2_ra.mul(aa_atten_3)     # feature * (1-mask)
        aa_atten_3_o = self.cor_conv3_1(aa_atten_3_o)
        aa_atten_3_o_m = self.cor_conv3_2(aa_atten_3_o_m)
        cor_1 = aa_atten_3_o_m.mul(aa_atten_3_o)    # correlation
        aa_atten_3_o = self.concat1(torch.cat([cor_1, aa_atten_3_o], dim=1))

        ra_3 = self.ra3_conv1(aa_atten_3_o)     # 32 --> 32
        ra_3 = self.ra3_conv2(ra_3)     # 32 --> 32
        ra_3 = self.ra3_conv3(ra_3)     # 32 --> 1

        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode='bilinear', align_corners=True)
        '''--- --- Attention two --- ---'''
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear', align_corners=True)
        mfi_out_2_1, mfi_out_2_2 = self.MFI_2(x3_rfb_1, x3_rfb_2)  # 32 --> 32
        mfi_out_2 = self.rfb_bn_2(torch.cat([mfi_out_2_1, mfi_out_2_2], dim=1))

        '''--- Correlation ---'''
        decoder_3_sig = torch.sigmoid(decoder_3)
        aa_atten_2 = self.aa_kernel_2(mfi_out_2)
        decoder_3_ra = self.to_channel(decoder_3_sig)
        aa_atten_2_o = decoder_3_ra.mul(aa_atten_2)
        aa_atten_2_o_m = (1 - decoder_3_ra).mul(aa_atten_2)
        aa_atten_2_o = self.cor_conv2_1(aa_atten_2_o)
        aa_atten_2_o_m = self.cor_conv2_2(aa_atten_2_o_m)
        cor_2 = aa_atten_2_o.mul(aa_atten_2_o_m)
        aa_atten_2_o = self.concat2(torch.cat([cor_2, aa_atten_2_o], dim=1))

        ra_2 = self.ra2_conv1(aa_atten_2_o)     # 32 --> 32
        ra_2 = self.ra2_conv2(ra_2)     # 32 --> 32
        ra_2 = self.ra2_conv3(ra_2)      # 32--> 1

        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode='bilinear', align_corners=True)

        '''--- --- Attention three --- ---'''
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear', align_corners=True)
        mfi_out_3_1, mfi_out_3_2 = self.MFI_1(x2_rfb_1, x2_rfb_2)      # 32 --> 32
        mfi_out_3 = self.rfb_bn_3(torch.cat([mfi_out_3_1, mfi_out_3_2], dim=1))

        ''' --- Correlation ---'''
        decoder_4_sig = torch.sigmoid(decoder_4)
        decoder_4_ra = self.to_channel(decoder_4_sig)
        aa_atten_1 = self.aa_kernel_1(mfi_out_3)
        aa_atten_1_o = decoder_4_ra.mul(aa_atten_1)
        aa_atten_1_o_m = (1 - decoder_4_ra).mul(aa_atten_1)
        aa_atten_1_o = self.cor_conv1_1(aa_atten_1_o)
        aa_atten_1_o_m = self.cor_conv1_2(aa_atten_1_o_m)
        cor_3 = aa_atten_1_o.mul(aa_atten_1_o_m)
        aa_atten_1_o = self.concat3(torch.cat([cor_3, aa_atten_1_o], dim=1))


        ra_1 = self.ra1_conv1(aa_atten_1_o)     # 32 --> 32
        ra_1 = self.ra1_conv2(ra_1)     # 32 --> 32
        ra_1 = self.ra1_conv3(ra_1)     # 32 --> 1

        x_1 = ra_1 + decoder_4
        lateral_map_4 = F.interpolate(x_1, scale_factor=8, mode='bilinear', align_corners=True)

        return lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1

