# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from utils.conv_layer import Conv

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, out_channel=1):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = Conv(32, 32, 3,1, padding=1)
        self.conv_upsample2 = Conv(32, 32, 3,1, padding=1)
        self.conv_upsample3 = Conv(32, 32, 3,1, padding=1)
        self.conv_upsample4 = Conv(32, 32, 3,1, padding=1)
        self.conv_upsample5 = Conv(2*32, 2*32, 3,1, padding=1)

        self.conv_concat2 = Conv(2*32, 2*32, 3,1, padding=1)
        self.conv_concat3 = Conv(3*32, 3*32, 3,1, padding=1)

        self.conv4 = Conv(3*32, 3*32, 3,1, padding=1)
        self.conv5 = nn.Conv2d(3*32, out_channel, (1, 1), padding=0)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x
