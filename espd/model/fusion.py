#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @created 2023-01-26
# @last-modified 2023-01-27
#

import torch
import torch.nn as nn
from model.cbrelu import CBRelu


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        # ESPD Spatial & Context Module 취합 모듈 초기화 설정
        super(FeatureFusion, self).__init__()
        self.conv1 = CBRelu(in_channels, out_channels, 1, 1, 0)

        # Attention설정
        self.atten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            CBRelu(out_channels, out_channels // 2, 1, 1, 0),
            CBRelu(out_channels // 2, out_channels, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # Feature Fusion 모듈 추론
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)

        # Attention 적용
        att = self.atten(x)
        out = x + x * att
        return out
