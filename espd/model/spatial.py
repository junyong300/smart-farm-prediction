#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @created 2023-01-26
# @last-modified 2023-01-27
#
import torch.nn as nn
from model.cbrelu import CBRelu

class SpatialModule(nn.Module):
    """
    Spatial Module 선언
    """
    def __init__(self, in_channels, inter_channels, out_channels):
        super(SpatialModule, self).__init__()
        self.conv1 = CBRelu(
            in_channels, inter_channels // 2, 7, 2, 3)
        self.conv2 = CBRelu(
            inter_channels // 2, inter_channels // 2 , 5, 2, 2)
        self.conv3 = CBRelu(
            inter_channels // 2, inter_channels, 5, 2, 2)
        self.conv4 = CBRelu(
            inter_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        """
        Spatial Analyis Module 진행
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
