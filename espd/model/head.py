#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @created 2023-01-26
# @last-modified 2023-01-27
#
import torch.nn as nn
import torch.nn.functional as F
from model.cbrelu import CBRelu


class HeadModule(nn.Module):
    # ESPD Segmentation 추론 결과 모듈
    def __init__(self, in_channels, inter_channels, class_num, img_size):
        # ESPD Segmentation 추론 결과 모듈 초기화 설정
        super(HeadModule, self).__init__()
        self.img_size = img_size
        self.conv1 = CBRelu(in_channels, inter_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(inter_channels, class_num, 1)

    def forward(self, x):
        # ESPD Segmentation 추론 결과 모듈 진행
        x = self.conv1(x)
        x = self.conv2(x)

        # 출력 크기에 맞게 Interpolation 적용
        x = F.interpolate(
            input=x, size=self.img_size, mode='bilinear', align_corners=False)
        return x
