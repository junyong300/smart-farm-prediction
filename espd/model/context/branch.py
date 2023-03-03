#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @created 2023-01-26
# @last-modified 2023-01-27
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..cbrelu import CBRelu

class BranchModule(nn.Module):
    """
    Branch Module based on features, 
    calculates confidence and sets exit function
    """
    def __init__(self, input, num_class,
        branch_out, img_size, id):
        # 모듈 초기화
        super().__init__()
        self.id = id
        self.num_class = num_class
        self.out_channels = branch_out
        self.img_size = img_size
        self.exit = False
        self.gate = False
        self.threshold = 0.9
        self.temperature = nn.Parameter(
            torch.Tensor([1.0]), requires_grad=False)

        # Input에 따라서 Branch 모듈 설정
        batch, channel, width, height = input.shape
        
        # Representation Vector 생성모듈 초기화
        self.transform = nn.Sequential(
            CBRelu(channel, channel // 2, 1, 1, 0),
            CBRelu(channel // 2, self.out_channels, 1, 1, 0),
        )

        # Linear Layer 초기화
        linear_int = self.out_channels * self.img_size * self.img_size

        # Early Exit Classifier 초기화
        self.classifier = nn.Sequential(
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(linear_int, num_class, bias=True),
        )

    def forward(self, x):
        # Representation Vector 생성
        x = F.interpolate(x, size = int(self.img_size), mode='bilinear')
        repr = self.transform(x) #(batch, inter, img_size, img_size)
        
        # Logits(추론 확률변수) 및 비율 조정 Temperature 계산
        logits = self.classifier(repr) #(batch, num_class)
        logits = logits / self.temperature     
        
        # 추론의 확률 분포 계산
        preds = F.softmax(logits, dim=1)
        conf, pred = torch.max(preds, 1)

        # Early Exit 설정
        if self.gate:
            # Threshold 보다 크면 Early Exit
            if conf[0] >= self.threshold:
                self.exit = True
            
            # Confidence가 0.6 이상이고, 예측값이 Background이면 Early Exit
            if conf[0]>= 0.6 and pred[0] == 0:
                self.exit = True

        # 확률분포, representation vector, confidence, 예측 Class값 출력
        return logits, repr, conf, pred