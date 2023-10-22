#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @created 2023-01-26
# @last-modified 2023-01-27
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.efficientnet import efficientnet_b0
from model.context import ContextModule
from model.cbrelu import CBRelu
from model.spatial import SpatialModule
from model.context.backbone import get_backbone
from model.fusion import FeatureFusion
from model.head import HeadModule
import random

random.seed(0)
torch.manual_seed(0)


class ESPD(nn.Module):
    """entire segmentation module"""
    def __init__(self, cfg):
        # ESPD 분석 모듈 초기화 설정
        super(ESPD, self).__init__()
        self.cfg            = cfg
        self.num_class      = cfg['num_class']
        self.device         = cfg['device']
        self.img_size       = cfg['img_size']
        self.N              = cfg['branch_num']
        self.backbone_name  = cfg['backbone']
        self.batch_size     = cfg['batch_size']
        self.train_mode     = True

        # ESPD 공간분석 모듈 선언
        self.spatial = SpatialModule(3, cfg['spatial_mid'], cfg['spatial_out'])

        # ESPD 객체분석 모듈 선언
        backbone = get_backbone(cfg)
        self.context = ContextModule(backbone, cfg)

        # ESPD 공간/객체 분석 취합 모듈 선언
        feature_in = cfg['branch_out'] + cfg['spatial_out']
        self.fusion = FeatureFusion(feature_in, cfg['head_in'], 1)

        # ESPD Segmentation 추론 결과 모듈 선언
        self.head = HeadModule(
            cfg['head_in'], cfg['head_mid'], self.num_class, self.img_size)

    def forward(self, x):
        # ESPD 공간 분석 모듈 추론
        spat = self.spatial(x)

        # # ESPD 객체 분석 모듈 추론
        logits, reprs, confs, preds = self.context(x)
        outs = []

        # ESPD 모드에 따른 추론 형태 취합
        if self.train_mode:
            for r in reprs:
                # ESPD 공간/객체 분석 취합
                fus = self.fusion(spat, r)

                # ESPD 추론 출력 및 출력 크기 rescaling
                out = self.head(fus)
                outs.append(out)

            t_outs = torch.stack(outs)
            return t_outs, logits, reprs, confs, preds

        else:
            # ESPD Spatial/Context 분석 취합
            fus = self.fusion(spat, reprs)

            # Segmentation Head
            out = self.head(fus)

            # ESPD 출력
            return out, logits, reprs, confs, preds

    def open_exit(self):
        for m in self.context.exactly:
            m.gate = True

    def test_mode(self):
        self.train_mode = False
        self.context.train_mode = False
        self.context.batch_size = 1
        self.eval()
        for bn_layer in self.modules():
            if isinstance(bn_layer, torch.nn.BatchNorm2d):
                bn_layer.train(False)

    def train_mode(self):
        self.train_mode = True
        self.context.train_mode = True
        self.context.train_mode = self.context.cfg['batch_size']
        self.train()
        for bn_layer in self.modules():
            if isinstance(bn_layer, torch.nn.BatchNorm2d):
                bn_layer.train(True)

def parse_args():
    """Parse Argument"""
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Edge Tomato Pest and \
        Disease Semantic Segmentation Training')
    parser.add_argument('--configs', type=str, default="./configs/base.yml")
    return parser.parse_args()

if __name__ == '__main__':
    from model.utils import read_yaml
    args = parse_args()
    cfg = read_yaml(args.configs)
    model = ESPD(cfg).to(cfg['device'])
    input = torch.randn(2, 3, cfg['base_size'], cfg['base_size'])
    out = model.forward(input)
