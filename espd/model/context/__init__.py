#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @created 2023-01-26
# @last-modified 2023-01-27
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .branch import BranchModule
from .backbone import get_backbone
from ..cbrelu import CBRelu

class ContextModule(nn.Module):
    """Context 분석 모듈"""
    def __init__(self, backbone, cfg):
        # 모듈 초기화
        super(ContextModule, self).__init__()

        # 모듈 초기화 및 설정 확인
        self.cfg = cfg
        self.num_class = cfg['num_class']
        self.img_size = cfg['img_size']
        self.batch_size = cfg['batch_size']
        self.n = cfg['branch_num']
        self.device= cfg['device']
        self.branch_out = cfg['branch_out']

        # Backbone 모델 설정
        self.backbone = backbone

        # Branch 넣기 위한 설정
        self.gate = []
        self.head_list, self.body_list, self.tail_list = [], [], []
        self.exit_count = torch.zeros(self.n + 1)
        self.head_layer = nn.Sequential()
        self.feats = nn.ModuleList([])
        self.fetc = nn.ModuleList([])

        # Branch 모듈 설정
        self.exactly = nn.ModuleList([])

        # Train 모드 설정
        self.train_mode=True

        # Early Exit 초기화 과정 진행
        self.devour(backbone, cfg['backbone'])

    def hunt(self, module):
        """모듈 탐색"""
        for n, m in module.named_children():
            print(n, ' ', type(m).__name__)

    def bite(self, module, start=0, end=0):
        """인공 신경망을 head, feats, fetc, tail으로 분리"""
        result = []
        counter = 0
        assert end >= start
        print("start: {}, end: {}".format(start, end))
        # print("----------------------------")
        for n, m in module.named_children():
            name = type(m).__name__
            if counter >= start and counter <= end:
                if name == "Linear":
                    result.append(nn.Flatten())
                result.append(copy.deepcopy(m))
                print("idx: *\t",counter, "\t",name)
            else:
                print("idx: \t",counter,"\t",name)
            counter += 1
        return result

    def devour(self, backbone, backbone_name):
        """Backbone모델에 따라 인공신경망 설정"""
        print("Scouting Module...")
        if 'resnet' in backbone_name:
            head, head_start, head_end = (backbone, 0, 3)
            body, body_start, body_end = (backbone, 4, 7)
            tail, tail_start, tail_end = (backbone, 8, 9)
        if 'efficientnet' in backbone_name:
            e = len(backbone.features)
            head, head_start, head_end = (backbone.features,    0, 0)
            body, body_start, body_end = (backbone.features,    1, e)
            tail, tail_start, tail_end = (backbone,             1, 2)

        # Backbone 분해 진행
        self.hunt(backbone)
        print("----------------------------------------")

        # Backbone 모델 초기 Layer 진행
        print('<Head_layer>')
        self.head_list = self.bite(head, start=head_start, end=head_end)
        print("----------------------------------------")

        # Backbone 모델 중기 Layer 진행
        print('<Body_layer>')
        self.body_list = self.bite(body, start=body_start, end=body_end)
        print("----------------------------------------")

        # Backbone 모델 후기 Layer 진행
        print('<Tail Layer')
        self.tail_list = self.bite(tail, start=tail_start, end=tail_end)

        # 필요없는 메모리 청소
        self.construct()
        del self.head_list
        del self.body_list
        del self.tail_list

    def construct(self):
        """Backbone 모델 Branch와 함께 재조합"""
        id = 0
        # Input 생성
        X = torch.randn(3, 3, self.img_size, self.img_size)

        # Head Layer 생성
        self.head_layer = nn.Sequential(*self.head_list)
        X = self.head_layer(X)

        # Feats Layer Branch와 함께 생성
        print('---------------------------------------------------')
        print("split feats={}, using N={} early exits"
              .format(len(self.body_list), self.n))
        N = self.n
        if self.n == 1:
            N += 1
        div = len(self.body_list) / N
        div = int(div)
        print("divide size:", div)
        split_list = lambda test_list, \
        x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
        final_list = split_list(self.body_list, div)
        print("Constructing head-body-tail layers with early exits")
        print("<head layer>")
        print('     || ')
        for x in range(self.n):
            for y in range(len(final_list[x])):
                if y < len(final_list[x]) - 1 :
                    print('[feat layer]')
                else:
                    print('[feat layer] -> [exit #{}]'.format(x))
            print('     || ')
            self.feats.append(nn.Sequential(*final_list[x]))
            X = self.feats[x].forward(X)
            self.exactly.append(
                BranchModule(
                    X,
                    self.num_class,
                    self.branch_out,
                    self.img_size//8, id)
            )
            id += 1

        # Branch 이외에는 Fetc Layer로 생성
        x += 1
        for y in range(x, len(final_list)):
            self.fetc.append(nn.Sequential(*final_list[x]))
            print('[fect layer]')
        for fetc in self.fetc:
            X = fetc(X)
        len(self.feats) , len(self.fetc)

        # Tail Layer 생성
        self.exactly.append(
            BranchModule(
                X,
                self.num_class,
                self.branch_out,
                self.img_size//8, id)
        )
        self.tail_layer = self.exactly[-1]
        self.tail_layer.okey = True
        self.n += 1
        print('     || ')
        print("<tail layer>")
        print('---------------------------------------------------')
        print("Model Set Complete!")

    def forward(self, x):
        """Forward 진행"""
        if self.train_mode:
            # Head Layer 진행
            x = self.head_layer(x)
            logits, reprs, confs, preds = [], [], [], []

            # Feats Layer, Branch와 함께 진행
            for i, (feat, exact) in enumerate(zip(self.feats, self.exactly)):
                x = feat(x)
                logit, repr, conf, pred = exact(x)
                logits.append(logit)
                reprs.append(repr)
                confs.append(conf)
                preds.append(pred)

            # Fetc Layer 진행
            for etc in self.fetc:
                x = etc(x)

            # Tail Layer 진행
            logit, repr, conf, pred = self.tail_layer(x)

            # Tail Layer 결과를 Stack
            logits.append(logit)
            t_logits = torch.stack(logits)
            reprs.append(repr)
            t_reprs  = torch.stack(reprs)
            confs.append(conf)
            t_confs  = torch.stack(confs)
            preds.append(pred)
            t_preds    = torch.stack(preds)

            # 진행 결과를 Return
            return t_logits, t_reprs, t_confs, t_preds

        else:
            # Test Mode일때 진행
            x = self.head_layer(x)

            # Feats Layer, Branch와 함께 진행
            for i, (feat, exact) in enumerate(zip(self.feats, self.exactly)):
                x = feat(x)
                logit, repr, conf, pred = exact(x)
                if exact.exit:
                    self.exit_count[i] += 1
                    exact.exit = False
                    return logit, repr, conf, pred

            # Fetc Layer 진행
            for etc in self.fetc:
                x = etc(x)

            # Tail Layer 진행
            logit, repr, conf, pred = self.tail_layer(x)
            self.exit_count[i+1] += 1

            # 진행 결과를 Return
            return logit, repr, conf, pred
