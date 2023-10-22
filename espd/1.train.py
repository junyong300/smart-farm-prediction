#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @created 2023-01-26
# @last-modified 2023-01-27
#

import os
import time
import random
import datetime
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, LBFGS,Adam
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

from model.espd import ESPD
from model.dataset import SegmentationDataset
from model.utils import SegmentationMetric, SegmentationMetrics, save_checkpoint, \
    FocalLoss, ConfidenceHistogram, ECELoss, ReliabilityDiagram, read_yaml, DiceLoss

# 프로그램 추론 Random Seed 고정
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True

def parse_args():
    """Parse Argument"""
    parser = ArgumentParser(description='Edge Tomato Pest and \
        Disease Semantic Segmentation Training')
    parser.add_argument('--configs', type=str, default="./configs/base.yml")
    return parser.parse_args()

class Trainer():
    """Trainer Class containing train, validation, calibration"""
    def __init__(self, cfg):
        # Trainer 초기화
        self.cfg = cfg
        #self.device = torch.device(cfg['device'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("cuda:" , self.device)
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

        # 기본적인 Dataset 선언
        trainset = SegmentationDataset(cfg = self.cfg, split='train')
        valset   = SegmentationDataset(cfg = self.cfg, split='val')

        # Dataset 주행할 Dataloader 선언
        self.train_loader = DataLoader(
            trainset, shuffle=True,
            batch_size=cfg['batch_size'], num_workers=cfg['workers'])
        self.val_loader = DataLoader(
            valset, shuffle=False,
            batch_size=2, num_workers=16)

        # ESPD 모델 선언
        self.model = ESPD(cfg).to(self.device)

        # 최적화 함수 및 손실 함수 정의
        self.s_criterion = FocalLoss()
        self.c_criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg['lr'])
        self.metric = SegmentationMetrics()
        self.best_pred = 0.0

    def train(self):
        """Training function"""
        print("Training session start")

        # 학습 파라미터 설정
        epochs = self.cfg['epochs']
        val_per_iters = self.cfg['val_epoch']
        save_per_iters = self.cfg['save_epoch']
        start_time = time.time()
        self.model.train()

        # 학습 epoch 시작
        for e in range(1, epochs+1):
            print("epoch: ", e)
            train_loader = tqdm(self.train_loader)

            # 학습 데이터 진행
            for img, targets, classname, filename in train_loader:
                # Branch별 손실 함수 설정
                c_losses, s_losses = [], []

                # 학습 데이터 설정
                img = img.to(self.device)
                labels = classname.type(torch.LongTensor).to(self.device)
                targets = targets.type(torch.LongTensor).to(self.device)

                # 추론 진행 및 결과 정리
                outs, logits, reprs, confs, preds  = self.model(img)

                # Branch별 손실 계산
                for n, (out, logit) in enumerate(zip(outs, logits)):
                    # out = torch.argmax(out, dim=1)
                    c_losses.append(self.c_criterion(logit, labels))
                    s_losses.append(self.s_criterion(out, targets))

                # 손실 총합 계산
                c_loss = torch.sum(torch.stack(c_losses))
                s_loss = torch.sum(torch.stack(s_losses))
                loss =  c_loss + s_loss

                # 손실 역전파
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # checkpoint마다 저장
            if e % save_per_iters == 0:
                save_checkpoint(self.model, self.cfg)

            # calibration 및 validation 진행
            if e % val_per_iters == 0:
                for bn_layer in self.model.modules():
                    if isinstance(bn_layer, torch.nn.BatchNorm2d):
                        bn_layer.train(False)
                self.calibrate()
                self.validation()
                self.model.train()

        # 최종 epoch 저장
        save_checkpoint(self.model, self.cfg)

        # 총 학습시간 계산 및 출력
        total_training_time = time.time() - start_time
        total_training_str = str(
            datetime.timedelta(seconds=total_training_time))
        print(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / e))

    def calibrate(self):
        # Branch별 temperature 학습 초기화
        criterion = nn.CrossEntropyLoss()
        N = self.model.context.n
        logs , scaled = [], []
        labels = torch.zeros(0).long()
        self.model.eval()

        for n in range(N):
            self.model.context.exactly[n].temperature.requires_grad=False
            self.model.context.exactly[n].temperature[0] = 1.0
            scaled.append(torch.zeros((0)).long())
            logs.append(torch.zeros((0)).long())

        # validation 수집
        val_loader = tqdm(self.val_loader)
        for iteration, (img, targets, classname, _) in enumerate(val_loader):
            # 학습 데이터 취득
            img = img.to(self.device)
            label = classname.long()

            # 학습 추론 진행 및 데이터 취합
            with torch.no_grad():
                outs, logits, reprs, confs, preds  = self.model(img)
            labels = torch.cat((labels ,label), dim=0)

            # Branch별 Logits 취합
            for n, log in enumerate(logits):
                logs[n] = torch.cat(
                    (logs[n], log.cpu().clone().detach()), dim=0)

        # Branch별 Calibration 진행
        for n in range(N):
            # Temperature scaling 모델 학습 초기화
            temp = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
            optimizer = torch.optim.LBFGS([temp], lr=0.1, max_iter=1000)

            # Temperature scaling 모델 학습 진행
            def eval():
                optimizer.zero_grad()
                scaled[n] = logs[n] / temp
                loss = criterion(scaled[n] , labels)
                loss.backward(retain_graph=True)
                return loss
            optimizer.step(eval)

            # Temperature 적용
            m = self.model.context.exactly[n]
            m.temperature[0] = temp.item()

    def validation(self):
        """Validation session"""
        # 학습 모델 평가 추론 초기화
        torch.cuda.empty_cache()
        self.model.eval()
        metrics = []
        losses = torch.zeros([self.model.context.n + 1]).to(self.device)
        c_losses = torch.zeros([self.model.context.n + 1]).to(self.device)

        # Branch별 Segmentation 평가지수 초기화
        for n in range(self.model.context.n):
            metrics.append(SegmentationMetric(nclass=self.cfg['num_class']))
            metrics[n].reset()

        # 평가 데이터 추론
        for i, (img, target, classname, _) in enumerate(tqdm(self.val_loader)):
            # 평가 추론 데이터 취득
            img = img.to(self.device)
            target = target.type(torch.LongTensor).to(self.device)
            labels = classname.type(torch.LongTensor).to(self.device)

            # 평가 데이터 추론 진행
            with torch.no_grad():
                outs, logits, reprs, confs, preds = self.model(img)


            # 추론 데이터 Metric에 따라 추론 진행
            for n, (out, logit, metric) in enumerate(zip(outs, logits, metrics)):
                metric.update(out, target)
                # out = torch.argmax(out, dim=1)
                losses[n] += self.s_criterion(out, target)
                c_losses[n] += self.c_criterion(logit, labels)


        # Branch 별 Pixel Accuracy 및 mIOU 취득
        for n, (metric) in enumerate(metrics):
            pixel_acc, mIoU = metric.get()
            loss = losses[n] / i
            c_loss = c_losses[n] / i

            print(
                "[{}]loss: {:.3f}, c_loss:{:.3f}, pixel_acc: {:.3f}, mIoU: {:.3f}"
                .format(n, loss, c_loss, pixel_acc, mIoU))


def main():
    """Main function"""
    print("Segmentation Trainer ")
    # 프로그램 실행 Argument 취득
    args = parse_args()

    # 설정 파일 Read yaml 적용
    cfg = read_yaml(args.configs)

    # 학습 Trainer 초기화 및 학습 진행
    trainer = Trainer(cfg)
    trainer.train()

    # 학습 최종 마무리
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
