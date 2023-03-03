#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# 2.demo.py
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @description
# @created 2023-01-26
# @last-modified 2023-01-27
#
import os
import sys
import time
import argparse
import yaml
import model
import torch
import torchvision
import shutil
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model.espd import ESPD
from model.utils import SegmentationMetric, \
batch_pix_accuracy, batch_intersection_union
from model.dataset import SegmentationDataset
import numpy as np
from model.utils import read_yaml, SegmentationMetrics


def parse_args():
    """Argument Parser function"""
    parser = argparse.ArgumentParser(
        description='Predict segmentation result from a given image')
    parser.add_argument('--configs', type=str, default="./configs/base.yml")
    parser.add_argument('--no_cuda', action='store_true')
    return parser.parse_args()

def computeIoU(y_pred_batch, y_true_batch, num_class, batch_size, features):
    return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i], num_class, batch_size, features) for i in range(len(y_true_batch))])) 

def pixelAccuracy(y_pred, y_true, num_class, batch_size, features):
    y_pred = np.argmax(np.reshape(y_pred,[num_class,batch_size,features]),axis=0)
    y_true = np.argmax(np.reshape(y_true,[num_class,batch_size,features]),axis=0)
    y_pred = y_pred * (y_true>0)

    return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)

def demo():
    """Demo function"""
    # Argument 및 기본 설정
    args = parse_args()
    cfg = read_yaml(args.configs)
    filename = '{}_{}_{}_'.format(
        cfg['dataset']['name'], cfg['segmentation'], cfg['backbone'])  
    path_filename = os.path.join(cfg['save-dir'], filename)
  
    # Device 기기 설정
    device = torch.device('cuda:0')
    if args.no_cuda:
        device = torch.device('cpu')
    
    # 결과 폴더 생성
    shutil.rmtree("./results")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # 학습된 ESPD 모델 로드
    model = ESPD(cfg)
    model.load_state_dict(torch.load( os.path.join(path_filename + '.pth')))
    model.open_exit()
    model.test_mode()
    model.eval()
   
    # 추론 테스트 할 Segmentation Dataset 및 Dataloader
    valset = SegmentationDataset(cfg = cfg, split = 'val')
    val_loader = DataLoader(dataset=valset, batch_size=1, num_workers=8)

    # 추론 성능 지표
    metric = SegmentationMetric(nclass=cfg['num_class'])
    metric.reset() 
       
    # 입력 폴더 및 출력 폴더 설정
    img_folder = cfg['dataset']['images']

    for image, targets, c, filename in val_loader:
        # 추론 데이터 입력
        image = image.view(1,3,cfg['img_size'], cfg['img_size'])
        target = targets.type(torch.LongTensor)
        if c.item() == 0:
            continue
        print("filename: "+filename[0])

        # 모델 추론 및 결과 성능 비교
        with torch.no_grad():
            outs, logits, reprs, confs, preds = model(image)
        out = torch.argmax(outs, dim=1)

        inter , union = batch_intersection_union(outs, target, cfg['num_class'])
        mIoU = (inter[union.nonzero()] / union[union.nonzero()]).mean().item()
        correct, labeled = batch_pix_accuracy(outs, target)
        pixAcc = correct / labeled

        print("mIoU: "+str(mIoU))
        print("pixAcc: "+str(pixAcc))
        


        # 추론 결과 이미지로 역전환
        tensor = image.squeeze()
        z = tensor * torch.tensor([0.2196, 0.2135, 0.2184]).view(3,1,1)
        z = z + torch.tensor([0.4742, 0.4680, 0.4666]).view(3,1,1)
        
        img = torchvision.transforms.ToPILImage(mode='RGB')(z)
        
        # 추론 결과으로부터 마스크 생성
        pred = out.squeeze(0).cpu().data.numpy()
        mask = Image.fromarray(pred.astype('uint8')).convert("L")
        mask.putpalette(cfg['colorpallete'])
        mask = mask.convert("RGBA")
        pixels = mask.getdata()
        new_pixels = []
        for pixel in pixels:
            if pixel[0] ==0 and pixel[1] == 0 and pixel[2] == 0:
                alpha = 0
            else:
                alpha = 200
            new_pixels.append((pixel[0], pixel[1], pixel[2], alpha))       
        mask.putdata(new_pixels)
        img = img.convert("RGBA")
        img.alpha_composite(mask)

        # 이미지 위에 생성된 마스크 잎이기
        predd = preds[-1]

        # 추론 레이블 텍스트 입히기
        predicted_label = cfg['dataset']['labels'][int(predd.item())]
        ImageDraw.Draw(img).text(
            (0,0), "Predicted Label: "+predicted_label, (255,255,255))
        
        # 정답 레이블 텍스트 입히기
        ground_label = cfg['dataset']['labels'][c]
        ImageDraw.Draw(img).text(
            (0,12),"Ground Label: "+ground_label, (255,255,255))

        # 신뢰도(Confidence) 텍스트 입히기
        ImageDraw.Draw(img).text(
            (0,24),"Confidence: {:.3f}".format(confs[-1].item()), (255,255,255))

        # 픽셀 정확도(Pixel Accuracy) 텍스트 입히기
        ImageDraw.Draw(img).text(
            (0,36),"Pixel Accuracy: {:.3f}".format(pixAcc), (255,255,255))
        
        # 영역 정확도(mIoU) 텍스트 입히기
        ImageDraw.Draw(img).text(
            (0,48),"mIoU: {:.3f}".format(mIoU), (255,255,255))

        # 출력 이미지 저장하기
        output_file = os.path.join("./results/", filename[0][:-4]+".png")
        img.save(output_file)
    
    print(model.context.exit_count)

if __name__=='__main__':
    torch.cuda.empty_cache()
    demo()

