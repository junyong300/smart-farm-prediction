'''
train-validation pickle파일을 text로 변환
'''
from argparse import ArgumentParser
import os
import pickle
from model.dataset import SegmentationDataset
from model.utils import read_yaml
from torch.utils.data import DataLoader

def parse_args():
    """Parse Argument"""
    parser = ArgumentParser(description='Edge Tomato Pest and \
        Disease Semantic Segmentation Training')
    parser.add_argument('--configs', type=str, default="./configs/base.yml")
    return parser.parse_args()


args = parse_args()
cfg = read_yaml(args.configs)

trainset = SegmentationDataset(cfg = cfg, split = 'train')
train_loader = DataLoader(dataset=trainset, batch_size=1, num_workers=32)
with open("train_list.txt", 'w') as f:
    for image, targets, c, filename in train_loader:
        label = c.item()
        f.write(F'{label} {filename[0]}\n')

valset = SegmentationDataset(cfg = cfg, split = 'val')
val_loader = DataLoader(dataset=valset, batch_size=1, num_workers=32)
with open("val_list.txt", 'w') as f:
    for image, targets, c, filename in val_loader:
        label = c.item()
        if label <= 1:
            continue
        f.write(F'{label} {filename[0]}\n')

