#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @created 2023-01-26
# @last-modified 2023-01-27
#

import os
import pickle
import random
import torch
import numpy as np
from tqdm import trange
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps, ImageFilter
import yaml

class SegmentationDataset():
    """Class module for Dataset"""
    def __init__(self, cfg, split = 'train'):
        # Dataset 초기화 설정
        self.mode   = split
        self.cfg    = cfg
        self.img_size = cfg['img_size']
        ann_file    = cfg['dataset']['annotation']

        # Mode(train/val)에 따라 indices 파일 설정
        if self.mode == 'train':
            ids_file    = cfg['dataset']['train_ids']
        else:
            ids_file    = cfg['dataset']['val_ids']

        # COCO 파일 로드
        self.coco = COCO(ann_file)
        self.coco_mask = mask

        # 이미지를 Tensor로 바꿀 Transform 정의
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.4742, 0.4680, 0.4666],
                [0.2196, 0.2135, 0.2184]),
        ])

        # COCO indices 초기화
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                self.ids = pickle.load(f)
        else:
            total_indices = list(self.coco.imgs.keys())
            indices = self._preprocess(total_indices, ids_file)

            # 데이터세트 train/val 배분
            np.random.shuffle(indices)

            splits = int(np.floor(float(cfg['split']) * len(indices)))
            train_idx, val_idx = indices[splits:], indices[:splits]

            # 데이터세트 train/val indices 저장
            print(
                "spliting len(dataset)", len(indices),
                "into train", len(train_idx), "and val", len(val_idx))
            with open(cfg['dataset']['train_ids'], 'wb') as f:
                pickle.dump(train_idx, f)
            with open(cfg['dataset']['val_ids'], 'wb') as f:
                pickle.dump(val_idx, f)

            # 데이터세트 indices 파일 로드
            with open(ids_file, 'rb') as f:
                self.ids = pickle.load(f)

    def __len__(self):
        """return length of dataset"""
        return len(self.ids)

    def __getitem__(self, index):
        global cnt
        """Fetch item of dataset"""
        # 받은 index로부터 ids 가져옴
        img_id = self.ids[index]

        # 받은 ids 로부터 이미지 가지고 옴
        img_metadata = self.coco.loadImgs(img_id)[0]
        filepath = self.cfg['dataset']['images']
        filename = img_metadata['file_name']
        img = Image.open(os.path.join(filepath, filename)).convert("RGB")

        # 받은 ids 로부터 Annotation 가지고옴
        annId = self.coco.getAnnIds(imgIds=img_id)
        cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        # Annotation으로부터 Mask 생성
        mask = Image.fromarray(
            self._gen_seg_mask(
                cocotarget, img_metadata['height'], img_metadata['width']))

        # Mode(train/val)에 따라 transform 구성 맞춤
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            img, mask = self._sync_transform(img, mask)

        # Normalize, toTensor으로 전환
        if self.transform is not None:
            img = self.transform(img)

        # Classname 읽어옴, 만약 없으면 background로 정의
        if len(cocotarget) > 0:
            classname = cocotarget[0]['category_id']
        else:
            classname = 1

        # 이미지, 마스크, 객체 이름, 파일명 출력
        return img, mask, classname, filename

    def _mask_transform(self, mask):
        """Get Mask transform"""
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _gen_seg_mask(self, target, height, width):
        """Generate segmentation mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], height, width)
            mas = coco_mask.decode(rle)
            classes = instance['category_id']
            if len(mas.shape) < 3:
                mask[:, :] += (mask == 0) * (mas * classes)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(mas, axis=2)) > 0) \
                 * classes).astype(np.uint8)
        return mask

    def _val_sync_transform(self, img, mask):
        """sync transform for validation dataset"""
        outsize = self.img_size
        short_size = outsize
        width, height = img.size

        # Ratio resize
        if width > height:
            oh = short_size
            ow = int(width * oh / height)
        else:
            ow = short_size
            oh = int(height * ow / width)

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # Center crop
        width, height = img.size
        x1 = int(round(width - outsize) / 2)
        y1 = int(round(height - outsize) / 2)

        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))

        # Numpy transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # Image, Mask모두에 대한 transform
        # Random resize, crop, flip, rotate
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Random resize
        img_size = self.img_size
        short_size = img_size
        width, height = img.size
        if height > width:
            ow = short_size
            oh = int(1.0 * height * ow / width)
        else:
            oh = short_size
            ow = int(1.0 * width * oh / height)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < img_size:
            padh = img_size - oh if oh < img_size else 0
            padw = img_size - ow if ow < img_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop img_size
        width, height = img.size
        x1 = random.randint(0, width - img_size)
        y1 = random.randint(0, height - img_size)
        img = img.crop((x1, y1, x1 + img_size, y1 + img_size))
        mask = mask.crop((x1, y1, x1 + img_size, y1 + img_size))

        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        """img transform into numpy"""
        return np.array(img)

    def _mask_transform(self, mask):
        """mask transformation"""
        return np.array(mask).astype('int32')

    @property
    def pred_offset(self):
        """return offset"""
        return 0

    def _preprocess(self, ids, ids_file):
        """preprocess mask info"""
        print("Preprocessing mask, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            # cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            # img_metadata = self.coco.loadImgs(img_id)[0]
            # mask = self._gen_seg_mask(
            #     cocotarget, img_metadata['height'], img_metadata['width'])
            new_ids.append(img_id)
            # more than 1k pixels
            # if (mask > 0).sum() > 100:
            #     new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids


if __name__ == "__main__":
    """Test example for dataloader"""
    abs_path = os.path.abspath("./")
    f = open(os.path.join(abs_path,"configs/base.yml"), 'r')
    cfg = yaml.safe_load(f)
    valdataset = SegmentationDataset(cfg, split='train')
    # Create Training Loader
    train_data = DataLoader(valdataset, 4, shuffle=True, num_workers=4)
    item = valdataset.__getitem__(0)
    print("Image: ")
    print(item[0])
    print("Mask: ")
    print(item[1])
    print("Class:")
    print(item[2])
    print("basename:")
    print(item[3])
