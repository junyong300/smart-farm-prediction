#
# @copyright Electronics and Telecommunications Research Institute (ETRI) All Rights Reserved.
# @author Junyong Park/junyong.park@etri.re.kr, Jong-Ryul Lee, Yong-Hyuk Moon
# @description
# @created 2023-01-26
# @last-modified 2023-01-27
#

from cProfile import label
from email.mime import image
import os
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
# from model.utils import
from model.utils import read_yaml
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from torch.utils.mobile_optimizer import optimize_for_mobile

import numpy as  np
import yaml
from model.espd import ESPD
import argparse
import onnx
from model.dataset import SegmentationDataset
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import resnet18
from model.utils import SegmentationMetric, \
batch_pix_accuracy, batch_intersection_union
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

def parse_args():
    """Argument Parser function"""
    parser = argparse.ArgumentParser(
        description='Predict segmentation result from a given image')
    parser.add_argument('--configs', type=str, default="./configs/base.yml")
    return parser.parse_args()

def mobile_convert(filename):
    # Pytorch Model(.pth) Torchscript Model(.pt)로 변환
    scripted = torch.jit.load(filename+".pt")
    mobile = optimize_for_mobile(scripted)
    mobile.save(filename+"_mobile.pt")

def torchscript_convert(model, filename):
    # Pytorch Model(.pth) Torchscript Model(.pt)로 변환
    scripted = torch.jit.script(model)
    scripted.save(filename + ".pt")

def onnx_convert(filename, input):
    # Torchscript Model(.pt)를 ONNX Model(.onnx)로 변환
    model1 = torch.jit.load(filename + ".pt")
    torch.onnx.export(
        model1, input, filename+".onnx",
        opset_version=12, input_names=["input_0"],
        output_names=[
            'output_0','output_1', 'output_2', 'output_3', 'output_4'],
        verbose=False)

def tensorflow_convert(filename):
    # ONNX Model(.onnx)를 Tensorflow로 변환
    onnx_model = onnx.load(filename+".onnx")
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid")
    print("Onnx to TensorFlow Export")
    tf_rep = prepare(onnx_model, device="CPU")
    tf_rep.export_graph(filename)

def tflite_convert(filename):
    # Tensorflow Model을 TFLite(.tflite)로 변환
    print("TensorFlow to TFLite Export")
    converter = tf.lite.TFLiteConverter.from_saved_model(filename)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()
    tflite_model_path = filename+".tflite"
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

def main():
    # Argument 입력 및 설정파일 읽기
    args = parse_args()
    cfg = read_yaml(args.configs)
    SIZE = cfg['img_size']
    filename = '{}_{}_{}_'.format(
        cfg['dataset']['name'], cfg['segmentation'], cfg['backbone'])
    path_filename = os.path.join(cfg['save-dir'], filename)
    full_path_filename = path_filename + "full"
    exit_path_filename = path_filename + "exit"
    input = torch.rand((1, 3, SIZE, SIZE)).float().to("cpu")

    # 학습 모델 재구성 및 모델 로드
    print("Load Model")
    from model.context.backbone import get_backbone
    model= ESPD(cfg).to("cpu")
    model.load_state_dict(torch.load( os.path.join(path_filename + '.pth')))
    model.test_mode()
    model.eval()
    for bn_layer in model.modules():
        if isinstance(bn_layer, torch.nn.BatchNorm2d):
            bn_layer.train(False)
    model.forward(input)
    print("Script Model...")

    # Full 모델 Torchscript로 변환 및 저장
    torchscript_convert(model, full_path_filename)

    # 모바일 버전 변환
    mobile_convert(full_path_filename)

    # Torchscript 모델 onnx로 변환 및 저장
    onnx_convert(full_path_filename, input)

    # Onnx 모델을 Tensorflow로 변환 및 저장
    tensorflow_convert(full_path_filename)

    # Tensorflow 모델을 TFLite모델로 변환
    tflite_convert(full_path_filename)

    # Early Exit 모델 변환
    print("Converting Early Exit...")
    model.open_exit()

    # Early Exit 모델 Torchscript로 변환 및 저장
    torchscript_convert(model, exit_path_filename)

    # Torchscript 모델 onnx로 변환 및 저장
    onnx_convert(exit_path_filename, input)

    # Onnx 모델을 Tensorflow로 변환 및 저장
    tensorflow_convert(exit_path_filename)

    # Tensorflow 모델을 TFLite모델로 변환
    tflite_convert(exit_path_filename)





if __name__ == "__main__":
    main()
