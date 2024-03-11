# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from src.utils import compute_global_accuracy, per_class_iu
from src.config import get_config


def file_name(file_dir):
    """
    :param file_dir : 文件名
    :return         : list, 文件夹内所有的图片名称
    """
    L = []
    for _, _, files in os.walk(file_dir):
        # print('root:', root)
        for name in files:
            L.append(name)
    return L


def infer(true_url, pred_url, num_class):
    image_sets1 = file_name(pred_url)
    image_sets2 = file_name(true_url)
    img_num = len(image_sets1)

    pa = []
    hist = np.zeros((num_class, num_class))

    for i in range(img_num):
        img1 = Image.open(pred_url + image_sets1[i])  # 预测图
        img2 = Image.open(true_url + image_sets2[i])  # 真实类标
        img1 = np.array(img1)
        img2 = np.array(img2)
        img_pred = img1.flatten()
        img_true = img2.flatten()
        a = compute_global_accuracy(img1, img2)   # a: 每张图的分割正确率（OA）
        pa.append(a)
        # 混淆矩阵与指标计算
        confu = confusion_matrix(img_true, img_pred, labels=[i for i in range(num_class)])
        hist += confu
    # pa列表中存的是每张图的OA
    print(hist)
    iou_list = per_class_iu(hist)
    miou = np.mean(iou_list)
    mpa = np.mean(pa)

    return miou, iou_list, mpa


class Metric:
    def __init__(self):
        self.epsilon = 1e-5


if __name__ == '__main__':
    print(os.getcwd())
    data_index = 0
    config = get_config(data_index)
    metric = Metric()

    if data_index == 0:
        print('WHDLD:')
        data_num_class = 6
        data_pred_url = os.getcwd() + config.pred_path + 'msWHDLD/'
        data_true_url = os.getcwd() + config.data_url + 'WHDLD/val_Labels/'

    else:
        print('DLRSD:')
        data_num_class = 17
        data_pred_url = os.getcwd() + config.pred_path + 'msDLRSD/'
        data_true_url = os.getcwd() + config.data_url + 'DLRSD/val_Labels/'

    pred_miou, pred_iou_list, pred_mpa = infer(data_true_url, data_pred_url, data_num_class)
    print('miou = {miou:.5f}, '
          'iou list = {iou_list},'
          'mpa = {mpa:.5f}'.format(miou=pred_miou, iou_list=pred_iou_list, mpa=pred_mpa))
