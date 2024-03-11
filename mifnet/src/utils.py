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
import csv
import numpy as np
import pandas as pd
import mindspore.ops as ops
from src.config import get_config

config = get_config()


def poly_lr_scheduler(init_lr, iters, max_iter=300, power=0.9):
    """
    :param init_lr  : base learning rate
    :param iters : current iteration
    :param max_iter : number of maximum iterations
    :param power    : a polymomial power
    :return         : lr
    """

    if config.is_parallel:
        if iters < 20:
            lr = init_lr * (1 - iters / (max_iter * 2)) ** power
        else:
            lr = init_lr * (1 - iters / max_iter) ** power

    else:
        lr = init_lr * (1 - iters / max_iter) ** power

    return lr


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    :param image: The one-hot format image
    :return    : A 2D array with the same width and height as the input, but
    with a depth size of 1, where each pixel value is the classified class key.
    """
    image = ops.transpose(image, (1, 2, 0))
    x = ops.Argmax(axis=-1)(image)
    return x


def get_label_info(csv_path):
    """
    :param csv_path : class dict path
    :return        : class_names（所有类别的名称）, label_values（所有类别的rgb像素值，其中每个类别的rgb像素值用一个列表表示）
    """

    _, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    # class_names中为所有类别的名称，label_values中为所有类别的rgb像素值，其中每个类别的rgb像素值用一个子列表表示
    class_names = []
    label_values = []

    # 读取csv文件,返回的是迭代类型
    with open(csv_path, 'r') as csvfile:
        # csv.reader返回一个reader对象，该对象将遍历csv文件中的行。从csv文件中读取的每一行都作为字符串列表返回。
        # next(iterator, default): default是迭代器已经到了最末端，再调用next()函数的输出值。不填参数的话，到了最末端还用next()的话会报错。
        file_reader = csv.reader(csvfile, delimiter=',')
        next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, label_values


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    :param image        : single channel array where each value represents the class key.
    :param label_values : 所有类别的rgb像素值，其中每个类别的rgb像素值用一个子列表表示
    :return                : Colour coded image for segmentation visualization
    """

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def compute_global_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)


def fast_hist(a, b, n):
    """
    混淆矩阵  产生n×n的分类统计表
    :param a: 经过flatten转换为一行输入的true label
    :param b: 经过flatten转换为一行输入的predict label
    :param n: number of classes
    :return: a为true label b为predict的混淆矩阵

    bincount()函数用于统计数组内每个非负整数的个数
    np.bincount: a = np.array([0, 1, 1, 4, 5, 5])  np.bincount(a)  array([1, 2, 0, 0, 1, 2], dtype=int64)
    如果minlength被指定，那么输出数组中bin的数量至少为它指定的数

    此处相当于是将classnum * a + b, 之后进行class_num**2数目的bincount,统计完成之后reshape为(class_num, class_num)大小
    """
    k = (a >= 0) & (a < n)  # k = [True ... True] k为掩膜（去除了255这些点（即标签图中的白色的轮廓） 目的是找出标签中需要计算的类别（去掉背景）
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,
                                                                              n)  # np.bincount计算了从0到n**2-1这n**2
    # 个数中每个数出现的次数，返回值形状(n, n)


# 计算每一类的IoU
def per_class_iu(hist):
    """
    计算miou
    :param hist : 混淆矩阵
    :return        : miou list
    """
    epsilon = 1e-5
    return (np.diag(hist) + epsilon) / (
            hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)  # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)


def cal_miou(iou_list, csv_path):
    """
    依据class dict，为iou list中的每个iou依次赋上对应class name
    :param iou_list: list, iou list
    :param csv_path : class_dict
    :return         : iou_dict(dict, 每一项为class name: iou）与 miou
    """
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    iou_dict = {}
    cnt = 0
    for _, row in ann.iterrows():
        label_name = row['name']
        iou_dict[label_name] = iou_list[cnt]
        cnt += 1
    return iou_dict, np.mean(iou_list)
