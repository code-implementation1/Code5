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
import mindspore
from mindspore.dataset.vision import Inter
from mindspore.dataset import vision, transforms
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)


class TrainDataset:
    def __init__(self, train_path, trainy_path, scale=None, is_scale=True):
        if scale is None:
            scale = [256, 256]
        self.train_path = train_path
        self.trainy_path = trainy_path
        self.train_batch = sorted(os.listdir(self.train_path))
        self.trainy_batch = sorted(os.listdir(self.trainy_path))
        self.scale = scale
        self.is_scale = is_scale

        if is_scale:

            self.data_transform = transforms.Compose([vision.Resize(scale, interpolation=Inter.BILINEAR),
                                                      vision.ToTensor()])
        else:
            self.data_transform = transforms.Compose([vision.ToTensor()])

    def __getitem__(self, idx):
        img_path = self.train_path + self.train_batch[idx]
        x = Image.open(img_path)
        x = self.data_transform(x)[0]

        label_path = self.trainy_path + self.trainy_batch[idx]
        y = Image.open(label_path)
        if self.is_scale:
            y = vision.Resize(size=self.scale)(y)

        y = mindspore.Tensor.from_numpy(np.array(y, dtype=np.int32))
        # y = np.array(y, dtype=np.int32)

        return x, y

    def __len__(self):
        return len(self.train_batch)


class TestDataset:
    def __init__(self, test_path, scale=None, is_scale=True):
        if scale is None:
            scale = [256, 256]
        self.test_path = test_path
        self.test_batch = os.listdir(self.test_path)
        self.scale = scale
        self.is_scale = is_scale

        if is_scale:
            self.data_transform = transforms.Compose([vision.Resize(scale, Inter.BILINEAR),
                                                      vision.ToTensor()])
        else:
            self.data_transform = transforms.Compose([vision.ToTensor()])

    def __getitem__(self, idx):
        img_path = self.test_path + self.test_batch[idx]
        x = Image.open(img_path)
        z = self.test_batch[idx]
        x = self.data_transform(x)[0]
        return x, z

    def __len__(self):
        return len(self.test_batch)
