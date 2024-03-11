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
import mindspore
from  mindspore import ops
from  mindvision.classification import models


class resnet18(mindspore.nn.Cell):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.backbone.conv1
        self.maxpool1 = self.features.backbone.max_pool
        self.layer1 = self.features.backbone.layer1
        self.layer2 = self.features.backbone.layer2
        self.layer3 = self.features.backbone.layer3
        self.layer4 = self.features.backbone.layer4
        # print('resnet18:done')

    def construct(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        feature1 = self.layer1(x)           # 1 / 4
        feature2 = self.layer2(feature1)    # 1 / 8
        feature3 = self.layer3(feature2)    # 1 / 16
        feature4 = self.layer4(feature3)    # 1 / 32
        tail = ops.mean(feature4, 3, keep_dims=True)
        tail = ops.mean(tail, 2, keep_dims=True)

        return feature1, feature2, feature3, feature4, tail


class resnet34(mindspore.nn.Cell):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet34(pretrained=pretrained)
        self.conv1 = self.features.backbone.conv1
        self.maxpool1 = self.features.backbone.max_pool
        self.layer1 = self.features.backbone.layer1
        self.layer2 = self.features.backbone.layer2
        self.layer3 = self.features.backbone.layer3
        self.layer4 = self.features.backbone.layer4

    def construct(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        feature1 = self.layer1(x)           # 1 / 4
        feature2 = self.layer2(feature1)    # 1 / 8
        feature3 = self.layer3(feature2)    # 1 / 16
        feature4 = self.layer4(feature3)    # 1 / 32
        tail = ops.mean(feature4, 3, keep_dims=True)
        tail = ops.mean(tail, 2, keep_dims=True)
        return feature1, feature2, feature3, feature4, tail


class resnet50(mindspore.nn.Cell):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet50(pretrained=pretrained)
        self.conv1 = self.features.backbone.conv1
        self.maxpool1 = self.features.backbone.max_pool
        self.layer1 = self.features.backbone.layer1
        self.layer2 = self.features.backbone.layer2
        self.layer3 = self.features.backbone.layer3
        self.layer4 = self.features.backbone.layer4

    def construct(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        feature1 = self.layer1(x)           # 1 / 4
        feature2 = self.layer2(feature1)    # 1 / 8
        feature3 = self.layer3(feature2)    # 1 / 16
        feature4 = self.layer4(feature3)    # 1 / 32
        tail = ops.mean(feature4, 3, keep_dims=True)
        tail = ops.mean(tail, 2, keep_dims=True)
        return feature1, feature2, feature3, feature4, tail




class resnet101(mindspore.nn.Cell):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet101(pretrained=pretrained)
        self.conv1 = self.features.backbone.conv1
        self.maxpool1 = self.features.backbone.max_pool
        self.layer1 = self.features.backbone.layer1
        self.layer2 = self.features.backbone.layer2
        self.layer3 = self.features.backbone.layer3
        self.layer4 = self.features.backbone.layer4

    def construct(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        feature1 = self.layer1(x)           # 1 / 4
        feature2 = self.layer2(feature1)    # 1 / 8
        feature3 = self.layer3(feature2)    # 1 / 16
        feature4 = self.layer4(feature3)    # 1 / 32
        tail = ops.mean(feature4, 3, keep_dims=True)
        tail = ops.mean(tail, 2, keep_dims=True)
        return feature1, feature2, feature3, feature4, tail


def resnet(name):
    model = {
        'resnet18': resnet18(pretrained=True),
        'resnet34': resnet34(pretrained=True),
        'resnet50': resnet50(pretrained=True),
        'resnet101': resnet101(pretrained=True)
    }
    return model[name]
