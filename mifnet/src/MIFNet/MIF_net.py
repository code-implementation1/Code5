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
import warnings
import mindspore
from mindspore import nn
from mindspore import ops
from src.MIFNet.backbone import resnet
warnings.filterwarnings(action='ignore')


class ConvBlock(mindspore.nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, pad_mode='pad', has_bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def construct(self, inputs):
        x = self.conv1(inputs)
        return self.relu(self.bn(x))


class AttentionRefinementModule(mindspore.nn.Cell):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.SequentialCell(
            # dw
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, pad_mode='valid', group=in_channels,
                      has_bias=True),
            nn.BatchNorm2d(num_features=in_channels),
            # pw-linear
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode='valid', has_bias=True),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.conv1 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.avgpool = ops.AdaptiveAvgPool2D(output_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def construct(self, inputs):
        x = self.relu(self.conv(inputs))
        feature = self.sigmoid(self.conv1(x))
        x = ops.add(ops.mul(feature, x), x)
        return x


class FeatureFusionModule(mindspore.nn.Cell):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1, pad_mode='valid')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, pad_mode='valid')
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=32, stride=32)

    def construct(self, input_1, input_2):
        x = ops.concat((input_1, input_2), axis=1)
        assert self.in_channels == x.shape[1], 'in_channels of ConvBlock should be {}'.format(x.shape[1])
        feature = self.convblock(x)
        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = ops.mul(feature, x)
        x = ops.add(x, feature)
        return x


class AttentionAdjust(mindspore.nn.Cell):

    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels1 + in_channels2, in_channels1, kernel_size=(1, 1), pad_mode='valid')
        self.conv2 = ConvBlock(in_channels1, out_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.avgpool = nn.AvgPool2d(kernel_size=64, stride=64)

    def construct(self, input1, input2):
        assert self.in_channels1 == input1.shape[1] and self.in_channels2 == input2.shape[1], \
            'Attention adjust: in_channels1 should be {}, in_channels2 should be {}'.format(input1.shape[1],
                                                                                            input2.shape[1])
        x = ops.concat((input1, input2), axis=1)
        x = self.avgpool(x)
        x = self.sigmoid(self.conv1(x))
        x = ops.mul(input1, x)
        x = self.relu(self.bn(self.conv2(x)))
        return x


class BiSeNet(mindspore.nn.Cell):


    def __init__(self, num_classes, resnet_name):
        super().__init__()

        # build context path
        self.resnet_name = resnet(name=resnet_name)

        # build attention refinement module  for resnet 101
        if resnet_name in ['resnet50', 'resnet101']:
            self.attention_refinement_module = AttentionRefinementModule(512, 256)
            self.feature_fusion_module = FeatureFusionModule(256, 1024 + 2048 + 256)
            self.attention_adjust = AttentionAdjust(in_channels1=256, in_channels2=256, out_channels=num_classes)

        elif resnet_name == 'resnet18':
            self.attention_refinement_module = AttentionRefinementModule(128, 64)
            self.feature_fusion_module = FeatureFusionModule(64, 256 + 512 + 64)
            self.attention_adjust = AttentionAdjust(in_channels1=64, in_channels2=64, out_channels=num_classes)

        else:
            print('Error: unspport resnet_name network \n')

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1, pad_mode='valid')

    def construct(self, inputs):
        down_4, down_8, down_16, down_32, tail = self.resnet_name(inputs)

        # output of spatial path
        sx = self.attention_refinement_module(down_8)

        # output of context path
        cx2 = ops.mul(down_32, tail)

        cx1 = ops.interpolate(down_16, size=sx.shape[-2:])
        cx2 = ops.interpolate(cx2, size=sx.shape[-2:])
        cx = ops.concat((cx1, cx2), axis=1)

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = ops.interpolate(result, scale_factor=2)
        result = self.attention_adjust(result, down_4)
        result = ops.interpolate(result, scale_factor=4)

        result = self.conv(result)
        return result
