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
import  argparse
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.MIFNet.MIF_net import BiSeNet
from src.config import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_name', type=str, default='WHDLD_19_epoch.ckpt', required=True, help='The name of ckpt')
    args = parser.parse_args()

    config = get_config()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=os.getenv('DEVICE_ID', '0'))

    ckpt_name = args.ckpt_name
    ckpt_path = config.checkpoint_path + ckpt_name

    net = BiSeNet(num_classes=config.num_classes, resnet_name=config.resnet_name)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.ones([1, 3, 256, 256]), ms.float32)
    export(net, input_arr, file_name='test_dir', file_format='MINDIR')


if __name__ == '__main__':
    main()
