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
import time
import argparse
import random
import numpy as np
from src.utils import poly_lr_scheduler
from src.data_loader import TrainDataset
from src.MIFNet.MIF_net import BiSeNet
from src.config import get_config
import pandas as pd
import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore as ms
from mindspore.context import ParallelMode
from mindspore import context, Model, Tensor, ops
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback._callback import Callback, _handle_loss
from mindspore import log as logger

ms.common.set_seed(0)
random.seed(0)
np.random.seed(0)


class Lrcube(Callback):
    def __init__(self, learning_rate_function, init_lr, dataset_size):
        super().__init__()
        self.learning_rate_function = learning_rate_function
        self.init_lr = init_lr
        self.new_lr = init_lr

        self.loss_list = []
        self.dataset_size = dataset_size
        self.last_print_time = 0
        self.loss_record = []
        self.epoch_time = 0
        self.step_time = 0
        print('dataset_size: ', dataset_size)

    def epoch_begin(self):
        """Called before each epoch beginning."""
        self.loss_record = []  # pylint: disable=W0107
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        loss_train_mean = np.mean(self.loss_record)
        print(f"Epoch time: {epoch_mseconds:5.3f} ms, "
              f"per step time: {per_step_mseconds:5.3f} ms, "
              f"avg loss: {loss_train_mean:5.5f}, ", flush=True)

        self.new_lr = self.learning_rate_function(self.init_lr, cb_params.cur_epoch_num, cb_params.epoch_num)
        ops.assign(cb_params.optimizer.learning_rate, Tensor(self.new_lr, ms.float32))
        logger.info(f'At epoch {cb_params.cur_epoch_num}, learning_rate change to {self.new_lr}')
        self.loss_list.append(loss_train_mean)

    def step_begin(self):
        """Called after each step finished."""
        self.step_time = time.time()

    def step_end(self, run_context):
        """Called after each step finished."""
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        loss = _handle_loss(cb_params.net_outputs)
        self.loss_record.append(float(loss))
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if (cb_params.cur_step_num - self.last_print_time) >= 30:
            self.last_print_time = cb_params.cur_step_num
            print(f"Epoch:[{(cb_params.cur_epoch_num - 1):3d}/{cb_params.epoch_num:3d}], "
                  f"step:[{cur_step_in_epoch:5d}/{cb_params.batch_num:5d}], "
                  f"time:{step_mseconds:5.3f} ms, "
                  f"lr:{self.new_lr:5.5f}", flush=True)

    def end(self):
        loss_list_df = pd.DataFrame(self.loss_list)
        loss_list_df.to_csv(os.path.join(config.checkpoint_path, 'loss_list.csv'), header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', type=str, default='./dataset/', help='The path of dataset')
    parser.add_argument('--ckpt_url', type=str, default='./saved_models/', help='The path of checkpoint')
    parser.add_argument('-dsi', '--dataset_index', choices=[0, 1], type=int, default=0, help='0:WHDLD 1:DLRSD')
    parser.add_argument('-n', '--name', help='You need to give this model a name.', default='MODEL')
    parser.add_argument('--epochs', help='epochs num', type=int, default=30)
    args = parser.parse_args()

    data_index = args.dataset_index
    config = get_config(data_index)

    config.data_url = args.data_url
    config.checkpoint_path = args.ckpt_url
    config.num_epochs = args.epochs

    rank_size = None
    rank_id = None

    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)

    context.set_context(device_target=config.device_target, mode=context.GRAPH_MODE)
    device_id = int(os.getenv('DEVICE_ID', '0'))

    if config.device_target == "Ascend":
        context.set_context(device_id=device_id)

    if config.is_parallel:
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=rank_size)

    dataset = ['WHDLD', 'DLRSD']
    data_path = config.data_url + config.dataset_name + '/'
    data_train_path = data_path + 'Images/'
    data_trainy_path = data_path + 'Labels/'
    print('train dataset:', config.dataset_name)
    print('Load data...')

    dataset_generator = TrainDataset(train_path=data_train_path, trainy_path=data_trainy_path,
                                     scale=config.scale, is_scale=config.is_scale)
    dataset_train = ds.GeneratorDataset(dataset_generator, column_names=['x', 'y'], column_types=[], shuffle=True,
                                        num_shards=rank_size, shard_id=rank_id)
    dataloader_train = dataset_train.batch(config.batch_size, True)

    print('Done!')
    # build model
    print('Load model...')

    network = BiSeNet(num_classes=config.num_classes, resnet_name=config.resnet_name)
    print('Done!')
    # build optimizer

    net_loss = nn.CrossEntropyLoss()
    lr = config.learning_rate
    net_opt = ms.nn.SGD(network.trainable_params(), learning_rate=lr, momentum=config.momentum,
                        weight_decay=float(config.weight_decay))
    model = Model(network, loss_fn=net_loss, optimizer=net_opt, amp_level="O2")

    lr_cb = Lrcube(poly_lr_scheduler, lr, len(dataset_generator))

    train_begin = time.time()
    model.train(config.num_epochs, dataloader_train, callbacks=lr_cb, dataset_sink_mode=False)
    train_end = time.time()
    print('train time:', train_end - train_begin)

    ms.save_checkpoint(network, os.path.join(config.checkpoint_path, config.dataset_name + '_'
                                             + args.name + '_' + str(config.num_epochs) + '.ckpt'))
