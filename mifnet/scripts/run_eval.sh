#!/bin/bash
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
ulimit -u unlimited

if [ $# != 2 ]
then
    echo "Usage: sh run_eval.sh [WHDLD|DLRSD] [MODEL_NAME]"
exit 1
fi

if [ ! -d $2 ]
then
    echo "error: DATA_PATH=$1 is not a directory"
exit 1
fi

export DEVICE_ID=0
dataset_idx=0

if [ $# == 2 ]
then
    if [ $1 != "WHDLD" ] && [ $1 != "DLRSD" ]
    then
        echo "error: the selected dataset is neither liberty, notredame nor yosemite"
    exit 1
    fi

    if [ $1 == "DLRSD" ]
    then
      dataset_idx=1
    fi
fi

python ../eval.py --dataset_index=$dataset_idx --model_name=$2 > ./eval.log 2>&1 &