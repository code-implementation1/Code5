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
set -e
if [ $# != 3 ] && [ $# != 4 ]
then
    echo "Usage: sh run_train.sh [RANK_TABLE_FILE] [RANK_SIZE] [DATA_PATH] [WHDLD|DLRSD]"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi

if [ $2 != 2 ] && [ $2 != 8 ]
then
  echo "error: RANK_SIZE only supports 2 and 8"
exit 1
fi

if [ ! -d $3 ]
then
    echo "error: DATA_PATH=$3 is not a directory"
exit 1
fi

data_idx=0

if [ $# == 4 ]
then
    if [ $4 != "WHDLD" ] && [ $4 != "DLRSD" ]
    then
        echo "error: the selected dataset is neither WHDLD or DLRSD"
    exit 1
    fi

    if [ $4 == "DLRSD" ]
    then
      data_idx=1
    fi
fi

ulimit -u unlimited
export RANK_SIZE=$2
DATA_PATH=$3
export DATA_PATH=${DATA_PATH}

test_dist_8pcs()
{
    PATH_RANK=$(realpath $1)
    export RANK_TABLE_FILE=${PATH_RANK}
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    PATH_RANK=$(realpath $1)
    export RANK_TABLE_FILE=${PATH_RANK}
    export RANK_SIZE=2
}
test_dist_${RANK_SIZE}pcs

export SERVER_ID=0
rank_start=$((RANK_SIZE * SERVER_ID))

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
#查看逻辑CPU的个数
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`

for((i=0; i<${RANK_SIZE}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    export DEVICE_ID=$i
    export RANK_ID=$i
    RANK_ID=$((rank_start + i))
    rm -rf device$i
    mkdir device$i
    cp -r ./train.py ./device$i
    cp -r ./src ./device$i
    cp -r ./*.yaml ./device$i
    cd ./device$i ||exit
    echo "start training for device $i"
    env > env$i.log
    taskset -c $cmdopt python ./train.py -dsi=$data_idx > train.log$i 2>&1 &
    cd ../
done
#rm -rf device0
#mkdir device0
#cp -r ./train.py ./device0
#cp -r ./src ./device0
#cp -r ./*.yaml ./device0
#cd ./device0
#export DEVICE_ID=0
#export RANK_ID=0
#echo "start training for device 0"
#env > env0.log
#python ../train.py -dsi=$data_idx > train.log$i 2>&1 &
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../
