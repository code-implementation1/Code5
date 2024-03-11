# 目录

<!-- TOC -->

- [目录](#目录)
- [MIFnet描述](#MIF描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出过程](#导出过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
            - [liberty上训练Cnet](#liberty上训练Cnet)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# MIFnet描述

## 概述

MIFnet是一种用于遥感语义分割的多源信息融合网络，其设计了用于细化低级细节特征的DIT模块，以及能以更合理的方式来融合来自不同阶段特征的AA模块，并将二者结合到编码器-解码器架构中，有效弥补了神经网络下采样过程中信息丢失导致位置信息不准确的缺陷。

## 论文

[Multi-Source Information Fusion Network for Semantic Segmentation of Remote Sensing Images](pass)

# 模型架构

MIFnet中采用了resnet主体框架，网络将resnet中被特征提取后缩小8倍的图像输入DIT模块，并将其输出与缩小16倍和32倍的图像进行上采样，共同输入FFM模块进行处理，在进行上采样后输入AA模块，最后输出一个经过类别分割的与原图相同大小的图像。

# 数据集

使用的数据集：[WHDLD](https://www.google.com/url?q=https%3A%2F%2Fnuisteducn1-my.sharepoint.com%2F%3Au%3A%2Fg%2Fpersonal%2Fzhouwx_nuist_edu_cn%2FESoA9a_7nYBEv18kPlabiAYB4sTBXbmtWppS_62nUGXSlA%3Fe%3DKaqZi2&sa=D&sntz=1&usg=AOvVaw33a8d3bVsBedpLm1e7fqtg)

- 数据集大小：包含6种类型的遥感地物类型，每张图像大小为256x256x3
    - 训练集：共3952张图像
    - 测试集：共988张图像
- 数据格式：JPEG

使用的数据集：[DLRSD](https://www.google.com/url?q=https%3A%2F%2Fnuisteducn1-my.sharepoint.com%2F%3Au%3A%2Fg%2Fpersonal%2Fzhouwx_nuist_edu_cn%2FEVjxkus-aXRGnLFxWA5K440B_k-WNNR5-BT1I6LTojuG7g%3Fe%3DrgSMHi&sa=D&sntz=1&usg=AOvVaw01PeEHjdNYw7wk51VLOO8W)

- 数据集大小：包含21种大类的图像，用17个类别进行了标记，每张图像大小为256x256x3
    - 训练集：共1660张图像
    - 测试集：共440张图像
- 数据格式：PNG

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```yaml
  # 添加数据集路径
  # 注：仅需添加数据集父目录即可，会自动识别数据集目录
  dataset_path: './dataset/'

  # 添加推理图片存储路径
  pred_path: ”./pred/“

  # 推理前添加checkpoint路径参数
  checkpoint_path: './saved_models/'
  ```

  ```python
  # 运行训练示例（WHDLD为例,当dsi参数为0时为WHDLD数据集，为1时为DLRSD数据集）
  python train.py -dsi=0 > train.log 2>&1 &

  # 运行分布式训练示例(以2卡为例)
  ./run_train.sh [RANK_TABLE_FILE] [RANK_SIZE] [DATA_PATH] [WHDLD|DLRSD]
  # example:./run_train.sh ./rank_table_2pcs.json 2 ./data WHDLD

  # 运行评估示例
  python eval.py -dsi=0 -n='mindspore' > eval.log 2>&1 &
  或
  ./run_eval.sh [WHDLD|DLRSD] [MODEL_NAME]
  # example: bash run_eval.sh WHDLD mindspore
  ```

  对于分布式训练，需要提前创建JSON格式的HCCL配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/models/tree/master/utils/hccl_tools.>

- 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://support.huaweicloud.com/modelarts/))

    - 在 ModelArts 上训练 WHDLD 数据集

      ```python
      # (1) 执行
      #          在 default_config.yaml 文件中设置 "modelArts_mode=True"
      #          在 default_config.yaml 文件中设置 其他参数
      #          在ModelArts网页上设置 "data_url='./dataset/'"
      #          在ModelArts网页上设置 "checkpoint_path='./saved_models/'"
      #          在ModelArts网页上设置 "dataset_index=0"
      #          在ModelArts网页上设置 其他参数
      # (2) 上传你的数据集到桶上
      # (3) 在ModelArts网页上设置你的代码路径为 "/mifnet"
      # (4) 在ModelArts网页上设置启动文件为 "train.py"
      # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (6) 创建训练作业
      ```

    - 在 ModelArts 上推理 WHDLD 数据集

      ```python
      # (1) 执行
      #          在 ubc_config.yaml 文件中设置 "modelArts_mode=True"
      #          在 ubc_config.yaml 文件中设置 其他参数
      #          在ModelArts网页上设置 "dataset_index=0"
      #          在ModelArts网页上设置 "eval_path='./pred'"
      #          在ModelArts网页上设置 "ckpt_path=''./saved_models/'"
      #          在ModelArts网页上设置 "model_name='checkpoint_mifnet_19epoch.ckpt'"
      #          在ModelArts网页上设置 其他参数
      # (2) 上传你的数据集到桶上
      # (3) 在ModelArts网页上设置你的代码路径为 "/mifnet"
      # (4) 在ModelArts网页上设置启动文件为 "eval.py"
      # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
      # (6) 创建训练作业
      ```

      其他数据集同理，只需修改data_index参数。

# 脚本说明

## 脚本及样例代码

```bash
├── MIFnet
    ├── scripts
    │   ├──run_train.sh             // 分布式到Ascend的shell脚本
    │   ├──run_eval.sh              // Ascend评估的shell脚本
    ├── src
    │   ├──MIFNet
    │   │   ├──backbone.py          // 网络backbone
    │   │   ├──MIF_net.py           // MIFnet架构
    │   ├──config.py                // 参数配置文件
    │   ├──data_loader.py           // 数据集加载文件
    │   ├──utils.py                 // 工具包
    ├── train.py                    // 训练脚本
    ├── eval.py                     // 评估脚本
    ├── README_CN.md                // 所有模型相关说明
    ├── default_config.yaml         // 参数配置
    ├── export.py                   // 将checkpoint文件导出到air/mindir
    ├──metrics.py                   // 模型评价指标计算
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

  ```python
    dataset_name: 'DLRSD'           #数据集名称
    num_classes: 17                 #数据集输出类型
    num_epochs: 90                  #训练代数
    batch_size: 16                  #每批次训练数量
    learning_rate: 0.025            #学习率
    momentum: 0.9                   #动量参数
    weight_decay: 5e-4              #权重衰减参数
    resnet_name: 'resnet50'         #backbone类型
    scale: [256, 256]               #数据大小格式
    is_scale: True                  #是否进行数据处理
    #path parameters
    dataset_path: './dataset/'      #数据集位置
    checkpoint_path: './saved_models/'  #模型保存位置
    pred_path: './pred/'            #推理得出图片保存路径
    #device parameters
    device_target: 'GPU'            #硬件模式
    is_parallel: True               #是否分布式计算
  ```

更多配置细节请参考配置文件`default_config.yaml`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py -dsi=0 > train.log 2>&1 &
  ```

  上述python命令将在后台运行，采用以下方式达到损失值：

  ```bash
  # train.log
  Epoch:[0/ 20], step:[100/  247], time:84.637 ms, lr:0.20000
  Epoch:[  0/ 20], step:[  200/  247], time:87.271 ms, lr:0.20000
  Epoch time: 159681.210 ms, per step time: 646.483 ms, avg loss: 0.80551,
  Epoch:[  1/ 20], step:[   53/  247], time:88.674 ms, lr:0.19098
  Epoch:[  1/ 20], step:[  153/  247], time:86.766 ms, lr:0.19098
  Epoch time: 28242.179 ms, per step time: 114.341 ms, avg loss: 0.61482,
  Epoch:[  2/ 20], step:[    6/  247], time:86.751 ms, lr:0.18191
  Epoch:[  2/ 20], step:[  106/  247], time:89.659 ms, lr:0.18191
  Epoch:[  2/ 20], step:[  206/  247], time:89.827 ms, lr:0.18191
  Epoch time: 27757.668 ms, per step time: 112.379 ms, avg loss: 0.55480
  ...
  ```

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash ./run_train.sh ./rank_table_2pcs.json 2 ./data WHDLD
  ```

  上述shell脚本将在后台运行分布训练。采用以下方式达到损失值：

  ```bash
  Epoch:[  0/ 20], step:[  100/  247], time:84.637 ms, lr:0.20000
  Epoch:[  0/ 20], step:[  200/  247], time:87.271 ms, lr:0.20000
  Epoch time: 159681.210 ms, per step time: 646.483 ms, avg loss: 0.80551,
  Epoch:[  1/ 20], step:[   53/  247], time:88.674 ms, lr:0.19098
  Epoch:[  1/ 20], step:[  153/  247], time:86.766 ms, lr:0.19098
  Epoch time: 28242.179 ms, per step time: 114.341 ms, avg loss: 0.61482,
  Epoch:[  2/ 20], step:[    6/  247], time:86.751 ms, lr:0.18191
  Epoch:[  2/ 20], step:[  106/  247], time:89.659 ms, lr:0.18191
  Epoch:[  2/ 20], step:[  206/  247], time:89.827 ms, lr:0.18191
  Epoch time: 27757.668 ms, per step time: 112.379 ms, avg loss: 0.55480
  ...
  ```

## 评估过程

### 评估

- 在Ascend环境运行时评估WHDLD数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径设置为绝对全路径，例如“username/mifnet/saved_model/checkpoint_mifnet_19epoch.ckpt”。

  ```bash
  python eval.py -dsi=0 -n='mindspore' > eval.log 2>&1 &
  OR
  bash run_eval.sh WHDLD mindspore
  ```

  上述python命令将在后台运行，通过运行以下脚本评估测试数据集的准确性：

  ```bash
  python metrics.py
  ```

## 导出过程

### 导出MindIR

```shell
python export.py --ckpt_name=[CKPT_PATH]
```

# 模型描述

## 性能

### 训练性能

#### WHDLD上训练Mifnet

|参数| Ascend 910                       |
|------------------------------|----------------------------------|
|模型版本| MIFnet                           |
|资源| Ascend 910；系统 ubuntu18.04        |
|上传日期| 2022-11-9                        |
|MindSpore版本| 1.8.1                            |
|数据集| liberty                          |
|训练参数| epoch=20, lr=0.2 batch_size = 16 |
|优化器| SGD                              |
|损失函数| CrossEntropyLoss                 |
|损失| 0.271                            |
|速度| 105毫秒/步                          |
|总时长| 1p:10min 8p:3min

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/models)。
