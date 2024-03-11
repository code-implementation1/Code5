# Contents

* [Contents](#contents)
    * [Mask Scoring R-CNN Description](#msrcnn-description)
        * [Model Architecture](#model-architecture)
        * [Dataset](#dataset)
    * [Environment Requirements](#environment-requirements)
    * [Quick Start](#quick-start)
        * [Prepare the model](#prepare-the-model)
        * [Run the scripts](#run-the-scripts)
    * [Script Description](#script-description)
        * [Script and Sample Code](#script-and-sample-code)
        * [Script Parameters](#script-parameters)
    * [Training](#training)
        * [Training Process](#training-process)
        * [Transfer Training](#transfer-training)
        * [Distribute training](#distribute-training)
    * [Evaluation](#evaluation)
        * [Evaluation Process](#evaluation-process)
            * [Evaluation on GPU](#evaluation-on-gpu)
        * [Evaluation result](#evaluation-result)
    * [Inference](#inference)
        * [Inference Process](#inference-process)
            * [Inference on GPU](#inference-on-gpu)
        * [Inference result](#inference-result)
   * [Model Description](#model-description)
        * [Performance](#performance)
   * [Description of Random Situation](#description-of-random-situation)
   * [ModelZoo Homepage](#modelzoo-homepage)

## [Mask Scoring R-CNN Description](#contents)

Letting a deep network be aware of the quality of its own predictions is an
interesting yet important problem. In the task of instance segmentation, the
confidence of instance classification is used as mask quality score in most
instance segmentation frameworks. However, the mask quality, quantified as the
IoU between the instance mask and its ground truth, is usually not well
correlated with classification score. Propose Mask Scoring R-CNN contains a
network block to learn the quality of the predicted instance masks. The
proposed network block takes the instance feature and the corresponding
predicted mask together to regress the mask IoU. The mask scoring strategy
calibrates the misalignment between mask quality and mask score, and improves
instance segmentation performance by prioritizing more accurate mask
predictions during COCO AP evaluation. By extensive evaluations on the COCO
dataset, Mask Scoring R-CNN brings consistent and noticeable gain with
different models, and outperforms the state-of-the-art Mask RCNN. We hope our
simple and effective approach will provide a new direction for improving
instance segmentation.

[Paper](https://arxiv.org/abs/1903.00241): Zhaojin Huang, Lichao Huang,
Yongchao Gong, Chang Huang, Xinggang Wang
Computer Vision and Pattern Recognition (CVPR), 2019 (In press).

### [Model Architecture](#contents)

**Overview of the pipeline of Mask Scoring R-CNN:**
Mask Scoring R-CNN extends Mask R-CNN framework for object instance
segmentation. The approach efficiently detects objects in an image while
simultaneously generating a high-quality segmentation mask for each instance.
There is additional block MaskIOUHead that predict IoU values between
predicted and ground truth masks. Such approach may make predicted mask scores
more correlated with the real maks prediction quality (represented as IoU).

Mask Scoring R-CNN inference pipeline:

1. Backbone (ResNet).
2. RPN + Proposal generator.
3. ROI extractor (based on ROIAlign operation).
4. Bounding box head.
5. multiclass NMS (reduce number of proposed boxes and omit objects with low
 confidence).
6. ROI extractor (based on ROIAlign operation).
7. Mask head.
8. Mask IoU head.

Mask Scoring R-CNN result training pipeline:

1. Backbone (ResNet).
2. RPN.
3. RPN Assigner+Sampler.
4. RPN Classification + Localization losses.
5. Proposal generator.
6. RCNN Assigner+Sampler.
7. ROI extractor (based on ROIAlign operation).
8. Bounding box head.
9. RCNN Classification + Localization losses.
10. ROI extractor (based on ROIAlign operation).
11. Mask head.
12. Mask loss.
13. Mask IoU head.
14. Mask IoU loss.

### [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original
paper or widely used in relevant domain/network architecture. In the following
sections, we will introduce how to run the scripts using the related dataset
below.

Dataset used: [COCO-2017](https://cocodataset.org/#download)

* Dataset size: 25.4G
    * Train: 18.0G，118287 images
    * Val: 777.1M，5000 images
    * Test: 6.2G，40670 images
    * Annotations: 474.7M, represented in 3 JSON files for each subset.
* Data format: image and json files.
    * Note: Data will be processed in dataset.py

## [Environment Requirements](#contents)

* Install [MindSpore](https://www.mindspore.cn/install/en).
* Download the dataset COCO-2017.
* Build CUDA and CPP dependencies:

```bash
bash scripts/build_custom.sh
```

* Install third-parties requirements:

```text
numpy~=1.21.2
opencv-python~=4.5.4.58
pycocotools>=2.0.5
matplotlib
seaborn
pandas
tqdm==4.64.1
```

* We use COCO-2017 as training dataset in this example by default, and you
 can also use your own datasets. Dataset structure:

```log
.
└── coco-2017
    ├── train
    │   ├── data
    │   │    ├── 000000000001.jpg
    │   │    ├── 000000000002.jpg
    │   │    └── ...
    │   └── labels.json
    ├── validation
    │   ├── data
    │   │    ├── 000000000001.jpg
    │   │    ├── 000000000002.jpg
    │   │    └── ...
    │   └── labels.json
    └── test
        ├── data
        │    ├── 000000000001.jpg
        │    ├── 000000000002.jpg
        │    └── ...
        └── labels.json
```

## [Quick Start](#contents)

### [Prepare the model](#contents)

1. Prepare yaml config file. Create file and copy content from
 `default_config.yaml` to created file.
2. Change data settings: experiment folder (`train_outputs`), image size
 settings (`img_width`, `img_height`, etc.), subsets folders (`train_dataset`,
 `val_dataset`), information about categories etc.
3. Change the backbone settings.
4. Change other training hyperparameters (learning rate, regularization,
 augmentations etc.).

### [Run the scripts](#contents)

After installing MindSpore via the official website, you can start training and
evaluation as follows:

* running on GPU

```shell
# distributed training on GPU
bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]

# standalone training on GPU
bash scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]

# run eval on GPU
bash scripts/run_eval_gpu.sh [CONFIG_PATH] [VAL_DATA] [CHECKPOINT_PATH] (OPTIONAL)[PREDICTION_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```log
ms_rcnn
├── configs
│   ├── default_config.yaml                       ## default config
│   ├── ms_rcnn_r101_caffe_fpn_1x_coco.yaml       ## config for resnet101 backbone
│   └── ms_rcnn_r50_caffe_fpn_1x_coco.yaml        ## config for resnet50 backbone
├── scripts
│   ├── build_custom.sh                           ## build C++/CUDA dependencies
│   ├── run_distribute_train_gpu.sh               ## run distribute training
│   ├── run_eval_gpu.sh                           ## run evaluation
│   ├── run_infer_gpu.sh                          ## run inference
│   └── run_standalone_train_gpu.sh               ## run training on single device
├── src
│   ├── blocks
│   │   ├── anchor_generator
│   │   │   ├── anchor_generator.py            ## Anchor generator.
│   │   │   └── __init__.py
│   │   ├── assigners_samplers
│   │   │   ├── assigner_sampler.py            ## Wrapper for assigner and sampler.
│   │   │   ├── __init__.py
│   │   │   ├── mask_assigner_sampler.py       ## Wrapper for assigner and sampler working with masks too.
│   │   │   ├── max_iou_assigner.py            ## MaxIOUAssigner.
│   │   │   └── random_sampler.py              ## Random Sampler.
│   │   ├── backbones
│   │   │   ├── __init__.py
│   │   │   └── resnet.py                      ## Resnet backbone
│   │   ├── bbox_coders
│   │   │   ├── bbox_coder.py                  ## Bounding box coder.
│   │   │   └── __init__.py
│   │   ├── bbox_heads
│   │   │   ├── __init__.py
│   │   │   └── shared_2fc_bbox_head.py        ## Bounding box head.
│   │   ├── dense_heads
│   │   │   ├── __init__.py
│   │   │   ├── proposal_generator.py          ## Proposal generator (part of RPN).
│   │   │   └── rpn.py                         ## Region Proposal Network.
│   │   ├── initialization
│   │   │   ├── initialization.py              ## Weight initialization functional.
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   ├── layers
│   │   │   ├── conv_module.py                 ## Convolution module.
│   │   │   └── __init__.py
│   │   ├── mask_heads
│   │   │   ├── fcn_mask_head.py               ## Mask head.
│   │   │   ├── __init__.py
│   │   │   └── maskiou_head.py                ## Mask IOU head.
│   │   ├── ms_rcnn.py
│   │   ├── necks
│   │   │   ├── fpn.py                         ## Feature Pyramid Network
│   │   │   └── __init__.py
│   │   ├── roi_extractors
│   │   │   ├── cpu
│   │   │   │   └── roi_align.cpp              ## ROIAlign C++ implementation
│   │   │   ├── cuda
│   │   │   │   └── roi_align_cuda_kernel.cu   ## ROIAlign CUDA implementation
│   │   │   ├── __init__.py
│   │   │   └── single_layer_roi_extractor.py  ## RoI extractor block
│   │   └── utils
│   │       ├── __init__.py
│   │       └── utils.py                       ## Common utilities for blocks
│   ├── callback.py                            ## Callbacks.
│   ├── common.py                              ## Common functional with common setting.
│   ├── dataset.py                             ## Images loading and preprocessing.
│   ├── detecteval.py                          ## DetectEval class to analyze predictions
│   ├── eval_utils.py                          ## Evaluation metrics utilities.
│   ├── __init__.py
│   ├── lr_schedule.py                         ## Optimizer settings.
│   ├── mlflow_funcs.py                        ## Model wrappers for training.
│   ├── model_utils
│   │   ├── config.py                          ## Configuration file parsing utils.
│   │   ├── device_adapter.py
│   │   ├── __init__.py
│   │   ├── local_adapter.py
│   │   └── moxing_adapter.py
│   └── network_define.py                      ## Model wrappers for training.
├── eval.py                                    ## Run models evaluation.
├── infer.py                                   ## Make predictions for models.
├── __init__.py
├── README.md
├── requirements.txt                           ## Dependencies.
└── train.py                                   ## Train script.
```

### [Script Parameters](#contents)

Major parameters in the yaml config file as follows:

```shell
train_outputs: '/data/ms_rcnn_models'
brief: 'gpu-1_1024x1024'
device_target: GPU
mode: 'graph'
# ==============================================================================
detector: 'mask_rcnn'

# backbone
backbone:
  type: 'resnet'                                                               ## Backbone type
  depth: 50                                                                    ## Backbone depth
  pretrained: '/data/backbones/resnet50.ckpt'                                  ## Path to pretrained backbone weights
  frozen_stages: 1                                                             ## Number of frozen stages
  norm_eval: True                                                              ## Whether freeze backbone batch normalizations
  num_stages: 4                                                                ## Number of output feature map
  out_indices: [0, 1, 2, 3]                                                    ## Indices of output feature map stages
  style: 'caffe'                                                               ## Layer building style (caffe and pytorch)

# neck
neck:
  fpn:
    in_channels: [256, 512, 1024, 2048]                                        ## Number of channels in input feature maps
    out_channels: 256                                                          ## Number of output channels for each feature map
    num_outs: 5                                                                ## Number of output feature maps

# rpn
rpn:
  in_channels: 256                                              ## Input channels
  feat_channels: 256                                            ## Number of channels in intermediate feature map in RPN
  num_classes: 1                                                ## Output classes number
  bbox_coder:
    target_means: [0., 0., 0., 0.]                              ## Parameter for bbox encoding (RPN targets generation)
    target_stds: [1.0, 1.0, 1.0, 1.0]                           ## Parameter for bbox encoding (RPN targets generation)
  loss_cls:
    loss_weight: 1.0                                            ## RPN classification loss weight
  loss_bbox:
    loss_weight: 1.0                                            ## RPN localization loss weight
  anchor_generator:
    scales: [8]                                                 ## Anchor scales
    strides: [4, 8, 16, 32, 64]                                 ## Anchor ratios
    ratios: [0.5, 1.0, 2.0]                                     ## Anchor strides for each feature map


bbox_head:
  in_channels: 256                                              ## Number of input channels
  fc_out_channels: 1024                                         ## Number of intermediate channels before classification
  roi_feat_size: 7                                              ## Input feature map side length
  reg_class_agnostic: False
  bbox_coder:
    target_means: [0.0, 0.0, 0.0, 0.0]                          ## Bounding box coder parameter
    target_stds: [0.1, 0.1, 0.2, 0.2]                           ## Bounding box coder parameter
  loss_cls:
    loss_weight: 1.0                                            ## Classification loss weight
  loss_bbox:
    loss_weight: 1.0                                            ## Localization loss weight

mask_head:
  num_convs: 4                                                 ## Number of convolution layers
  in_channels: 256                                             ## Number of input channels
  conv_out_channels: 256                                       ## Number of intermediate layers output channels
  loss_mask:
    loss_weight: 1.0                                           ## Segmentation loss weight

mask_iou_head:
  num_convs: 4                                                ## Number of convolution layers
  num_fcs: 2                                                  ## Number of dense layers
  roi_feat_size: 14                                           ## Input feature maps height and width
  in_channels: 256                                            ## Input channels in feature maps
  conv_out_channels: 256                                      ## Convolution layers output channels
  fc_out_channels: 1024                                       ## FC output layers
  loss_iou:
    loss_weight: 0.5                                          ## IOU prediction loss weight

# roi_align
roi:                                                                    ## RoI extractor parameters
  roi_layer: {type: 'RoIAlign', out_size: 7, sampling_ratio: 0}         ## RoI configuration
  out_channels: 256                                                     ## out roi channels
  featmap_strides: [4, 8, 16, 32]                                       ## strides for RoIAlign layer
  finest_scale: 56                                                      ## parameter that define roi level
  sample_num: 640

mask_roi:                                                               ## RoI extractor parameters for masks head
  roi_layer: {type: 'RoIAlign', out_size: 14, sampling_ratio: 0}        ## RoI configuration
  out_channels: 256                                                     ## out roi channels
  featmap_strides: [4, 8, 16, 32]                                       ## strides for RoIAlign layer
  finest_scale: 56                                                      ## parameter that define roi level
  sample_num: 128

train_cfg:
  rpn:
    assigner:
      pos_iou_thr: 0.7                                            ## IoU threshold for negative bboxes
      neg_iou_thr: 0.3                                            ## IoU threshold for positive bboxes
      min_pos_iou: 0.3                                            ## Minimum iou for a bbox to be considered as a positive bbox
      match_low_quality: True                                     ## Allow low quality matches
    sampler:
      num: 256                                                    ## Number of chosen samples
      pos_fraction: 0.5                                           ## Fraction of positive samples
      neg_pos_ub: -1                                              ## Max positive-negative samples ratio
      add_gt_as_proposals: False                                  ## Allow low quality matches
  rpn_proposal:
      nms_pre: 2000                                              ## max number of samples per level
      max_per_img: 1000                                          ## max number of output samples
      iou_threshold: 0.7                                         ## NMS threshold for proposal generator
      min_bbox_size: 0                                           ## min bboxes size
  rcnn:
    assigner:
      pos_iou_thr: 0.5                                            ## IoU threshold for negative bboxes
      neg_iou_thr: 0.5                                            ## IoU threshold for positive bboxes
      min_pos_iou: 0.5                                            ## Minimum iou for a bbox to be considered as a positive bbox
      match_low_quality: True                                     ## Allow low quality matches
    sampler:
      num: 512                                                    ## Number of chosen samples
      pos_fraction: 0.25                                          ## Fraction of positive samples
      neg_pos_ub: -1                                              ## Max positive-negative samples ratio
      add_gt_as_proposals: True                                   ## Allow low quality matches
    mask_size: 28                                                 ## Output mask size
    mask_thr_binary: 0.5                                          ## Threshold for predicted masks

test_cfg:
  rpn:
    nms_pre: 1000                                                 ## max number of samples per level
    max_per_img: 1000                                             ## max number of output samples
    iou_threshold: 0.7                                            ## NMS threshold for proposal generator
    min_bbox_size: 0                                              ## min bboxes size
  rcnn:
    score_thr: 0.05                                               ## Confidence threshold
    iou_threshold: 0.5                                            ## IOU threshold
    max_per_img: 100                                              ## Max number of output bboxes
    mask_thr_binary: 0.5                                          ## mask threshold for masks

# optimizer
opt_type: 'sgd'                                                   ## Optimizer type (sgd or adam)
lr: 0.02                                                          ## Base learning rate
min_lr: 0.0000001                                                 ## Minimum learning rate
momentum: 0.9                                                     ## Optimizer parameter
weight_decay: 0.0001                                              ## Regularization
warmup_step: 500                                                  ## Number of warmup steps
warmup_ratio: 0.001                                               ## Initial learning rate = base_lr * warmup_ratio
lr_steps: [8, 11]                                                 ## Epochs numbers when learning rate is divided by 10 (for multistep lr_type)
lr_type: 'multistep'                                              ## Learning rate scheduling type
grad_clip: 0                                                      ## Gradient clipping (set 0 to turn off)

# train
num_gts: 100                                                      ## Train batch size
batch_size: 2                                                     ## Train batch size
accumulate_step: 1                                                ## artificial batch size multiplier
test_batch_size: 1                                                ## Test batch size
loss_scale: 256                                                   ## Loss scale
epoch_size: 12                                                    ## Number of epochs
run_eval: 1                                                       ## Whether evaluation or not
eval_every: 1                                                     ## Evaluation interval
enable_graph_kernel: 0                                            ## Turn on kernel fusion
finetune: 0                                                       ## Turn on finetune (for transfer learning)
datasink: 0                                                       ## Turn on data sink mode
pre_trained: ''                                                   ## Path to pretrained model weights

#distribution training
run_distribute: 0                                                 ## Turn on distributed training
device_id: 0                                                      ##
device_num: 1                                                     ## Number of devices (only if distributed training turned on)
rank_id: 0                                                        ##

# Number of threads used to process the dataset in parallel
num_parallel_workers: 6
# Parallelize Python operations with multiple worker processes
python_multiprocessing: 0
# dataset setting
train_dataset: '/data/coco-2017/train'
val_dataset: '/data/coco-2017/validation/'
coco_classes: ['background', 'person', 'bicycle', 'car', 'motorcycle',
               'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
num_classes: 81
train_dataset_num: 0
train_dataset_divider: 0

# images
img_width: 1024                                                          ## Input images width
img_height: 1024                                                         ## Input images height
mask_divider: 2                                                          ##
divider: 64                                                              ## Automatically make width and height are dividable by divider
img_mean: [103.53, 116.28, 123.675]                                      ## Image normalization parameters
img_std: [1.0, 1.0, 1.0]                                                 ## Image normalization parameters
to_rgb: 0                                                                ## RGB or BGR
keep_ratio: 1                                                            ## Keep ratio in original images

# augmentation
flip_ratio: 0.5                                                         ## Probability of image horizontal flip

# callbacks
save_every: 100                                                         ## Save model every <n> steps
keep_checkpoint_max: 5                                                  ## Max number of saved periodical checkpoints
keep_best_checkpoints_max: 5                                            ## Max number of saved best checkpoints
 ```

## [Training](#contents)

To train the model, run `train.py`.

### [Training process](#contents)

Standalone training mode:

```bash
bash scripts/run_standalone_train_gpu.sh [CONFIG_PATH] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]
```

We need several parameters for these scripts.

* `CONFIG_PATH`: path to config file.
* `TRAIN_DATA`: path to train dataset.
* `VAL_DATA`: path to validation dataset.
* `TRAIN_OUT`: path to folder with training experiments.
* `BRIEF`: short experiment name.
* `PRETRAINED_PATH`: the path of pretrained checkpoint file, it is better
 to use absolute path.

Training result will be stored in the current path, whose folder name is "LOG".
Under this, you can find checkpoint files together with result like the
following in log.

```log
[2023-10-18T17:02:52.543] INFO: Train dataset size: 14786
[2023-10-18T17:02:52.543] INFO: Creating network...
[2023-10-18T17:02:52.956] INFO: Load backbone weights...
[2023-10-18T17:02:54.933] INFO: Number of parameters: 60789760
[2023-10-18T17:02:54.934] INFO: Device type: GPU
[2023-10-18T17:02:54.934] INFO: Creating criterion, lr and opt objects...
[2023-10-18T17:02:55.564] INFO:     Done!

[2023-10-18T17:02:59.976] WARNING: Directory already exists: /experiments/models/231018_MSRCNN_gpu-4_1024x1024_bs-2_lr-0.02_div-2/best_ckpt
[2023-10-18T17:02:59.976] WARNING: Directory already exists: /experiments/models/231018_MSRCNN_gpu-4_1024x1024_bs-2_lr-0.02_div-2/best_ckpt
[2023-10-18T17:08:27.572] INFO: epoch: 1, step: 100, loss: 3.359563,  lr: 0.004016
[2023-10-18T17:11:27.158] INFO: epoch: 1, step: 200, loss: 2.573514,  lr: 0.008012
[2023-10-18T17:14:29.258] INFO: epoch: 1, step: 300, loss: 2.502361,  lr: 0.012008
[2023-10-18T17:17:33.649] INFO: epoch: 1, step: 400, loss: 2.270288,  lr: 0.016004
...
[2023-10-19T00:17:29.843] INFO: epoch: 1, step: 14400, loss: 1.343409,  lr: 0.020000
[2023-10-19T00:20:28.143] INFO: epoch: 1, step: 14500, loss: 1.280331,  lr: 0.020000
[2023-10-19T00:23:28.032] INFO: epoch: 1, step: 14600, loss: 1.272581,  lr: 0.020000
[2023-10-19T00:26:29.806] INFO: epoch: 1, step: 14700, loss: 1.324733,  lr: 0.020000
[2023-10-19T01:12:18.672] INFO: Eval epoch time: 2596038.062 ms, per step time: 519.208 ms
2023-10-19T01:18:48.392] INFO: Result metrics for epoch 1: {'bbox_mAP': 0.16358279916365012, 'loss': 1.5375246801717821, 'seg_mAP': 0.17514182718966956}
[2023-10-19T01:18:50.212] INFO: Train epoch time: 29749928.598 ms, per step time: 2012.034 ms
[2023-10-19T01:22:16.480] INFO: epoch: 2, step: 100, loss: 1.301324,  lr: 0.020000
[2023-10-19T01:25:43.752] INFO: epoch: 2, step: 200, loss: 1.326169,  lr: 0.020000
[2023-10-19T01:29:11.448] INFO: epoch: 2, step: 300, loss: 1.310192,  lr: 0.020000
...
[2023-10-19T08:30:50.580] INFO: epoch: 2, step: 14400, loss: 1.204791,  lr: 0.020000
[2023-10-19T08:33:48.427] INFO: epoch: 2, step: 14500, loss: 1.225261,  lr: 0.020000
[2023-10-19T08:36:47.218] INFO: epoch: 2, step: 14600, loss: 1.278867,  lr: 0.020000
[2023-10-19T08:39:46.144] INFO: epoch: 2, step: 14700, loss: 1.231805,  lr: 0.020000
[2023-10-19T09:14:55.207] INFO: Eval epoch time: 1951353.219 ms, per step time: 390.271 ms
[2023-10-19T09:18:50.534] INFO: Result metrics for epoch 2: {'bbox_mAP': 0.2025139679459373, 'loss': 1.2778380052019005, 'seg_mAP': 0.20876829537504527}
[2023-10-19T09:18:52.020] INFO: Train epoch time: 28801250.453 ms, per step time: 1947.873 ms
[2023-10-19T09:21:54.762] INFO: epoch: 3, step: 100, loss: 1.240407,  lr: 0.020000
...
[2023-10-19T16:39:35.460] INFO: epoch: 3, step: 14700, loss: 1.160627,  lr: 0.020000
[2023-10-19T17:11:34.467] INFO: Eval epoch time: 1761864.067 ms, per step time: 352.373 ms
[2023-10-19T17:19:36.161] INFO: Result metrics for epoch 3: {'bbox_mAP': 0.21417936476772473, 'loss': 1.2174880106831503, 'seg_mAP': 0.22142110409964336}
[2023-10-19T17:19:37.757] INFO: Train epoch time: 28845641.964 ms, per step time: 1950.875 ms
[2023-10-19T17:22:37.871] INFO: epoch: 4, step: 100, loss: 1.180489,  lr: 0.020000
...
[2023-10-20T00:51:19.761] INFO: epoch: 4, step: 14700, loss: 1.189194,  lr: 0.020000
[2023-10-20T01:21:20.905] INFO: Eval epoch time: 1639106.320 ms, per step time: 327.821 ms
[2023-10-20T01:27:27.997] INFO: Result metrics for epoch 4: {'bbox_mAP': 0.2211869016305763, 'loss': 1.1886965869787975, 'seg_mAP': 0.23117119791347884}
[2023-10-20T01:27:30.073] INFO: Train epoch time: 29272216.041 ms, per step time: 1979.725 ms
[2023-10-20T01:30:38.736] INFO: epoch: 5, step: 100, loss: 1.102833,  lr: 0.020000
...
[2023-10-20T09:04:11.507] INFO: epoch: 5, step: 14700, loss: 1.162766,  lr: 0.020000
[2023-10-20T09:34:35.124] INFO: Eval epoch time: 1664997.991 ms, per step time: 333.000 ms
[2023-10-20T09:36:30.842] INFO: Result metrics for epoch 5: {'bbox_mAP': 0.2368496022559375, 'loss': 1.1716172063921488, 'seg_mAP': 0.24360710083879814}
[2023-10-20T09:36:33.155] INFO: Train epoch time: 29342993.119 ms, per step time: 1984.512 ms
[2023-10-20T09:39:35.145] INFO: epoch: 6, step: 100, loss: 1.182110,  lr: 0.020000
...
[2023-10-20T17:05:46.820] INFO: epoch: 6, step: 14700, loss: 1.150267,  lr: 0.020000
[2023-10-20T17:36:41.588] INFO: Eval epoch time: 1687190.546 ms, per step time: 337.438 ms
[2023-10-20T17:40:39.216] INFO: Result metrics for epoch 6: {'bbox_mAP': 0.2323328953413487, 'loss': 1.1637766443110573, 'seg_mAP': 0.2372523265832174}
[2023-10-20T17:40:40.920] INFO: Train epoch time: 29047683.017 ms, per step time: 1964.540 ms
[2023-10-20T17:43:47.247] INFO: epoch: 7, step: 100, loss: 1.222094,  lr: 0.020000
...
[2023-10-21T01:18:25.275] INFO: epoch: 7, step: 14700, loss: 1.174593,  lr: 0.020000
[2023-10-21T01:49:29.974] INFO: Eval epoch time: 1701296.476 ms, per step time: 340.259 ms
[2023-10-21T01:53:42.055] INFO: Result metrics for epoch 7: {'bbox_mAP': 0.23739080824188594, 'loss': 1.1584632571456088, 'seg_mAP': 0.23640228861225113}
[2023-10-21T01:53:44.069] INFO: Train epoch time: 29583058.171 ms, per step time: 2000.748 ms
[2023-10-21T01:56:50.634] INFO: epoch: 8, step: 100, loss: 1.141379,  lr: 0.020000
...
[2023-10-21T09:33:59.445] INFO: epoch: 8, step: 14700, loss: 1.176318,  lr: 0.020000
[2023-10-21T10:04:36.032] INFO: Eval epoch time: 1674769.747 ms, per step time: 334.954 ms
[2023-10-21T10:08:42.075] INFO: Result metrics for epoch 8: {'bbox_mAP': 0.23983684584207804, 'loss': 1.1491663503057092, 'seg_mAP': 0.24545918363917843}
[2023-10-21T10:08:44.590] INFO: Train epoch time: 29699974.007 ms, per step time: 2008.655 ms
[2023-10-21T10:12:09.440] INFO: epoch: 9, step: 100, loss: 1.098465,  lr: 0.002000
...
[2023-10-21T17:49:38.031] INFO: epoch: 9, step: 14700, loss: 1.009663,  lr: 0.002000
[2023-10-21T18:19:07.211] INFO: Eval epoch time: 1604670.229 ms, per step time: 320.934 ms
[2023-10-21T18:25:07.211] INFO: Result metrics for epoch 9: {'bbox_mAP': 0.31532773398657704, 'loss': 1.0096940788615887, 'seg_mAP': 0.302751502119639}
[2023-10-21T18:25:09.261] INFO: Train epoch time: 29784056.698 ms, per step time: 2014.342 ms
[2023-10-21T18:28:34.965] INFO: epoch: 10, step: 100, loss: 1.043987,  lr: 0.002000
...
[2023-10-22T02:08:48.391] INFO: epoch: 10, step: 14700, loss: 1.004963,  lr: 0.002000
[2023-10-22T02:38:25.091] INFO: Eval epoch time: 1615252.810 ms, per step time: 323.051 ms
[2023-10-22T02:40:25.690] INFO: Result metrics for epoch 10: {'bbox_mAP': 0.32090214228894937, 'loss': 0.9871455143230639, 'seg_mAP': 0.3055546761356429}
[2023-10-22T02:40:28.443] INFO: Train epoch time: 29718576.473 ms, per step time: 2009.913 ms
[2023-10-22T02:43:46.231] INFO: epoch: 11, step: 100, loss: 0.972540,  lr: 0.002000
...
[2023-10-22T10:30:58.806] INFO: epoch: 11, step: 14700, loss: 0.982379,  lr: 0.002000
[2023-10-22T10:59:46.581] INFO: Eval epoch time: 1567532.710 ms, per step time: 313.507 ms
[2023-10-22T11:01:45.900] INFO: Result metrics for epoch 11: {'bbox_mAP': 0.32304703136317914, 'loss': 0.9795711686748779, 'seg_mAP': 0.30816876295733}
[2023-10-22T11:01:49.136] INFO: Train epoch time: 30080103.401 ms, per step time: 2034.364 ms
[2023-10-22T11:05:31.902] INFO: epoch: 12, step: 100, loss: 1.009295,  lr: 0.000200
...
[2023-10-22T18:43:47.546] INFO: epoch: 12, step: 14700, loss: 0.968011,  lr: 0.000200
[2023-10-22T19:12:43.340] INFO: Eval epoch time: 1574634.339 ms, per step time: 314.927 ms
[2023-10-22T19:14:42.607] INFO: Result metrics for epoch 12: {'bbox_mAP': 0.33233140228812175, 'loss': 0.94968860102602, 'seg_mAP': 0.31512801454461914}
[2023-10-22T19:14:45.563] INFO: Train epoch time: 29575796.956 ms, per step time: 2000.257 ms
[2023-10-22T19:18:18.771] INFO: epoch: 13, step: 100, loss: 1.010170,  lr: 0.000200
...
[2023-10-23T03:01:36.134] INFO: epoch: 13, step: 14700, loss: 0.936117,  lr: 0.000200
[2023-10-23T03:30:48.827] INFO: Eval epoch time: 1559964.662 ms, per step time: 311.993 ms
[2023-10-23T03:32:48.976] INFO: Result metrics for epoch 13: {'bbox_mAP': 0.33329894448340347, 'loss': 0.951012151178875, 'seg_mAP': 0.3153160615656784}
[2023-10-23T03:32:51.759] INFO: Train epoch time: 29885561.862 ms, per step time: 2021.207 ms
```

### [Transfer Training](#contents)

You can train your own model based on either pretrained classification model
or pretrained detection model. You can perform transfer training by following
steps.

1. Prepare your dataset.
2. Change configuration YAML file according to your own dataset, especially the
 change `num_classes` value and `coco_classes` list.
3. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by
 `pretrained` argument. Transfer training means a new training job, so just set
 `finetune` 1.
4. Run training.

### [Distribute training](#contents)

Distribute training mode (OpenMPI must be installed):

```shell
bash scripts/run_distribute_train_gpu.sh [CONFIG_PATH] [DEVICE_NUM] [TRAIN_DATA] [VAL_DATA] [TRAIN_OUT] [BRIEF] (OPTIONAL)[PRETRAINED_PATH]
```

We need several parameters for this script:

* `CONFIG_PATH`: path to config file.
* `DEVICE_NUM`: number of devices.
* `TRAIN_DATA`: path to train dataset.
* `VAL_DATA`: path to validation dataset.
* `TRAIN_OUT`: path to folder with training experiments.
* `BRIEF`: short experiment name.
* `PRETRAINED_PATH`: the path of pretrained checkpoint file, it is better
 to use absolute path.

## [Evaluation](#contents)

### [Evaluation process](#contents)

#### [Evaluation on GPU](#contents)

```shell
bash scripts/run_eval_gpu.sh [CONFIG_PATH] [VAL_DATA] [CHECKPOINT_PATH] (Optional)[PREDICTION_PATH]
```

We need four parameters for this script.

* `CONFIG_PATH`: path to config file.
* `VAL_DATA`: the absolute path for dataset subset (validation).
* `CHECKPOINT_PATH`: path to checkpoint.
* `PREDICTION_PATH`: path to file with predictions JSON file (predictions may
 be saved to this file and loaded after).

> checkpoint can be produced in training process.

### [Evaluation result](#contents)

Result for GPU:

```log
100%|██████████| 5000/5000 [42:15<00:00,  1.97it/s]
Loading and preparing results...
DONE (t=0.46s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=20.77s).
Accumulating evaluation results...
DONE (t=2.99s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.379
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.583
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.410
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.201
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.507
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.317
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.307
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.676
Bbox eval result: 0.37853318252822526
Loading and preparing results...
DONE (t=1.08s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=23.58s).
Accumulating evaluation results...
DONE (t=2.96s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.552
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.383
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.151
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.297
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.640
Segmentation eval result: 0.3562784036195482

Evaluation done!

Done!
Time taken: 2599 seconds
```

## [Inference](#contents)

### [Inference process](#contents)

#### [Inference on GPU](#contents)

Run model inference from mask_rcnn directory:

```bash
bash scripts/run_infer_gpu.sh [CONFIG_PATH] [CHECKPOINT_PATH] [PRED_INPUT] [PRED_OUTPUT]
```

We need 4 parameters for these scripts:

* `CONFIG_FILE`： path to config file.
* `CHECKPOINT_PATH`: path to saved checkpoint.
* `PRED_INPUT`: path to input folder or image.
* `PRED_OUTPUT`: path to output JSON file.

### [Inference result](#contents)

Predictions will be saved in JSON file. File content is list of predictions
for each image. It's supported predictions for folder of images (png, jpeg
file in folder root) and single image.

Typical outputs of such script for single image:

```log
{
 "/data/coco-2017/validation/data/000000000785.jpg": {
  "height": 425,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 279.3103942871094,
     "y_min": 39.71790313720703,
     "width": 214.66873168945312,
     "height": 352.16082763671875
    },
    "class": {
     "label": 0,
     "category_id": "unknown",
     "name": "person"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "QWe32V=3L3N3L3N2M3N3L3N100O100O10O1000O0100O10O010000O1000O1000000000O0100O100O100O0O2O001O000001O001O010O0010O1O001N2N2M3N1O2O1N2M3M2M4E:J7L3N2O2L2L5]Oc0G9O1`ImLj2T3TMQMi2P3VMSMg2n2WMUMg2l2WMXMf2i2WM]Me2d2YMaMSOTOU1]3FdMQOQOW1\\3FgMPOoNX1[3GhMoNnNY1[3GiMnNmN[1[3EjMnNlN]1[3BmMnNiN_1]3_OnMnNfNd1_3XOQNoNbNi1^3VOSNmNaNm1^3ROUNmN^NR2^3oNVNlN^NU2]3mNWNlN]NX2]3iNXNlN\\N]2]3cN[NlNZNb2^3ZNeNiNoMn2`3PNMR25gMM[26^MMc25VMOk23nL1S3Q52O2N3M3M3L2O2N3M5K8H3M1N3N1O001O1O1O2\\MXKcLi4Y3cK^L_4]3hK_LY4`3iK^LW4c3iK\\LX4d3iKZLX4f3iKXLW4h3lKULU4k3mKRLS4o3nKoKS4P4oKnKQ4U4mKjKT4X4jKgKV4\\4jKaKW4b4kKYKU4j4PLmJP4U5QLhJP4Y5PLfJQ4Y5PLfJP4[5oKdJS4]5lKcJS4_5lKaJS4`5RLZJn3f5VLVJi3k5ZLRJf3n5ZLRJf3n5YLTJf3m5XLTJi3m5TLUJl3m5PLUJP4l5mKUJT4l5hKVJX4l5eKUJ[4l5cKTJ^4o5\\KTJd4n5UKVJl4S73M2O1N1O1O1O2N1O3M2O0O2eKiJ`0X5WORKg0o4PO]Kl0d4eMbJ]O\\1i2S4bMZMY2h2`MaM\\2d2[MdMa2c2UMbMi2a6N1O1O1O1O1O2N1O2N1O2N1O1O1O1O2N1N3N2N3M3M3M3M1O2N1N3N3M3L4M2N2M3K8EVel1"
    },
    "score": 0.9995935559272766,
    "iou_score": 0.8712058067321777
   },
   ...
   {
    "bbox": {
     "x_min": 234.1322479248047,
     "y_min": 364.7861022949219,
     "width": 349.34197998046875,
     "height": 33.81195068359375
    },
    "class": {
     "label": 31,
     "category_id": "unknown",
     "name": "snowboard"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "jlQ35T=1N101N10000000000000000000000001O00000000000000000O1000000000O10000WCI`<7_CK`<5`CL_<4`CN_<2aCO^<1bCO^<1bCO^<1bC0]<OdC1\\<OdC1\\<OdC2[<NeC2[<NeC2[<NeC2[<NeC2[<NeC3Z<MfC3Z<MgC2Y<NgC2Y<NgC2Y<NgC2Y<NgC2Y<NgC2Y<NhC1X<OhC2W<MkC2U<NkC2U<NkC2U<NkC2U<NkC2U<NlC1T<OlC1T<OlC1T<OlC1T<OlC1T<OlC1T<OlC1T<OlC1T<OkC2U<NkC2U<NkC2U<NkC2U<NkC2T<OlC1T<OlC1T<OkC2U<NkC2U<NkC2U<NkC2U<MlC3T<MlC3T<MmC2S<NmC2S<NmC2S<NmC2S<NmC2S<NmC2S<NmC2S<NmC2S<NmC2S<NmC2S<NmC2S<NmC2R<OnC1R<OnC1R<OnC1R<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<OnC1R<OnC1R<OnC1R<OnC1R<OnC1R<OnC1R<OnC1R<OnC1R<OnC1R<OnC1R<OnC1S<NmC2S<NmC2S<NmC2S<NmC2S<NmC2S<NmC2S<NmC1T<OlC1T<OlC1T<OlC1S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<0mC0S<OnC1Q<0oC0Q<0oC0Q<1nCOR<1nCOR<1nCOR<1nCOR<1nCOR<1nCOR<1nCOQ<2oCNQ<2oCNQ<2oCNQ<2oCMR<3nCMR<4mCLS<4mCLS<5kCLU<4kCLT<5lCKT<6jCKV<5jCKV<8gCHY<8dCA17[<8dCA17[<8dCB06\\<8dCB06\\<6iCJW<4kCLU<2mCNS<2mCNS<1nCOS<OoC0Q<0oC0Q<OPD1P<OPD1P<NQD2P<MPD3P<LQD4o;KRD5a<00O10001O00000000000O100000000000000000000000000000000000000000O100000000000000000O100000000000000000000000O10000000000000000000000000000000000O100000000000000000000O100000000000000000000000000000O100000000000000000000000000000000000000000000000000000000000000000O10000000000000000000000000000O1001N101O0OYfg0"
    },
    "score": 0.06887401640415192,
    "iou_score": 0.027314260601997375
   },
   {
    "bbox": {
     "x_min": 353.4029541015625,
     "y_min": 357.71539306640625,
     "width": 262.7508544921875,
     "height": 34.744873046875
    },
    "class": {
     "label": 30,
     "category_id": "unknown",
     "name": "skis"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "hSd44T=2N10000000000000O10000000000O1000000000O1000000000000O1000000000000000O100000000O1000000O10000O1000000000000O10O100000000000000001O0000T_<Om`C000000000000000000000000000000000000000000000000000000000O100000000000000000000000000O1000000000000000000000000O10000000000000000000000000000000001O000000000000000000000000000000000000000000000O1000000000000000000000000000O1000000000000000000000000000O10O10000000000000000000000O100000000000000000000O100000000OlW;"
    },
    "score": 0.06499292701482773,
    "iou_score": 0.005751008167862892
   },
   {
    "bbox": {
     "x_min": 477.01318359375,
     "y_min": 377.57855224609375,
     "width": 138.9749755859375,
     "height": 15.829010009765625
    },
    "class": {
     "label": 30,
     "category_id": "unknown",
     "name": "skis"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "lUW62W=000000000000000000000000000000000000000000000000000000000IOVC1j<OVC1j<OUC2Q=000000000000000001O00000000000000000001O00000000000000000000001O0000000O1000000000000000O1000000000O1000000000000000000000000000O10000000000000O10000000O10000000000000000000000O100000O1000000000000000O101NPP:"
    },
    "score": 0.06146843731403351,
    "iou_score": 0.020840032026171684
   }
  ]
 }
}
```

Typical outputs for folder with images:

```log
{
 "/data/coco-2017/validation/data/000000194832.jpg": {
  "height": 425,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 282.8522033691406,
     "y_min": 92.26748657226562,
     "width": 58.37261962890625,
     "height": 38.16679382324219
    },
    "class": {
     "label": 62,
     "category_id": "unknown",
     "name": "tv"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "aae3R1V<2O00000000000000000000000000000001O00000000000O10000000000000010O000000000000000001O00000000000000000000O100O2EiC[OO3[<3hPl3"
    },
    "score": 0.9735541343688965,
    "iou_score": 0.8671513199806213
   },
   {
    "bbox": {
     "x_min": 414.29974365234375,
     "y_min": 210.41567993164062,
     "width": 225.70025634765625,
     "height": 214.58432006835938
    },
    "class": {
     "label": 56,
     "category_id": "unknown",
     "name": "chair"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "[ma57P=5K5L6J4M3L4M1N3N3L4M2M4L3N2N2N3M2N3M3N1N1O1O1N2O1N3N2M3M2O2N2O010N2O1N2N1O1O1O1O2N1O2N1N3N1001O010O10O0O2O0O002N1O1O2N1O1O1O1O1O001O1O1O001O001O00001O1O00001N10O100000O10000000001O000O10000O100O100O100O100O2O0O100O100O1O1O1O1O1O1O1N2O1O1N2N2N2N2O1N2O1O1N2O1M3N2O1O100O1O1O1O1O1M3O1N2O1O1N2O1N2N2O1O1O1O100O1O1O1O1O100O100000000O100000000000000000000000000000001O00000000000000000001O000000001O0000001N2O001N2O0O2O1N2L5K:jMjJ"
    },
    "score": 0.8528746366500854,
    "iou_score": 0.7308270335197449
   },
   {
    "bbox": {
     "x_min": 9.537391662597656,
     "y_min": 197.74517822265625,
     "width": 213.72845458984375,
     "height": 224.90188598632812
    },
    "class": {
     "label": 56,
     "category_id": "unknown",
     "name": "chair"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "h[4o5U7;I2N2O0O1000001O000O1000000O100000000O10000000000000000001O0000001O001O00001O001O1O1O2N1O1O2N1O1O1O1O1O1O1O2N1O1O1O2N1O3M1O2N1O1O1O2N2N2N2N2N1O1O1O2N2N2N2N2N2N1O2N1O1O2N1O2N1O1O2N2N2N1O1O00001O000000000000001O0000000000000000O100000000001O00000000000000000O100000O10000000001O000000O100000000000O1O100O1O1O1O1000O010O100N2O1O1O0O2O0N2NgNYLUIe3i6bLTI[3n6hLPIV3S7jLmHT3U7lLlHR3U7nLkHP3W7QMhHn2Y7RMhHj2Z7WMfHg2\\7YMdHe2]7\\McHc2]7_MbH_2^7dMaHZ2_7iM`HV2_7mM`HQ2a7YNVHe1k7^NRHb1o7_NoG_1S8bNlG^1U8cNjG[1X8eNgGZ1[8fNeGW1^8iNaGT1c8mN\\GQ1f8oNYGP1i8QOUGo0l8f11O1N1M4M3M3N2M4K4J7M3L3M4L4M5F<UNoY]5"
    },
    "score": 0.7906664609909058,
    "iou_score": 0.6069739460945129
   },
   ...
   {
    "bbox": {
     "x_min": 366.46875,
     "y_min": 78.4329833984375,
     "width": 273.53125,
     "height": 321.17999267578125
    },
    "class": {
     "label": 7,
     "category_id": "unknown",
     "name": "truck"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "bbh4[1f9WO_G`1Q8QO^G]1U8R2L4N001O001O00001O00001O0000000000001O000000000000000O0100000000O1O1O1O00100O100O1000O10001O0O10000O101N100O1O101PHSKa7n4ZHXKe7h4XH\\Kf7e4YH]Kf7T5N2O2M2O1N3N1N101N2O0O2O0O2N2N101N2_NlIgKX6X4mI^KZ6`4lITK\\6j4mIhJ[6T5o0N2M3N2M3N2M3N2M2N3M3N5K;E3N2N1NWL\\LQNb3e1jLZNU3f1lL[NR3e1nL\\NP3d1RM[Nn2e1RM\\Nm2d1TM[Nl2e1TM[Nl2e1UM[Nj2e1VM[Nj2e1WM[Nh2d1ZM[Nf2c1]M\\Nc2c1_M\\N`2d1bM[N^2d1dM[N\\2e1dM[N\\2d1eM]NZ2b1hM]NX2T1hLdKP1X3X2S1lLaKm0\\3W2S1lLaKm0\\3W2T1jLaKo0[3W2T1hLcKQ1Y3W2b1iM^NW2b1iM^NW2b1iM^NW2b1iM^NW2b1iM^NW2b1iM^NW2c1hM]NX2e1fM[NZ2h1cMXN]2j1aMVN_2k1`MUN`2m1]MTNc2m1\\MSNd2n1[MRNe2o1ZMQNf2P2YMPNg2Q2WMPNi2P2VMQNj2o1UMRNk2o1SMRNm2n1SMRNm2n1RMSNn2m1QMTNo2l1QMTNo2l1QMTNo2l1PMUNP3k1PMUNP3k1PMUNP3k1PMUNP3k1PMUNP3l1oLTNQ3l1nLUNR3k1nLUNR3k1nLUNR3k1nLUNR3k1mLVNS3k1lLUNT3k1kLVNU3j1kLVNU3j1jLWNV3j1iLVNW3j1iLVNW3j1hLWNX3i1hLWNX3m500000000000000000`KiLnNW3P1mLnNS3>hLiK5i3S3d50000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001N100000000O101O0O2N2N:aNm6_IWH"
    },
    "score": 0.0541781485080719,
    "iou_score": 0.032013799995183945
   },
   {
    "bbox": {
     "x_min": 129.17379760742188,
     "y_min": 183.64930725097656,
     "width": 160.9864501953125,
     "height": 82.79652404785156
    },
    "class": {
     "label": 28,
     "category_id": "unknown",
     "name": "suitcase"
    },
    "mask": {
     "size": [
      425,
      640
     ],
     "counts": "lTf1>c<<K3N0O2O0O101O00O0100000000000000O2OO10000O10O1000000000000O01000000000000000O100000001O00000000000000000001O0O100000000000000000000000000000000O100000000O1N2HiC]OX<b0800000000000000000bC]ONOS<e0nC]OMOV<c0lCCT<=lCCT<=lCCT<=kCDU<=jCCV<h000000001O00001O000000001O0000001O001O1O010O1O1O1O1O003M3M7I4M1N2N1O1O101N100001N101O004K2N:F4M1N1O2N2N4L3M4L2M4M3L5K5L2KSka4"
    },
    "score": 0.05066123232245445,
    "iou_score": 0.028819410130381584
   }
  ]
 },
 "/data/coco-2017/validation/data/000000104572.jpg": {
  "height": 419,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 88.95381927490234,
     "y_min": 291.34234619140625,
     "width": 143.5157470703125,
     "height": 77.3465576171875
    },
    "class": {
     "label": 71,
     "category_id": "unknown",
     "name": "sink"
    },
    "mask": {
     "size": [
      419,
      640
     ],
     "counts": "bTU1i0Y<2N3L3N2O1N100O2O0O101N100O10001N100O10000O2O000O10001N100O100O10001O0O100000001O00000O101O000000000O2O000O1O1O2N100O1O101N100O100O10001O00000O100000000000000001O00000O1000000O2O0O100O100O2O0O1O100O1O1O1O100O1O2O0O1O1O101N1O100O100O100O2O0O100O100O100O1O100O2O0O1O2O0O100O2N1O2N2N9DcdV5"
    },
    "score": 0.9895582795143127,
    "iou_score": 0.9149816632270813
   },
   {
    "bbox": {
     "x_min": 0.0,
     "y_min": 365.7265625,
     "width": 83.60945892333984,
     "height": 52.9224853515625
    },
    "class": {
     "label": 71,
     "category_id": "unknown",
     "name": "sink"
    },
    "mask": {
     "size": [
      419,
      640
     ],
     "counts": "b;h06IZ;S1O00000000000000000000000000000000000001O0000001O000000001O00000000001O000000001O000000001O00001O000000001O000000001O000000001O0000001O000000001O00001O001O001O5K9Gl`S7"
    },
    "score": 0.9093805551528931,
    "iou_score": 0.8558843731880188
   },
   ...
   {
    "bbox": {
     "x_min": 384.8828125,
     "y_min": 200.68560791015625,
     "width": 40.55206298828125,
     "height": 10.92425537109375
    },
    "class": {
     "label": 71,
     "category_id": "unknown",
     "name": "sink"
    },
    "mask": {
     "size": [
      419,
      640
     ],
     "counts": "agm43P=000O100O1O1O100O1000001OO100M300O10000000000O1000000000010O01O0000001O00N2OTCOh<1Pig2"
    },
    "score": 0.05123830586671829,
    "iou_score": 0.017626700922846794
   }
  ]
 },
 ...
 "/data/coco-2017/validation/data/000000533855.jpg": {
  "height": 428,
  "width": 640,
  "predictions": [
   {
    "bbox": {
     "x_min": 371.3515625,
     "y_min": 111.49694061279297,
     "width": 268.080322265625,
     "height": 243.64804077148438
    },
    "class": {
     "label": 54,
     "category_id": "unknown",
     "name": "donut"
    },
    "mask": {
     "size": [
      428,
      640
     ],
     "counts": "lXk48_<d1nN7I6L3M2M4M2M4M3L3M4L3N2N2N2N2N2M4L4M3L3N3M2O1N2N2N3M2N4L4L3M3N1N2N2N3M2N3M2O2M101N2O0O2O0O2N1O2N1O2O1N2O0O2O1N101N101N1O2O0O2N2O0O2O1O0O2O001O0O101O000O101O000O100O2N100O1O2O000O2O001O00001O0O101O000000001O000O10000000000O2O00000O10000O101O000O1000000000001N1000000000001O000000000001O0000000000001O0000001O00000O101O000000001O0000000O10001O000000000O1000001O000000000O101O001O0O2O001O1N101O0O101N1O100O2O0O101N101O1N3N1N2O1N101N1O2N2O1N2N2O1N2N101N1O2O0N3O1N2N2N2O1N101N2N2O0O2M3N2O1N2N2N2O1N2M4M2M4M2N3M2N2N3M4K4L5K4M3L4M3L4L6J8DU7"
    },
    "score": 0.998783528804779,
    "iou_score": 0.9730523228645325
   },
   ...
   {
    "bbox": {
     "x_min": 233.8058319091797,
     "y_min": 3.3231470584869385,
     "width": 398.69427490234375,
     "height": 174.04608154296875
    },
    "class": {
     "label": 44,
     "category_id": "unknown",
     "name": "spoon"
    },
    "mask": {
     "size": [
      428,
      640
     ],
     "counts": "]cX3n0\\<;F3L3N3L3M3N2N1O2O1N2N101N1O2O0O2O0O1000001O0O10000000000O10000000000O1000O10O10O10O1000O10O10O100O10000O10O010000O100O1000O01O100O1O1O00100O1O10O0100O10O10O1000O01000O0100O10O0100O100O2O0O1O100O1O2N1O100O10eN^NlFa1U9^NlFb1^:100O1O010O100O1N1M4N2N2GRNREo1n:8N2O1O1O001O100O010O10O10O010O1O100O1O10O01O100O100O1000O10O100000O10O100000000O10000000O10O10000000000O10O1000O10000O1000O01000O0100000O0100000O0100O10O10O10O1000O10O100000O0100000O1000O0100O100O1000O01000000O10O10O1000000O01000O100O1000000O1000000O100000O1000O10O10000O1000O10O10000000O10O100000O10O10000000O10O1000000O01000000000O10O100000O1000O0100000O010000O0100000O10O0100O100O100O100O100O010O10000O10000O1000000O1000000O10000O100O10000O10000O1000000O10000O10000O10000O10000O100000000O1000000000000O1000001O0O1O2O2K:FdZ3"
    },
    "score": 0.06164628639817238,
    "iou_score": 0.050566934049129486
   }
  ]
 }
}
```

## [Model Description](#contents)

### [Performance](#contents)

| Parameters          | GPU                                                  |
|---------------------|------------------------------------------------------|
| Model Version       | Mask Scoring RCNN ResNet50                           |
| Resource            | NVIDIA GeForce RTX 3090 (x4)                         |
| Uploaded Date       | 31/10/2023 (day/month/year)                          |
| MindSpore Version   | 2.1.0                                                |
| Dataset             | COCO2017                                             |
| Pretrained          | noised checkpoint (bbox_mAP=36.9%, segm_mAP=34.8%)   |
| Training Parameters | epoch = 5, batch_size = 3 (per device)               |
| Optimizer           | SGD (momentum)                                       |
| Loss Function       | Sigmoid Cross Entropy, SoftMax Cross Entropy, L1Loss |
| Speed               | 4pcs: 3269 ms/step                                   |
| Total time          | 4pcs: 44h 45m 15s                                    |
| outputs             | mAP(bbox), mAP(segm)                                 |
| mAP(bbox)           | 37.4                                                 |
| mAP(segm)           | 35.5                                                 |
| Model for inference | 231.9M(.ckpt file)                                   |
| configuration       | ms_rcnn_r50_caffe_fpn_1x_coco_experiment.yaml        |
| Scripts             |                                                      |

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also set
random seed in train.py.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
