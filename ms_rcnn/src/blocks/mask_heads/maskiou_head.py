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
#
# This file or its part has been derived from the following repository
# and modified: https://github.com/open-mmlab/mmdetection/tree/v2.28.2
# ============================================================================
"""Mask IOU Head for Mask Scoring R-CNN"""
import numpy as np
import mindspore as ms
from mindspore import nn, ops

from ..utils import pair
from .. import Config

class MaskIoUHead(nn.Cell):
    """Mask IoU Head.

    This head predicts the IoU of predicted masks and corresponding gt masks.
    """

    def __init__(
            self,
            img_height: int,
            img_width: int,
            train_batch_size: int,
            test_batch_size: int,
            loss_iou,
            loss_iou_weight=0.5,
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=80,
            mask_divider=2,
            train_cfg: Config = Config(
                dict(
                    rpn_proposal=dict(max_per_img=1000),
                    rcnn=dict(mask_thr_binary=0.5)
                )
            ),
            test_cfg: Config = Config(
                dict(
                    rcnn=dict(
                        score_thr=0.05,
                        iou_threshold=0.5,
                        max_per_img=100,
                    )
                )
            )
        ):
        super(MaskIoUHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.max_per_img = test_cfg.rcnn.max_per_img
        self.train_pos_num = int(
            train_cfg.rcnn.sampler.num * train_cfg.rcnn.sampler.pos_fraction
        )
        self.train_mask_thr_binary = train_cfg.rcnn.mask_thr_binary
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.img_width = img_width
        self.img_height = img_height

        convs = []
        for i in range(num_convs):
            if i == 0:
                # concatenation of mask feature and mask prediction
                in_channels = self.in_channels + 1
            else:
                in_channels = self.conv_out_channels
            stride = 2 if i == num_convs - 1 else 1
            convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.conv_out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    pad_mode='pad',
                    has_bias=True
                )
            )

        self.convs = nn.CellList(convs)
        self.roi_feat_size = pair(roi_feat_size)
        self.mask_size = pair(roi_feat_size * 2)

        pooled_area = (
            (self.roi_feat_size[0] // 2) * (self.roi_feat_size[1] // 2)
        )
        fcs = []
        for i in range(num_fcs):
            in_channels = (
                self.conv_out_channels *
                pooled_area if i == 0 else self.fc_out_channels)
            fcs.append(nn.Dense(in_channels, self.fc_out_channels))
        self.fcs = nn.CellList(fcs)

        self.fc_mask_iou = nn.Dense(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

        self.loss_iou_weight = loss_iou_weight
        self.loss_iou = loss_iou

        self.bboxes_range = ms.Tensor(
            np.arange(self.max_per_img * test_batch_size).reshape(-1, 1),
            ms.int32
        )
        self.bboxes_range_train = ms.Tensor(
            np.arange(self.train_pos_num * train_batch_size).reshape(-1, 1),
            ms.int32
        )

        self.y_img, self.x_img = ops.meshgrid(
            ms.Tensor(np.arange(img_height // mask_divider)),
            ms.Tensor(np.arange(img_width // mask_divider)),
            indexing='ij'
        )
        self.mask_divider = mask_divider
        self.map = ops.Map()
        self.eps = ms.Tensor(1e-7, ms.float32)


    def construct(self, mask_feat, mask_pred):
        mask_pred = ops.reshape(
            mask_pred, (-1, 1, self.mask_size[0], self.mask_size[1])
        )
        mask_pred_pooled = self.max_pool(mask_pred)

        x = ops.concat([mask_feat, mask_pred_pooled], 1)

        for conv in self.convs:
            x = self.relu(conv(x))
        x = ops.flatten(x)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.fc_mask_iou(x)
        return mask_iou

    def loss(self, mask_iou_pred, mask_iou_targets, mask_weights):
        mask_weights = ops.cast(mask_weights, ms.float32)
        loss_mask_iou = self.loss_iou(
            mask_iou_pred, mask_iou_targets
        )
        weights_sum = ops.reduce_sum(mask_weights)
        loss_mask_iou = loss_mask_iou * mask_weights
        loss_mask_iou = (
            ops.reduce_sum(loss_mask_iou) / (weights_sum + self.eps)
        )
        return loss_mask_iou * self.loss_iou_weight

    def get_targets(
            self, pos_bboxes_list, pos_ids_list, gt_masks_list, mask_pred,
            mask_targets
    ):
        """Compute target of mask IoU.

        Mask IoU target is the IoU of the predicted mask (inside a bbox) and
        the gt mask of corresponding gt mask (the whole instance).
        The intersection area is computed inside the bbox, and the gt mask area
        is computed with two steps, firstly we compute the gt area inside the
        bbox, then divide it by the area ratio of gt area inside the bbox and
        the gt area of the whole instance.

        Args:
            pos_bboxes_list (List[Tensor]): List of pos_bboxes per each image.
            pos_ids_list (List[Tensor]): List of sampled pos ids for each image.
            gt_masks_list (List[Tensor]): List of GT segmentation masks for each image.
            mask_pred (Tensor): Predicted masks of each positive proposal,
                shape (num_pos, h, w).
            mask_targets (Tensor): Gt mask of each positive proposal,
                binary map of the shape (num_pos, h, w).

        Returns:
            Tensor: mask iou target (length == num positive).
        """
        # compute the area ratio of gt areas inside the proposals and
        # the whole instance
        area_ratios_list = []
        for i in range(self.train_batch_size):
            pos_proposals = pos_bboxes_list[i]
            pos_ids = pos_ids_list[i]
            gt_masks = gt_masks_list[i]
            area_ratios = self.map(
                self._get_area_ratio, [x for x in pos_proposals],
                [gt_masks[j] for j in pos_ids]
            )
            area_ratios_list.append(ops.stack(area_ratios))
        area_ratios = ops.concat(area_ratios_list)
        assert mask_targets.shape[0] == area_ratios.shape[0]

        mask_pred = ops.cast(mask_pred > self.train_mask_thr_binary, ms.int32)
        mask_pred_areas = ops.reduce_sum(mask_pred, (-1, -2))

        # mask_pred and mask_targets are binary maps
        overlap_areas = ops.reduce_sum(mask_pred * mask_targets, (-1, -2))

        # compute the mask area of the whole instance
        gt_full_areas = (
            ops.reduce_sum(mask_targets, (-1, -2)) / (area_ratios + self.eps)
        )
        mask_iou_targets = (
            overlap_areas /
            (mask_pred_areas + gt_full_areas - overlap_areas + self.eps)
        )
        return mask_iou_targets

    def _get_area_ratio(self, pos_proposal, gt_mask):
        """Compute area ratio of the gt mask inside the proposal and the gt
        mask of the corresponding instance."""
        pos_proposal = pos_proposal / self.mask_divider
        left_mask = self.x_img > pos_proposal[0]
        right_mask = self.x_img < pos_proposal[2]
        up_mask = self.y_img > pos_proposal[1]
        down_mask = self.y_img < pos_proposal[3]
        proposal_mask = ops.logical_and(
            ops.logical_and(ops.logical_and(left_mask, right_mask), up_mask),
            down_mask
        )
        proposal_mask = ops.cast(proposal_mask, ms.float32)
        proposal_area = ops.reduce_sum(proposal_mask * gt_mask)
        gt_area = ops.reduce_sum(gt_mask)
        area_ratio = proposal_area / (gt_area + self.eps)
        return ops.cast(area_ratio, ms.float32)

    def get_mask_scores(self, mask_iou_logits, det_bboxes, det_labels):
        """Get the mask scores.

        mask_score = bbox_score * mask_iou
        """
        det_labels = ops.reshape(det_labels, (-1, 1))
        if self.training:
            indices = ops.concat((self.bboxes_range_train, det_labels), axis=1)
        else:
            indices = ops.concat((self.bboxes_range, det_labels), axis=1)
        mask_iou_pred = ops.gather_nd(mask_iou_logits, indices)
        if not self.training:
            det_bboxes = ops.reshape(det_bboxes, (-1, 5))
            mask_iou_pred = ops.reshape(mask_iou_pred * det_bboxes[::, -1],
                                        (self.test_batch_size, -1))
        return mask_iou_pred
