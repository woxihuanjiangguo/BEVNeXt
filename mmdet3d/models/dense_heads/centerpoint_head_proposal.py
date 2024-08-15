import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.core import multi_apply, reduce_mean
from torch import nn

from mmdet3d.core import (draw_heatmap_gaussian, gaussian_radius)
from mmdet3d.models import builder
from mmdet3d.models.utils import clip_sigmoid
from ..builder import HEADS, build_loss


@HEADS.register_module()
class CenterHeadProposal(BaseModule):
    """CenterHead for CenterPoint.

    Args:
        in_channels (list[int] | int, optional): Channels of the input
            feature map. Default: [128].
        tasks (list[dict], optional): Task information including class number
            and class names. Default: None.
        train_cfg (dict, optional): Train-time configs. Default: None.
        test_cfg (dict, optional): Test-time configs. Default: None.
        bbox_coder (dict, optional): Bbox coder configs. Default: None.
        common_heads (dict, optional): Conv information for common heads.
            Default: dict().
        loss_cls (dict, optional): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict, optional): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict, optional): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int, optional): Output channels for share_conv
            layer. Default: 64.
        num_heatmap_convs (int, optional): Number of conv layers for heatmap
            conv layer. Default: 2.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=256,
                 max_num=500,
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 init_cfg=None):
        # assert init_cfg is None, 'To prevent abnormal initialization ' \
        #     'behavior, init_cfg is not allowed to be set'
        super(CenterHeadProposal, self).__init__(init_cfg=init_cfg)
        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.max_num = max_num

        self.loss_cls = build_loss(loss_cls)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = dict(heatmap=(num_cls, num_heatmap_convs))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(builder.build_head(separate_head))

    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_single, feats)

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor, optional): Mask of the feature map with the
                shape of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which
                        boxes are valid.
        """
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                                     x - pc_range[0]
                             ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                                     y - pc_range[1]
                             ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    box_dim = box_dim.log()
                    vx, vy = task_boxes[idx][k][7:]
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0)
                    ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, _, _, _ = self.get_targets(
            gt_bboxes_3d, gt_labels_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(
                reduce_mean(heatmaps[task_id].new_tensor(num_pos)),
                min=1).item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=cls_avg_factor)
            # reconstruct the anno_box from multiple reg heads
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap

        return loss_dict

    def get_heatmap_peaks(self, preds_dicts):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        K = self.max_num
        for task_id, preds_dict in enumerate(preds_dicts):
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()
            batch, cat, height, width = batch_heatmap.size()

            topk_scores, topk_inds = torch.topk(batch_heatmap.view(batch, cat, -1), K)

            topk_inds = topk_inds % (height * width)
            topk_ys = (topk_inds.float() /
                       torch.tensor(width, dtype=torch.float)).int().float()
            topk_xs = (topk_inds % width).int().float()

            topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
            topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
            topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1),
                                          topk_ind).view(batch, K)
            topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1),
                                        topk_ind).view(batch, K)
            topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1),
                                        topk_ind).view(batch, K)
            rets.append({
                'scores': topk_score, 'inds': topk_inds, 'clses': topk_clses, 'ys': topk_ys, 'xs': topk_xs
            })
        return rets
