# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models import DETECTORS

from mmdet3d.models.detectors import BEVDet4D
from ..builder import build_head, build_neck
from ...core import bbox3d2result


@DETECTORS.register_module()
class BEVNeXt(BEVDet4D):

    def __init__(self, transformer_neck, proposal_head, **kwargs):
        super(BEVNeXt, self).__init__(**kwargs)
        train_cfg = kwargs['train_cfg']
        test_cfg = kwargs['test_cfg']
        if train_cfg is not None:
            proposal_head['train_cfg'] = train_cfg['pts']
        if test_cfg is not None:
            proposal_head['test_cfg'] = test_cfg['pts']
        self.transformer_neck = build_neck(transformer_neck)
        self.proposal_head = build_head(proposal_head)
        self.depth_emb = nn.Sequential(*[
            nn.Conv2d(self.img_view_transformer.D, self.img_view_transformer.D, 1),
            nn.BatchNorm2d(self.img_view_transformer.D),
            nn.ReLU(),
            nn.Conv2d(self.img_view_transformer.D, self.img_view_transformer.out_channels, 1),
        ])
        assert not self.align_after_view_transfromation

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        res = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(res)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return [x, res]

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        assert not sequential
        assert not pred_prev
        imgs, rots, trans, intrins, post_rots, post_trans, bda = \
            self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        context_list = []
        others_list = []
        key_frame = True  # back propagation for key frame only
        for img, rot, tran, intrin, post_rot, post_tran in zip(
                imgs, rots, trans, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    rot, tran = rots[0], trans[0]
                prev_info = others_list[-1] if len(others_list) > 0 else None
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran,
                    bda) if self.img_view_transformer.use_mlp_input else None
                inputs_curr = (img, rot, tran, intrin, post_rot,
                               post_tran, bda, mlp_input, prev_info, kwargs['canvas'])
                if key_frame:
                    bev_feat, depth, others, context_feat = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth, others, context_feat = self.prepare_bev_feat(*inputs_curr)
                others_list.append(others)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
                context_feat = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            context_list.append(context_feat)
            key_frame = False

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        heatmap = self.proposal_head([x])
        feats_2d = context_list[0]
        # detach depth embedding in perspective refinement
        depth_detached = depth_list[0].detach()
        feats_2d = feats_2d + self.depth_emb(depth_detached)
        cam_params = (rots[0], trans[0], intrins[0], post_rots[0], post_trans[0], bda)
        B, _, H_BEV, W_BEV = bev_feat.shape
        BN, C, H, W = feats_2d.shape
        feats_list = [feats_2d.view(B, 6, C, H, W)]
        # [B L C]
        refined_feats = self.transformer_neck(feats_list,
                                              lss_feats=x,
                                              cam_params=cam_params)
        refined_feats = refined_feats.permute(0, 2, 1).view(B, -1, H_BEV, W_BEV)
        return [refined_feats], depth_list[0], others_list, [heatmap]

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input, prev_info, canvas):
        x = self.image_encoder(img)
        bev_feat, depth, others, tran_feat = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input, prev_info, canvas])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, others, tran_feat

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth, others_list, context = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth, others_list, context)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth, others_list, proposal_heatmap = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        losses = dict()
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth,
                                                              depth,
                                                              kwargs['canvas'])
        losses.update(loss_depth)
        losses_reg = self.forward_pts_train((img_feats, proposal_heatmap),
                                            gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_reg)
        proposal_losses = self.proposal_head.loss(*[gt_bboxes_3d, gt_labels_3d, proposal_heatmap[0]])
        losses.update(proposal_losses)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, pts_feats, depth, others_list, proposal_heatmap = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts((img_feats, proposal_heatmap), img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
