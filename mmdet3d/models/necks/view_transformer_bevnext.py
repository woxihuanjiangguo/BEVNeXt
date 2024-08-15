import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast

from .depthnet import DepthNet
from .view_transformer import LSSViewTransformerBEVDepth
from ..builder import NECKS
from ...ops.bev_pool_v2.bev_pool import bev_pool_v2


@torch.no_grad()
def patch_color(image_tensor, patch_size=16):
    B, N, H, W, C = image_tensor.shape
    image_tensor = image_tensor.long()
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    patches = image_tensor.view(B, N, num_patches_h, patch_size, num_patches_w, patch_size, C)
    patches = patches.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    patches = patches.view(B * N, num_patches_h, num_patches_w, -1, C)
    patches = patches.float().mean(3)
    patches = patches.permute(0, 3, 1, 2).contiguous()
    return patches


def color_diff(color, zeta, unfold):
    B, C, H, W = color.shape
    unfolded_hists = unfold(color).reshape(B, C, 3 ** 2, H, W)
    return torch.exp(
        -(((unfolded_hists - color.reshape(B, C, 1, H, W)) ** 2) / (2 * zeta ** 2)).sum(1))


class MeanField(nn.Module):

    def __init__(self,
                 crf_num_iter,
                 crf_zeta=0.001,
                 grad=True,
                 tolerance=0.0
                 ):
        super(MeanField, self).__init__()
        self.tolerance = tolerance
        self.kernel_size = 3
        assert self.kernel_size % 2 == 1
        self.zeta = crf_zeta
        self.num_iter = crf_num_iter
        self.ws = nn.ParameterList(
            [
                nn.Parameter(0.015 * torch.ones(1), requires_grad=True) if grad else
                nn.Parameter(0.01 * torch.ones(1), requires_grad=False)
            ],
        )
        self.unfold = torch.nn.Unfold(self.kernel_size, stride=1, padding=self.kernel_size // 2)

    def forward(self, color, feats, logits):
        kernels = [
            self.gaussian_kernel(color, self.zeta),
        ]
        q = logits.softmax(1)
        for it in range(self.num_iter):
            q = self.single_forward(logits, kernels, q)
        return q

    def miu(self, D, device):
        rows = torch.arange(D, device=device).reshape(-1, 1)
        cols = torch.arange(D, device=device).reshape(1, -1)
        matrix = (torch.abs(rows - cols) - self.tolerance).clamp(0.0)
        return matrix.float()

    def gaussian_kernel(self, feats, zeta):
        B, C, H, W = feats.shape
        unfolded_feats = self.unfold(feats).reshape(B, C, self.kernel_size ** 2, H * W)
        kernel = torch.exp(
            -(((unfolded_feats - feats.reshape(B, C, 1, H * W)) ** 2) / (2 * zeta ** 2)).sum(1))
        return kernel

    def single_forward(self, logits, kernels, q):
        B, D, H, W = logits.shape
        kernel_size = self.kernel_size
        # unfold_x [B, D, kernel_size**2, H * W]
        # kernel   [B,    kennel_size**2, H * W]
        unfold_q = self.unfold(q).reshape(B, D, kernel_size ** 2, H * W)
        # msg passing
        # q_tilde [B, D, H * W]
        q_tilde = 0.0
        for i, kernel in enumerate(kernels):
            q_tilde += (unfold_q * kernel[:, None]).sum(2) * self.ws[i]
        # miu [B, D, D]
        miu = self.miu(D, logits.device)[None].repeat(B, 1, 1)
        # q_hat [B, D, H * W]
        q_hat = torch.bmm(miu, q_tilde)

        result = (logits.reshape(B, D, -1) - q_hat).softmax(1)
        return result.reshape(B, D, H, W)


@NECKS.register_module()
class LSSViewTransformerBEVNeXt(LSSViewTransformerBEVDepth):

    def __init__(self,
                 crf_config=None,
                 loss_depth_weight=3.0,
                 patch_size=8,
                 depthnet_cfg=dict(),
                 **kwargs):
        super(LSSViewTransformerBEVNeXt, self).__init__(loss_depth_weight, depthnet_cfg, **kwargs)
        self.depth_net = DepthNet(
            **depthnet_cfg,
            context_channels=self.out_channels,
            depth_channels=self.D,
        )

        self.depth_cfg = self.grid_config['depth']
        self.use_mlp_input = True
        self.unfold = torch.nn.Unfold(3, stride=1, padding=1)
        self.crf = MeanField(**crf_config)
        self.patch_size = patch_size
        self.mid_channels = depthnet_cfg['mid_channels']

    def forward(self, input):
        (x_res, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input, prev_info, canvas) = input[:10]
        x, res_feat = x_res
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        logits, tran_feat = self.depth_net(x, mlp_input=mlp_input)

        if isinstance(canvas, list):
            color = patch_color(canvas[0], patch_size=self.patch_size)
        else:
            color = patch_color(canvas, patch_size=self.patch_size)

        depth_prob = self.crf(color, None, logits)

        bev, _ = self.view_transform([x_res[0], rots, trans, intrins, post_rots, post_trans, bda, ],
                                     depth_prob,
                                     tran_feat)
        others = {
        }
        return (bev, depth_prob,
                others,
                tran_feat)

    @force_fp32()
    def get_depth_loss(self, ori_labels, depth_pred, canvas, eps=1e-7):
        loss_dict = dict()
        depth_labels, D = self.get_downsampled_gt_depth(ori_labels, self.downsample,
                                                        depth_config=self.depth_cfg)
        depth_labels = F.one_hot(
            depth_labels.long(), num_classes=D + 1)[..., 1:].float()

        depth_labels_flattened = depth_labels.view(-1, D)
        fg_mask = torch.max(depth_labels_flattened, dim=1).values > 0.0

        BN, D, H, W = depth_pred.shape
        depth_pred = depth_pred.permute(0, 2, 3, 1).contiguous().view(-1, D)
        depth_labels_flattened = depth_labels_flattened[fg_mask]
        depth_pred = depth_pred[fg_mask]

        with autocast(enabled=False):
            # depth loss
            depth_loss = F.binary_cross_entropy(
                depth_pred,
                depth_labels_flattened,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())

        loss_dict['loss_depth'] = depth_loss * self.loss_depth_weight

        return loss_dict

    def get_downsampled_gt_depth(self, gt_depths, downsample, depth_config):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)

        gt_depths = (gt_depths - (depth_config[0] - depth_config[2])) / depth_config[2]
        D = math.ceil((depth_config[1] - depth_config[0]) / depth_config[2])
        gt_depths = torch.where((gt_depths < D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        return gt_depths, D

    def get_other_loss(self, others):
        return dict()

    def view_transform_core(self, input, depth, tran_feat):
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_lidar_coor(*input[1:7])
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat, depth
