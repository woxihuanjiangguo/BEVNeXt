import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_positional_encoding
from mmcv.runner import auto_fp16
from mmdet3d.models.model_utils.spatial_cross_attention import MSDeformableAttention

from mmdet3d.models import NECKS


@NECKS.register_module()
class PerspectiveRefinement(nn.Module):
    def __init__(self, decoder, bev_shape, positional_encoding, in_channels,
                 embed_dims):
        super(PerspectiveRefinement, self).__init__()
        self.decoder = build_transformer_layer_sequence(decoder)
        self.bev_h, self.bev_w = bev_shape
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, embed_dims)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.shared_conv = ConvModule(
            in_channels,
            embed_dims,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            bias='auto')

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def forward(
            self,
            mlvl_feats,
            lss_feats,
            # grid_length=[0.512, 0.512],
            cam_params=None,
            gt_bboxes_3d=None,
            pred_img_depth=None,
            prev_bev=None,
            bev_mask=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        dtype = mlvl_feats[0].dtype

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        lss_feats = self.shared_conv(lss_feats)
        bev_queries = bev_queries + lss_feats.flatten(2).permute(2, 0, 1)
        bev_pos = self.positional_encoding(bs, self.bev_h, self.bev_w, bev_queries.device).to(dtype)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        bev_embed = self.decoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            cam_params=cam_params,
            gt_bboxes_3d=gt_bboxes_3d,
            pred_img_depth=pred_img_depth,
            prev_bev=prev_bev,
            bev_mask=bev_mask,
            **kwargs
        )

        return bev_embed
