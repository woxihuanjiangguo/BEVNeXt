import torch.nn as nn
import torch.utils.checkpoint as ckpt
from mmcv.cnn import build_conv_layer
from mmseg.models.backbones.resnet import BasicBlock

from .view_transformer import Mlp
from .view_transformer import SELayer, ASPP


class DepthNet(nn.Module):

    def __init__(self,
                 depth_in_channels,
                 mid_channels,
                 context_in_channels,
                 context_channels,
                 depth_channels,
                 aspp_mid_c=None,
                 use_dcn=False,
                 with_cp=True
                 ):
        super(DepthNet, self).__init__()
        self.with_cp = with_cp
        if aspp_mid_c is None:
            aspp_mid_c = mid_channels
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                depth_in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, aspp_mid_c),
            nn.Conv2d(mid_channels, depth_channels, 1, 1)
        ]
        self.depth_conv = nn.Sequential(*depth_conv_list)
        # macro design in AeDet for training stability
        if use_dcn:
            self.context_conv = nn.Sequential(
                build_conv_layer(cfg=dict(
                    type='DCN',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    groups=4,
                    im2col_step=128
                )),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels,
                          context_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0)
            )
        else:
            self.context_conv = nn.Conv2d(mid_channels, context_channels, 1, 1)

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        if self.with_cp:
            for conv in self.depth_conv:
                depth = ckpt.checkpoint(conv, depth)
            depth_result = depth
        else:
            depth_result = self.depth_conv(depth)
        return depth_result, self.context_conv(context)
