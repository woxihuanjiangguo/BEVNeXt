import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.utils.checkpoint as cp
from mmcls.models.backbones.resnet import Bottleneck as _Bottleneck
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import ModuleList, Sequential
from ..builder import BACKBONES

class Bottle2neck(_Bottleneck):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=4,
                 base_width=26,
                 base_channels=64,
                 expansion=4,
                 stage_type='normal',
                 **kwargs):
        """Bottle2neck block for Res2Net."""
        super(Bottle2neck, self).__init__(in_channels, out_channels, **kwargs)
        self.expansion = expansion
        assert scales > 1, 'Res2Net degenerates to ResNet when scales = 1.'

        mid_channels = out_channels // self.expansion
        width = int(math.floor(mid_channels * (base_width / base_channels)))

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width * scales, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            width * scales,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        if stage_type == 'stage':
            self.pool = nn.AvgPool2d(
                kernel_size=3, stride=self.conv2_stride, padding=1)

        self.convs = ModuleList()
        self.bns = ModuleList()
        for i in range(scales - 1):
            self.convs.append(
                build_conv_layer(
                    self.conv_cfg,
                    width,
                    width,
                    kernel_size=3,
                    stride=self.conv2_stride,
                    padding=self.dilation,
                    dilation=self.dilation,
                    bias=False))
            self.bns.append(
                build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width * scales,
            self.out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.stage_type = stage_type
        self.scales = scales
        self.width = width
        delattr(self, 'conv2')
        delattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            spx = torch.split(out, self.width, 1)
            sp = self.convs[0](spx[0].contiguous())
            sp = self.relu(self.bns[0](sp))
            out = sp
            for i in range(1, self.scales - 1):
                if self.stage_type == 'stage':
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp.contiguous())
                sp = self.relu(self.bns[i](sp))
                out = torch.cat((out, sp), 1)

            if self.stage_type == 'normal' and self.scales != 1:
                out = torch.cat((out, spx[self.scales - 1]), 1)
            elif self.stage_type == 'stage' and self.scales != 1:
                out = torch.cat((out, self.pool(spx[self.scales - 1])), 1)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class FusionLayer(_Bottleneck):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=4,
                 base_width=26,
                 base_channels=64,
                 expansion=1,
                 group_wise=True,
                 **kwargs):
        super(FusionLayer, self).__init__(in_channels, out_channels, **kwargs)
        self.expansion = expansion
        assert scales > 1, 'Res2Net degenerates to ResNet when scales = 1.'

        mid_channels = out_channels // self.expansion
        width = int(math.floor(mid_channels * (base_width / base_channels)))

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width * scales, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.out_channels, postfix=3)
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            width * scales,
            kernel_size=1,
            stride=self.conv1_stride,
            groups=scales if group_wise else 1,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        self.convs = ModuleList()
        self.bns = ModuleList()
        for i in range(scales - 1):
            self.convs.append(
                build_conv_layer(
                    self.conv_cfg,
                    width,
                    width,
                    kernel_size=3,
                    stride=self.conv2_stride,
                    padding=self.dilation,
                    dilation=self.dilation,
                    bias=False))
            self.bns.append(
                build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width * scales,
            self.out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.scales = scales
        self.width = width
        delattr(self, 'conv2')
        delattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            # identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            spx = list(torch.split(out, self.width, 1))
            spx.reverse()

            sp = self.convs[0](spx[0].contiguous())
            sp = self.relu(self.bns[0](sp))
            out = sp
            for i in range(1, self.scales - 1):
                # fusion
                sp = sp + spx[i]
                sp = self.convs[i](sp.contiguous())
                sp = self.relu(self.bns[i](sp))
                out = torch.cat((out, sp), 1)

            out = torch.cat((out, spx[self.scales - 1]), 1)

            out = self.conv3(out)
            out = self.norm3(out)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Res2Layer(Sequential):
    """Res2Layer to build Res2Net style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck. Defaults to True.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        scales (int): Scales used in Res2Net. Default: 4
        base_width (int): Basic width of each scale. Default: 26
    """

    def __init__(self,
                 block,
                 in_channels,
                 out_channels,
                 num_blocks,
                 expansion=4,
                 stride=1,
                 avg_down=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 scales=4,
                 base_width=26,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or in_channels != out_channels:
            if avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False),
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                )
            else:
                downsample = nn.Sequential(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                )

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                scales=scales,
                base_width=base_width,
                stage_type='stage',
                expansion=expansion,
                **kwargs))
        in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    scales=scales,
                    base_width=base_width,
                    expansion=expansion,
                    **kwargs))
        super(Res2Layer, self).__init__(*layers)


@BACKBONES.register_module()
class Res2Fusion(nn.Module):

    def __init__(
            self,
            numC_input,
            scales, width,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            expansion=[4, 4, 4],
            fusion_layer=dict()
    ):
        super(Res2Fusion, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        # set layers
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [
                Res2Layer(
                    Bottle2neck,
                    curr_numC,
                    num_channels[i],
                    num_layer[i],
                    stride=stride[i],
                    norm_cfg=norm_cfg,
                    scales=scales[i],
                    base_width=width[i],
                    expansion=expansion[i]
                )
            ]
            curr_numC = num_channels[i]
            layers.append(nn.Sequential(*layer))

        self.layers = nn.Sequential(*layers)
        if fusion_layer is not None:
            self.fusion_layer = FusionLayer(
                **fusion_layer
            )
        else:
            self.fusion_layer = None
        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        if self.fusion_layer is not None:
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(self.fusion_layer, x)
            else:
                x_tmp = self.fusion_layer(x)
        else:
            x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
