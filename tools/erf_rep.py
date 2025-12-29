'''ERF-adapted version of encoder_s42.py for effective receptive field analysis
Modified to work as a standalone feature extractor without classification head
Integrates real xb.py backbone for accurate architecture analysis
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import os
import sys
from timm.models.layers import DropPath
from collections import OrderedDict

# RepLKBlock related functions
def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

# Common layers from xb.py
def get_activation(act: str, inplace: bool = True):
    if act is None:
        return nn.Identity()
    elif isinstance(act, nn.Module):
        return act
    act = act.lower()
    if act == 'swish':
        return nn.SiLU(inplace=inplace)
    elif act == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif act == 'silu':
        return nn.SiLU(inplace=inplace)
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise RuntimeError('')

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size, stride, groups=1, norm_type='bn', norm_decay=0., norm_groups=32, use_dcn=False, bias_on=False, lr_scale=1., freeze_norm=False, initializer=None, skip_quant=False, dcn_lr_scale=2., dcn_regularizer=None, act='relu'):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', None]

        if bias_on:
            bias_attr = True
        else:
            bias_attr = False

        if not use_dcn:
            self.conv = nn.Conv2d(
                in_channels=ch_in, 
                out_channels=ch_out,
                kernel_size=filter_size, 
                stride=stride, 
                padding=(filter_size - 1) // 2,
                groups=groups, 
                bias=bias_attr)
        else:
            # DCN implementation would go here
            raise NotImplementedError("DCN not implemented for ERF analysis")

        norm_lr = 0. if freeze_norm else 1.
        param_attr = None
        bias_attr = None
        
        if norm_type in ['bn', 'sync_bn']:
            self.norm = nn.BatchNorm2d(ch_out)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out)
        else:
            self.norm = None

        act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act
        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

# Wavelet transform from xb.py
def multi_level_haar_wavelet_transform(X):
    """Multi-level 2D discrete Haar wavelet transform"""
    device = X.device
    haar_filter = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=X.dtype, device=device).unsqueeze(0).unsqueeze(0)

    def apply_wavelet_transform(x):
        B, C, H, W = x.shape
        low_pass = F.conv2d(
            x.view(B * C, 1, H, W),
            haar_filter,
            stride=2,
            padding=0
        )
        return low_pass.view(B, C, low_pass.shape[-2], low_pass.shape[-1])

    x2 = apply_wavelet_transform(X)
    x4 = apply_wavelet_transform(x2)
    return x2, x4

class SpacialAttention(nn.Module):
    """Spatial attention from xb.py"""
    def __init__(self, kernel_size=7):
        super(SpacialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out = torch.max(x, dim=1, keepdim=True).values
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim=1)
        out = self.sigmoid(self.conv(pool_out))
        return out * x

class SmallObjectEnhancer(nn.Module):
    """Small object enhancer from xb.py"""
    def __init__(self, channels):
        super(SmallObjectEnhancer, self).__init__()
        self.local_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),  # depthwise
            nn.Conv2d(channels, channels, 1, 1, 0),  # pointwise
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        local_feat = self.local_enhance(x)
        att_map = self.attention(local_feat)
        enhanced = local_feat * att_map
        return x + enhanced

class BasicBlock(nn.Module):
    """BasicBlock from xb.py"""
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()
        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)
        return out

class BottleNeck(nn.Module):
    """BottleNeck from xb.py"""
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()
        
        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1, act=None)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1, act=None))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride, act=None)

        self.act = get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)
        return out

class RepLKBlock(nn.Module):
    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path):
        super().__init__()
        self.pw1 = conv_bn(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = conv_bn(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                    stride=1, padding=block_lk_size//2, groups=dw_channels)
        self.lk_origin = conv_bn(in_channels=dw_channels, out_channels=dw_channels, kernel_size=small_kernel,
                                 stride=1, padding=small_kernel//2, groups=dw_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        out = self.pw1(x)
        out = F.relu(out)
        out = self.large_kernel(out) + self.lk_origin(out)
        out = F.relu(out)
        out = self.pw2(out)
        return x + self.drop_path(out)

class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiScaleRepLKEnhancer(nn.Module):
    """多尺度RepLK增强器 - 结合多尺度检测和大核增强的小目标特征增强模块"""
    def __init__(self, hidden_dim):
        super().__init__()
        # 多尺度前景检测器
        self.multi_scale_detectors = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2, bias=False)
            for k in [1, 3, 7]
        ])
        
        # RepLKBlock增强器
        self.small_obj_enhancer = RepLKBlock(
            in_channels=hidden_dim,
            dw_channels=hidden_dim * 2,
            block_lk_size=13,
            small_kernel=3,
            drop_path=0.1
        )
        
        # 维度压缩
        self.dim_reduction = nn.Conv2d(hidden_dim*3, hidden_dim, 1, bias=False)
        
        # SE注意力
        self.channel_attention = SEAttention(hidden_dim, reduction=16)

    def forward(self, x):
        # 多尺度检测
        multi_scale_feats = [detector(x) for detector in self.multi_scale_detectors]
        fused_feat = torch.cat(multi_scale_feats, dim=1)
        fused_feat = self.dim_reduction(fused_feat)
        
        # RepLK增强
        enhanced_feat = self.small_obj_enhancer(fused_feat)
        
        # SE注意力
        enhanced_feat = self.channel_attention(enhanced_feat)
        
        return enhanced_feat + x

class LightweightRepLKRefiner(nn.Module):
    """轻量级RepLK细化器 - 专门用于L1层的高分辨率特征细化"""
    def __init__(self, hidden_dim):
        super().__init__()
        # 轻量级RepLKBlock
        self.lightweight_enhancer = RepLKBlock(
            in_channels=hidden_dim,
            dw_channels=hidden_dim,
            block_lk_size=7,
            small_kernel=3,
            drop_path=0.05
        )
        
        # SE注意力
        self.channel_attention = SEAttention(hidden_dim, reduction=16)

    def forward(self, x):
        # RepLK增强
        enhanced_feat = self.lightweight_enhancer(x)
        
        # SE注意力
        enhanced_feat = self.channel_attention(enhanced_feat)
        
        return enhanced_feat + x

class XBResNet(nn.Module):
    """Complete xb.py ResNet backbone for ERF analysis"""
    
    def __init__(self, depth=50, variant='d', return_idx=[1, 2, 3], act='relu', freeze_at=-1, freeze_norm=False, pretrained=False):
        super().__init__()
        
        ResNet_cfg = {50: [3, 4, 6, 3]}
        layers = ResNet_cfg[depth]
        self.act = act
        self.depth = depth
        self.variant = variant
        
        # Channel configurations
        _out_strides = [4, 8, 16, 32]
        _out_channels = [64, 256, 512, 1024, 2048]
        
        # Initial conv layer
        self.conv1 = ConvNormLayer(ch_in=3, ch_out=64, filter_size=7, stride=2, act=act)
        
        # Residual layers
        self.res_layers = nn.ModuleList()
        ch_in = 64
        for i, num_blocks in enumerate(layers):
            stride = 2 if i > 0 else 1
            ch_out = _out_channels[i + 1] // BottleNeck.expansion
            
            res_layer = nn.Sequential()
            for j in range(num_blocks):
                block_stride = stride if j == 0 else 1
                shortcut = False if j == 0 and (ch_in != ch_out * BottleNeck.expansion) else True
                
                res_layer.add_module(
                    f'block_{j}',
                    BottleNeck(ch_in, ch_out, block_stride, shortcut, act=act, variant=variant)
                )
                ch_in = ch_out * BottleNeck.expansion
            
            self.res_layers.append(res_layer)
        
        # Wavelet processing components
        self.x2 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.x4 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.x8 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        
        # Small object enhancer
        self.small_enhancer1 = SmallObjectEnhancer(64)
        
        # Spatial attention modules
        self.SpacialAttention1 = SpacialAttention()
        self.SpacialAttention2 = SpacialAttention()
        self.SpacialAttention3 = SpacialAttention()
        self.SpacialAttention4 = SpacialAttention()
        self.SpacialAttention5 = SpacialAttention()
        self.SpacialAttention6 = SpacialAttention()
        
        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]
        
    def forward(self, x):
        # Apply wavelet transform
        x2, x4 = multi_level_haar_wavelet_transform(x)
        
        # Process wavelet components
        x_2 = self.x2(x2)
        x_4 = self.x4(x4)
        x_8 = self.x8(x4)
        
        # Apply spatial attention to wavelet components
        x_2 = self.SpacialAttention1(x_2)
        x_4 = self.SpacialAttention2(x_4)
        x_8 = self.SpacialAttention3(x_8)
        
        # Process input through initial conv layers
        conv1 = self.conv1(x)
        
        # Apply spatial attention to conv1 output and add wavelet component
        conv1 = self.SpacialAttention4(conv1) + x_2
        
        # Max pooling
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        x = self.small_enhancer1(x)
        
        outs = []
        # Process through residual blocks with wavelet fusion at different stages
        for idx, stage in enumerate(self.res_layers):
            if idx == 0:
                # Apply spatial attention and add wavelet component before stage 1
                x = self.SpacialAttention5(x) + x_4
            elif idx == 1:
                # Apply spatial attention and add wavelet component before stage 2
                x = self.SpacialAttention6(x) + x_8
            
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        
        return outs

class EncoderS42ForERF(nn.Module):
    """ERF分析专用的encoder_s42模型，集成真实的xb.py backbone"""
    
    def __init__(self, hidden_dim=256, enable_enhancement=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enable_enhancement = enable_enhancement
        
        # 真实的xb.py backbone
        self.backbone = XBResNet(depth=50, return_idx=[1, 2, 3])  # 返回S3, S4, S5
        
        # 特征投影层 (匹配真实的通道数)
        self.input_proj = nn.ModuleList([
            nn.Conv2d(512, hidden_dim, 1),   # S3: 80x80
            nn.Conv2d(1024, hidden_dim, 1),  # S4: 40x40  
            nn.Conv2d(2048, hidden_dim, 1),  # S5: 20x20
        ])
        
        # RepLK增强模块
        if enable_enhancement:
            self.s4_enhancer = MultiScaleRepLKEnhancer(hidden_dim)  # L2层增强
            self.s3_enhancer = LightweightRepLKRefiner(hidden_dim)  # L1层增强
        
        # 简化的FPN用于特征融合
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1) for _ in range(3)
        ])
        
        # 最终输出投影
        self.output_proj = nn.Conv2d(hidden_dim, 1, 1)  # 输出单通道用于ERF分析
        
    def forward(self, x):
        # 使用真实的xb.py backbone提取特征
        backbone_feats = self.backbone(x)  # [S3, S4, S5]
        
        # 特征投影到统一维度
        proj_feats = [
            self.input_proj[i](feat) for i, feat in enumerate(backbone_feats)
        ]
        
        # 应用RepLK增强模块 (如果启用)
        if self.enable_enhancement:
            proj_feats[1] = self.s4_enhancer(proj_feats[1])  # S4增强 (40x40)
            proj_feats[0] = self.s3_enhancer(proj_feats[0])  # S3增强 (80x80)
        
        # 简化的FPN融合
        for i, conv in enumerate(self.fpn_convs):
            proj_feats[i] = conv(proj_feats[i])
        
        # 使用S4特征作为主要输出 (40x40分辨率适合ERF分析)
        output = self.output_proj(proj_feats[1])
        print(output.shape)
        return output

# 创建模型实例的便捷函数
def create_baseline_model():
    """创建无增强的baseline模型"""
    return EncoderS42ForERF(hidden_dim=256, enable_enhancement=False)

def create_enhanced_model():
    """创建有RepLK增强的模型"""
    return EncoderS42ForERF(hidden_dim=256, enable_enhancement=True)
