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
    """获取卷积层，支持大核卷积优化"""
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if (has_large_impl and in_channels == out_channels and out_channels == groups and 
        use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1):
        try:
            sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
        except ImportError:
            pass
    
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                     kernel_size=kernel_size, stride=stride, padding=padding, 
                     dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                    stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    """融合BatchNorm到卷积层"""
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

# RepLKBlock核心组件
class ReparamLargeKernelConv(nn.Module):
    """重参数化大核卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                     stride=stride, padding=padding, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                         stride=stride, padding=small_kernel//2, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

class RepLKBlock(nn.Module):
    """RepLK大核卷积块"""
    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel, drop_path, small_kernel_merged=False):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = get_bn(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)

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
    """小目标增强模块 - 完全按照xb.py实现"""
    def __init__(self, channels, reduction=16):
        super(SmallObjectEnhancer, self).__init__()
        
        # 局部感受野增强
        self.local_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # SE注意力机制 (没有spatial attention)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        local_feat = self.local_enhance(x)
        # att_map = self.attention(local_feat)  # 在xb.py中这行被注释掉了
        
        b, c, _, _ = local_feat.size()
        y = self.avg_pool(local_feat).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + local_feat * y

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
    """Squeeze-and-Excitation 通道注意力
    参考: Hu et al., Squeeze-and-Excitation Networks (CVPR 2018)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        assert channels % reduction == 0, "channels must be divisible by reduction"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # 标准全局平均池化
        pooled = self.avg_pool(x)
        scale = self.fc(pooled)  # [B,C,1,1]
        return x * scale  # 应用注意力权重

class MultiScaleRepLKEnhancer(nn.Module):
    """多尺度RepLK增强器 - 结合多尺度检测和大核增强的小目标特征增强模块"""
    def __init__(self, hidden_dim):
        super().__init__()
        # 优化的多尺度前景检测器 - 与RepLKBlock(13×13)形成互补体系
        # 1×1: 点特征, 3×3: 局部结构, 7×7: 中等上下文, 13×13: 大上下文(RepLKBlock)
        self.multi_scale_detectors = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2, bias=False)
            for k in [1, 3, 7]  # 优化的尺度分布
        ])
        
        # 小目标专用增强器 - 使用RepLKBlock
        self.small_obj_enhancer = RepLKBlock(
            in_channels=hidden_dim,
            dw_channels=hidden_dim * 2,  # 扩展通道数以增强表达能力
            block_lk_size=13,  # 大核尺寸，适合捕获小目标周围的上下文信息
            small_kernel=3,    # 小核用于重参数化
            drop_path=0.1      # 轻微的drop path用于正则化
        )
        
        # 简化的维度压缩
        self.dim_reduction = nn.Conv2d(hidden_dim*3, hidden_dim, 1, bias=False)
        
        # 混合注意力机制 - 结合通道和空间注意力
        self.channel_attention = SEAttention(hidden_dim, reduction=16)

    def forward(self, x):
        # 优化的计算顺序：先多尺度检测，再RepLKBlock增强
        # 让RepLKBlock在更丰富的多尺度特征基础上工作
        
        # 多尺度前景检测
        multi_scale_feats = []
        for detector in self.multi_scale_detectors:
            feat = detector(x)
            multi_scale_feats.append(feat)
        
        # 拼接多尺度特征
        concat_feats = torch.cat(multi_scale_feats, dim=1)
        
        # 维度压缩
        multi_scale_reduced = self.dim_reduction(concat_feats)
        
        # 小目标专用增强 - 在多尺度特征基础上进行
        enhanced = self.small_obj_enhancer(multi_scale_reduced)
        
        # 混合注意力机制
        # 1. 通道注意力 (SEAttention已经内置了权重应用)
        channel_refined = self.channel_attention(enhanced)
        
        # 残差连接 - 关键！确保梯度能够正常传播
        return channel_refined + x

class LightweightRepLKRefiner(nn.Module):
    """轻量级RepLK细化器 - 专门用于高分辨率特征的轻量级增强"""
    def __init__(self, channels):
        super().__init__()
        
        # 轻量级特征增强器 - 使用RepLKBlock替代原有的深度可分离卷积
        # 参数设计：较小的大核尺寸7适合L1层，保持通道数不变，轻量化设计
        self.local_enhance = RepLKBlock(
            in_channels=channels,
            dw_channels=channels,      # 保持通道数不变，轻量化设计
            block_lk_size=7,          # 适中的大核尺寸，适合L1层的细节增强
            small_kernel=3,           # 小核用于重参数化
            drop_path=0.05            # 更小的drop path，保持稳定性
        )
        
        # 标准SE通道注意力
        self.se_attention = SEAttention(channels, reduction=16)
        
    def forward(self, x):
        """
        Args:
            x: L1特征 (80×80)
        Returns:
            enhanced: 增强后的L1特征
        """
        # 轻量级特征增强
        enhanced = self.local_enhance(x)
        
        # 使用标准SE通道注意力 (SEAttention已经内置了权重应用)
        guided = self.se_attention(enhanced)
        
        # 残差连接 - 确保梯度能够正常传播
        return guided + x

# Transformer组件
class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = get_activation(activation)
        self.normalize_before = normalize_before

    def forward(self, src, src_mask=None, pos_embed=None):
        if pos_embed is not None:
            src = src + pos_embed
        
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = residual + self.dropout1(src2)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

# CSP和FPN组件
class CSPRepLayer(nn.Module):
    """CSP RepVGG层"""
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=None, act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, act=act)
        self.bottlenecks = nn.Sequential(*[
            ConvNormLayer(hidden_channels, hidden_channels, 3, 1, act=act) for _ in range(num_blocks)
        ])
        self.conv3 = ConvNormLayer(hidden_channels * 2, out_channels, 1, 1, act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(torch.cat((x_1, x_2), dim=1))

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
            self.s4_enhancer = MultiScaleRepLKEnhancer(hidden_dim)  # S4层增强
        
        # 移除basic_conv，测试backbone是否已经提供足够的梯度传播
        # self.basic_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        # 最终输出投影
        self.output_proj = nn.Conv2d(hidden_dim, 1, 1)  # 输出单通道用于ERF分析
        
    def forward(self, x):
        # 使用真实的xb.py backbone提取特征
        backbone_feats = self.backbone(x)  # [S3, S4, S5]
        
        # 只处理S4特征 (40x40分辨率，最适合ERF分析)
        s4_feat = self.input_proj[1](backbone_feats[1])  # S4特征投影
        
        # 应用RepLK增强模块 (如果启用)
        if self.enable_enhancement:
            s4_feat = self.s4_enhancer(s4_feat)  # S4 RepLK增强
            print("✓ Applied RepLK enhancement to S4 features")
        else:
            print("✓ Using baseline S4 features (no enhancement)")
        
        # 测试：移除basic_conv，直接使用backbone+input_proj的特征
        # s4_feat = self.basic_conv(s4_feat)
        
        # 直接输出S4特征用于ERF分析
        output = self.output_proj(s4_feat)
        print(f"Output shape: {output.shape}")
        return output

# 创建模型实例的便捷函数
def create_baseline_model():
    """创建无增强的baseline模型"""
    return EncoderS42ForERF(hidden_dim=256, enable_enhancement=False)

def create_enhanced_model():
    """创建有RepLK增强的模型"""
    return EncoderS42ForERF(hidden_dim=256, enable_enhancement=True)
