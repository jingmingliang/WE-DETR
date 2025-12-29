'''by lyuwenyu, modified to include wavelet processing
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d

from src.core import register

__all__ = ['WFCbackbone']

ResNet_cfg = {
    50: [3, 4, 6, 3],
}

donwload_url = {
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
}


def apply_wavelet_transform_single(x):

    haar_filter_ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    haar_filter_lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    haar_filter_hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    haar_filter_hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    
    B, C, H, W = x.shape
    ll = F.conv2d(x.reshape(B * C, 1, H, W), haar_filter_ll, stride=2, padding=0)
    lh = F.conv2d(x.reshape(B * C, 1, H, W), haar_filter_lh, stride=2, padding=0)
    hl = F.conv2d(x.reshape(B * C, 1, H, W), haar_filter_hl, stride=2, padding=0)
    hh = F.conv2d(x.reshape(B * C, 1, H, W), haar_filter_hh, stride=2, padding=0)
    

    ll = ll.view(B, C, ll.shape[-2], ll.shape[-1])
    lh = lh.view(B, C, lh.shape[-2], lh.shape[-1])
    hl = hl.view(B, C, hl.shape[-2], hl.shape[-1])
    hh = hh.view(B, C, hh.shape[-2], hh.shape[-1])
    
    return ll, lh, hl, hh


def multi_level_haar_wavelet_transform(X):

 
    ll1, lh1, hl1, hh1 = apply_wavelet_transform_single(X)

    ll2, lh2, hl2, hh2 = apply_wavelet_transform_single(ll1)

    return {
        'level1': (ll1, lh1, hl1, hh1),
        'level2': (ll2, lh2, hl2, hh2)
    }


class ImprovedWaveletBranch(nn.Module):

    def __init__(self, out_channels=64, internal_channels=64, level=1):
        super().__init__()
        self.level = level  
        self.internal_channels = internal_channels
        self.out_channels = out_channels
        

        context_kernel_size = 7 if level == 1 else 5
        

        expand_channels = 3

        

        self.ll_context = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, 
                     context_kernel_size, 
                     padding=context_kernel_size // 2,
                     groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_channels, expand_channels, 1, bias=False), 
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )
        self.hl_context = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, 
                     context_kernel_size,  
                     padding=context_kernel_size // 2,
                     groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_channels, expand_channels, 1, bias=False),  
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )
        self.lh_context = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, 
                     context_kernel_size,  
                     padding=context_kernel_size // 2,
                     groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_channels, expand_channels, 1, bias=False),  
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )
        self.hh_context = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, 
                     context_kernel_size,  
                     padding=context_kernel_size // 2,
                     groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_channels, expand_channels, 1, bias=False),  
            nn.BatchNorm2d(expand_channels),
            nn.ReLU(inplace=True)
        )


        self.register_buffer('h_kernel', torch.tensor([
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
        

        self.register_buffer('v_kernel', torch.tensor([
            [1.,  2.,  1.],
            [0.,  0.,  0.],
            [-1., -2., -1.]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)

        self.register_buffer('d_kernel', torch.tensor([
            [-1., -1., -1.],
            [-1.,  8., -1.],
            [-1., -1., -1.]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)
        

        self.fusion = nn.Sequential(
            nn.Conv2d(expand_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def apply_directional_conv(self, x, kernel):

        B, C, H, W = x.shape

        kernel_repeated = kernel.repeat(C, 1, 1, 1)
        out = F.conv2d(x, kernel_repeated, padding=1, groups=C)
        return out
    
    def forward(self, ll, lh, hl, hh):

        ll_feat = self.ll_context(ll)  
        

        ll_h = self.apply_directional_conv(ll_feat, self.h_kernel)  
        ll_v = self.apply_directional_conv(ll_feat, self.v_kernel)  
        ll_d = self.apply_directional_conv(ll_feat, self.d_kernel)  

        

        lh_feat = self.lh_context(lh)  
        hl_feat = self.hl_context(hl)  
        hh_feat = self.hh_context(hh)  
        

        lh_final = lh_feat + ll_h  
        hl_final = hl_feat + ll_v  
        hh_final = hh_feat + ll_d  
        

        concat = torch.cat([ll_feat, lh_final, hl_final, hh_final], dim=1)
        output = self.fusion(concat)
        
        return output, ll_feat


class CrossAttentionFusion(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, feat_main, feat_wave):

        main_ca = feat_main * self.channel_attention(feat_main)
        

        wave_spatial = torch.cat([
            torch.mean(feat_wave, dim=1, keepdim=True),
            torch.max(feat_wave, dim=1, keepdim=True)[0]
        ], dim=1)
        wave_sa = feat_wave * self.spatial_attention(wave_spatial)
        
        path1 = main_ca + wave_sa
        

        main_spatial = torch.cat([
            torch.mean(feat_main, dim=1, keepdim=True),
            torch.max(feat_main, dim=1, keepdim=True)[0]
        ], dim=1)
        main_sa = feat_main * self.spatial_attention(main_spatial)
        
        wave_ca = feat_wave * self.channel_attention(feat_wave)
        
        path2 = main_sa + wave_ca
        

        output = path1 + path2
        
        return output


class BasicBlock(nn.Module):
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
        self.act = nn.Identity() if act is None else get_activation(act)

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
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

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


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in,
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    act=act)
            )

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out

@register
class WFCbackbone(nn.Module):
    def __init__(
            self,
            depth,
            variant='d',
            num_stages=4,
            return_idx=[0, 1, 2, 3],
            act='relu',
            freeze_at=-1,
            freeze_norm=True,
            pretrained=False):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
            )
            ch_in = _out_channels[i]


        self.wavelet_branch1 = ImprovedWaveletBranch(
            internal_channels=16, out_channels=64, level=1)    
        self.wavelet_branch2 = ImprovedWaveletBranch(
            internal_channels=24, out_channels=256, level=2)    


        self.adaptive_fusion1 = CrossAttentionFusion(64, reduction=4)  
        self.adaptive_fusion2 = CrossAttentionFusion(256, reduction=4)   

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            self.load_state_dict(state,
                                 strict=False)  
            print(f'Load PResNet{depth} state_dict')

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        # Apply wavelet transform - now returns separate subbands
        wavelet_dict = multi_level_haar_wavelet_transform(x)
        
        # Get level 1 subbands (H/2, W/2)
        ll1, lh1, hl1, hh1 = wavelet_dict['level1']
        
        # Get level 2 subbands (H/4, W/4)
        ll2, lh2, hl2, hh2 = wavelet_dict['level2']


        x_2, ll_2 = self.wavelet_branch1(ll1, lh1, hl1, hh1)  
        x_4, ll_4 = self.wavelet_branch2(ll2, lh2, hl2, hh2)  

        conv1 = self.conv1(x)  # H/2 × W/2
        

        conv1 = self.adaptive_fusion1(conv1, x_2)

        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1) 

        
        outs = []

        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            

            if idx == 0:
                x = self.adaptive_fusion2(x, x_4)  
                
            if idx in self.return_idx:
                outs.append(x)

        return outs