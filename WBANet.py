import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from myutils.wavelets_function import DWT_2D, IDWT_2D
import math
import argparse


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# patch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WaveAttention(nn.Module):

    def __init__(self, sr_ratio, dim, heads, dropout):
        super(WaveAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        head_dim = dim // heads
        self.head_dim = dim // heads
        self.sr_ratio = sr_ratio
        self.scale = head_dim ** -0.5
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")

        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )

        self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()

        self.proj = nn.Sequential(
            nn.Linear(dim + dim // 4, dim),
            nn.Dropout(dropout)
        )

        self.apply(self._init_weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        # checkShape("q", q)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.reduce(x)
        x = torch.tensor(x, device=device).type(torch.float16)
        x_dwt = self.dwt(x)
        x_dwt = x_dwt.float()
        x_dwt = self.filter(x_dwt)

        x_dwt = x_dwt.half()
        x_idwt = self.idwt(x_dwt)
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2) * x_idwt.size(-1)).transpose(1, 2)

        x_dwt = x_dwt.float()
        kv = self.kv_embed(x_dwt).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k = kv[0]
        v = kv[1]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)  # N ->H*W
        x = self.proj(torch.cat([x, x_idwt], dim=-1))
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class WSM(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be ' \
                                                                                    'divisible by the patch size. '
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.transformer = WaveAttention(sr_ratio=2, dim=dim, heads=heads, dropout=dropout)
        self.reshape = Rearrange('b (h w) (p1 p2  c) -> b (h p1) (w p2) c', p1=patch_height, p2=patch_width,
                                 h=image_height // patch_height)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x = self.transformer(x, H=4, W=4)
        x = self.reshape(x)
        return x


wsm = WSM(image_size=8, patch_size=2, num_classes=2, dim=12, depth=6, heads=4, mlp_dim=32).to(device)


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.5, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x

        x = self.norm(x)
        copy = x
        copy = copy.permute(0, 3, 1, 2)
        x_global = F.avg_pool2d(copy, x.shape[2], x.shape[3])
        x_global = x_global.permute(0, 2, 3, 1)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)
        s_attn = self.spatial_select(torch.cat([x_local, x_global.repeat(1, x.shape[1], x.shape[2], 1)], dim=-1))
        s_attn = self.gate_fn(s_attn)
        attn = c_attn * s_attn
        return ori_x * attn


class ConcactFeature(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(ConcactFeature, self).__init__()
        patch_size = args.patchsize
        self.catConv = nn.Conv2d(3, 3, kernel_size=1)
        self.norm1 = nn.LayerNorm([3, patch_size, patch_size])
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        x = self.catConv(x)
        x = self.norm1(x)
        x = self.conv(x)
        return x


class WBANet(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(WBANet, self).__init__()
        self.wsm = wsm
        self.bam = BiAttn(in_channels=3)
        self.cf = ConcactFeature(args)
        patch_size = args.patchsize
        self.linear1 = nn.Linear(patch_size * patch_size * 3, 20)
        self.linear2 = nn.Linear(20, 2)
        self.drop = nn.Dropout(0.2)
        self.catConv = nn.Conv2d(6, 3, kernel_size=1)

    def forward(self, img):
        wsmOut = self.wsm(img)
        bamOut = self.bam(img.permute(0, 2, 3, 1))
        catOut = torch.cat((wsmOut, bamOut), 3).permute(0, 3, 1, 2)
        catOut = self.catConv(catOut)

        x = self.cf(catOut)

        out1 = x.view(x.size(0), -1)  # 128 192
        out = self.linear1(out1)
        out = self.drop(out)
        out = self.linear2(out)
        return out
