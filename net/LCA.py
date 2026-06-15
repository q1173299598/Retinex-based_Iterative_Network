import torch.nn as nn
import torch.fft
from einops import rearrange
from net.transformer_utils import *  #net.transformer_utils
import numpy as np

    
class FrequencyDomainConv(nn.Module):
    def __init__(self):
        super(FrequencyDomainConv, self).__init__()

    def forward(self, x, kernel):
        # 对输入和卷积核进行傅里叶变换，转化到频域
        x_fft = torch.fft.fft2(x)
        kernel_fft = torch.fft.fft2(kernel, s=x.size()[-2:])  # 调整kernel的尺寸与输入匹配

        # 在频域进行卷积（逐元素相乘）
        output_fft = x_fft * kernel_fft

        # 将结果转回空间域
        output = torch.fft.ifft2(output_fft)
        return torch.real(output)  # 获取实部

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #对每个通道进行全局平均池化，将空间信息压缩为一个标量
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )#一个两层全连接网络，用于生成通道权重。
         #channel // reduction 是为了减少计算量，通过 ReLU 激活，再恢复维度，并通过 Sigmoid 激活生成权重

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)#池化并将形状调整为 [batch_size, channels]
        y = self.fc(y).view(b, c, 1, 1)#通过全连接层生成每个通道的权重，并将其调整回 [batch_size, channels, 1, 1] 的形状
        return x * y.expand_as(x)#按通道加权，重新校准输入特征图的通道重要性

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)

        return x * y


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm=False, leaky=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=True):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() if act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm=True, leaky=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm=True, leaky=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentiveDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm=False, leaky=True):
        super().__init__()
        self.down = Down(in_channels, out_channels, norm=norm, leaky=leaky)
        self.attention = nn.Sequential(
            CALayer(out_channels),
            PALayer(out_channels)
        )

    def forward(self, x):
        return self.attention(self.down(x))


class AttentiveUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, norm=False, leaky=True):
        super().__init__()
        self.up = Up(in_channels, out_channels, bilinear, norm=norm, leaky=leaky)
        self.attention = nn.Sequential(
            CALayer(out_channels),
            PALayer(out_channels)
        )

    def forward(self, x1, x2):
        return self.attention(self.up(x1, x2))


class AttentiveDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm=False, leaky=False):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, norm=norm, leaky=leaky)
        self.attention = nn.Sequential(
            CALayer(out_channels),
            PALayer(out_channels)
        )

    def forward(self, x):
        return self.attention(self.conv(x))

class BaseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(BaseNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.inc = DoubleConv(in_channels, 32, norm=norm)
        self.down1 = Down(32, 64, norm=norm)
        self.down2 = Down(64, 128, norm=norm)
        self.down3 = Down(128, 128, norm=norm)

        self.up1 = Up(256, 64, bilinear=True, norm=norm)
        self.up2 = Up(128, 32, bilinear=True, norm=norm)
        self.up3 = Up(64, 32, bilinear=True, norm=norm)
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class ANSN(BaseNet):
    def __init__(self, in_channels=1, out_channels=1, norm=True):
        super(ANSN, self).__init__(in_channels, out_channels, norm)
        self.outc = OutConv(32, out_channels, act=False)

# Cross Attention Block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

# Intensity Enhancement Layer
class IEL(nn.Module):                  #CDL=IEL
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
       
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x
  
 
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim) # IEL 和 CDL 结构相同
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        self.ansn = ANSN(dim, dim)
        self.se = SELayer(dim)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))#交叉注意力
        x = self.se(x)#自注意力
        x = self.gdfn(self.norm(x))#增强操作
        #x = self.ansn(x)#多尺度特征融合
        x = self.norm(x)#归一化
        x = self.se(x)
        return x

# 修改 I_LCA 模型
class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.se = SELayer(dim)
        self.ansn = ANSN(dim, dim)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))#交叉注意力
        x = self.se(x)#自注意力
        x = self.gdfn(self.norm(x))#增强操作
        #x = self.ansn(x)#多尺度特征融合
        x = self.norm(x)#归一化
        x = self.se(x)
        return x

