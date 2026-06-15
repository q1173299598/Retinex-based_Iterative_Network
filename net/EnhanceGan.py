import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from einops import rearrange
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from net.CIDGAN import CIDNet  # 引入CIDNet

class Generator(nn.Module):
    def __init__(self, cidnet):
        super(Generator, self).__init__()
        self.cidnet = cidnet  # 使用CIDNet作为生成器

    def forward(self, x):
        return self.cidnet(x)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 初始化生成器和多个判别器
cidnet = CIDNet().cuda()
G = Generator(cidnet).cuda()
D1 = Discriminator().cuda()
D2 = Discriminator().cuda()
D3 = Discriminator().cuda()  # 假设有三个判别器，分别评估亮度、对比度和细节
