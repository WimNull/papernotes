import torch
import unfoldNd
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class SwinTransBlock(nn.Module):
    def __init__(self, features=256, hdim=256, ff_dim=256, shiftwin=False):
        super().__init__()
        self.lnpre1 = nn.LayerNorm(features)
        self.q_proj = nn.Linear(features, hdim)
        self.k_proj = nn.Linear(features, hdim)
        self.hdim = hdim
        self.v_proj = nn.Linear(features, features)
        self.mlp = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, features)
        )

    def forward(self, x):
        x = self.lnpre1(x)
        q,k,v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # (N, N)
        att = q@k.transpose(1, 2)/math.sqrt(self.hdim)
        att = F.softmax(att, dim=-1)
        v = att@v
        x = v+x
        x = self.mlp(x)+x
        return x


class PatchMerge(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.unfold = unfoldNd.UnfoldNd(kernel_size=2, stride=2)

        # self.input_resolution = input_resolution
        # self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x = x.view(x.size(0), self.img_size, self.img_size, -1).permute(0, 3, 1, 2)
        x = self.unfold(x).transpose(-1, -2).contiguous()

        # x = x.view(B, H, W, C)
        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        # x = self.norm(x)
        # x = self.reduction(x)

        return x

class SwinTransfomer(nn.Module):
    def __init__(self, patch_size=4, cdim=256, img_size=224, inchannel=3, layers_num=[2, 2, 6, 2], num_classes=1000):
        super().__init__()
        self.partion = unfoldNd.UnfoldNd(kernel_size=patch_size, stride=patch_size)
        features = patch_size*patch_size*inchannel
        patch_num = img_size//4
        self.stages = []
        for i in range(4):
            stage = []
            if i==0:
                stage.append(nn.Linear(features, cdim))
                features = cdim
            else:
                stage.append(PatchMerge(patch_num))
                patch_num = patch_num//2
                stage.append(nn.Linear(features*4, features*2))
                features = features*2

            for j in range(1, layers_num[i]):
                stage.append(SwinTransBlock(features=features))
                stage.append(SwinTransBlock(features=features, shiftwin=True))
            self.stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*self.stages)
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x):
        # patch partition
        x = self.partion(x).transpose(-1, -2).contiguous()
        print(x.shape)
        # stage1-4
        for stage in self.stages:
            x = stage(x)
            print(x.shape)
        return x

def param_num(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params

if __name__=='__main__':
    inp = torch.randn((2, 3, 224, 224))
    model = SwinTransfomer()
    print(param_num(model))
    out = model(inp)
    print(out.shape)