import torch
import math
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def compute_l1_loss(input, output):
    return torch.mean(torch.abs(input-output))
        
def loss_Textures(x, y, nc=3, alpha=1.2, margin=0):
  xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
  yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
  
  xi2 = torch.sum(xi * xi, dim=2)
  yi2 = torch.sum(yi * yi, dim=2)
  #pdb.set_trace()    #15*32*32
  out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
  
  return torch.mean(out)

class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = models.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '13': "relu3",
            '22': "relu4",
            '31': "relu5",        #1_2 to 5_2
        }
        
    def forward(self, x):
        output = {}
        #import pdb
        #pdb.set_trace()
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        
        return output
        
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class WaveletTransform(nn.Module):
    def __init__(self, scale=1, dec=True, params_path='wavelet_weights_c2.pkl', transpose=True):
        super(WaveletTransform, self).__init__()
        self.scale = scale
        self.dec = dec
        self.transpose = transpose
        ks = int(math.pow(2, self.scale))
        nc = 3 * ks * ks

        if dec:
            self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0, groups=3, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path, 'rb')
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                dct = u.load()
                #dct = pickle.load(f)
                f.close()
                m.weight.data = torch.from_numpy(dct['rec%d' % ks])
                m.weight.requires_grad = False

    def forward(self, x):
        if self.dec:
            # pdb.set_trace()
            output = self.conv(x)
            if self.transpose:
                osz = output.size()
                # print(osz)
                output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1, 2).contiguous().view(xsz)
            output = self.conv(xx)
        return output


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer

class DMDB2(nn.Module):
    """
    DeMoireing  Dense Block
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=1):
        super(DMDB2, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

        self.deli = nn.Sequential(
            Conv2d(64, 64 , 3, stride=1, dilation=delia),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deli2 = nn.Sequential(
            Conv2d(64, 64 , 3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #self.sam1 = SAM(64,64,1)
        #self.sam2 = SAM(64,64,1)
    def forward(self, x):
        #att1 = self.sam1(x)
        #att2 = self.sam2(x)

        out = self.RDB1(x)
        out = out+x
        out2 = self.RDB2(out)
        
        out3 = self.deli(x)+0.2*self.deli2(self.deli(x))
        return out2.mul(0.2)+ out3


class WDNet(nn.Module):
    def __init__(self,in_channel=3):
        super(WDNet,self).__init__()

        self.cascade1=nn.Sequential(
            nn.Conv2d(48, 64 , 1 , stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 , 3 , stride=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.cascade2=nn.Sequential(

            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=1),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=2),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=5),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=7),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=12),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=19),
            
            DMDB2(64, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA',delia=31)
        )
        
        self.final=nn.Sequential(
            conv_block(64,48, kernel_size=1, norm_type=None, act_type=None)
        )
        self.xbranch=nn.Sequential(
            conv_block(3,64, kernel_size=3, norm_type=None, act_type='leakyrelu')
        )
        
        
    def forward(self, x):
        x1 = self.cascade1(x)
        #pdb.set_trace()
        
        x1 = self.cascade2(x1)

        x = self.final(x1)
        
        return x

if __name__=='__main__':
    img_height,img_width = 256, 256
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()#  smoothl1loss()
    tvloss = TVLoss()
    lossmse = torch.nn.MSELoss()
    # if use GAN loss
    lambda_pixel = 100
    patch = (1, img_height//2**4, img_width//2**4)   

    # Initialize wdnet
    generator = WDNet()


    wavelet_dec = WaveletTransform(scale=2, dec=True)
    wavelet_rec = WaveletTransform(scale=2, dec=False) 
