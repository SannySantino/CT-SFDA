import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import CNN,UNet
import copy

#-------------- backbone network --------------#
class CNN1dClassifier(nn.Module):
    def __init__(self, input_d,output_d,configs):
        super(CNN1dClassifier, self).__init__()
        self.ConvBlock=CNN(configs)
        self.midlayer=nn.Linear(input_d,int(input_d/4))
        self.logits = nn.Linear(int(input_d/4),output_d)
        self.norm1D=nn.BatchNorm1d(configs.input_channels)
        self.final_out_channels=configs.final_out_channels
        self.features_len=configs.features_len
        self.stage=configs.stage      

    def forward(self, x):
        x=self.norm1D(x)                                        
        x=self.ConvBlock(x)  
        x=self.midlayer(x)
        predictions = self.logits(x)
        return predictions


#-------------- AdaWarp network --------------#
class AdaWarp(nn.Module):
    def __init__(self, args):
        super(AdaWarp, self).__init__()
        configs=copy.deepcopy(args)
        self.stage=configs.stage
        self.dataset_name=configs.data
        self.is_training=configs.is_training

        if self.dataset_name=='FDA':
            self.C=configs.input_channels=1                                                    # signal channel
            self.ae_channels=8
            self.D=64                                                   # number of windows
            self.L=80                                                   # length of windows
            self.T=self.D*self.L                                        # original length
            self.num_classes=configs.num_classes=3
            self.final_out_channels=configs.final_out_channels=128      # final output channel of 1DCNN
            self.mid_channels=configs.mid_channels=64                   # mid channel number of 1DCNN
            self.features_len=configs.features_len=1                    # final output length of 1DCNN
            self.kernel_size=configs.kernel_size=32                     #kernel size of the first layer in 1DCNN
            self.stride=configs.stride=6                                #stride size of the first layer in 1DCNN

        elif self.dataset_name=='SSC':
            self.C=configs.input_channels=1        
            self.ae_channels=8
            self.D=48
            self.L=64
            self.T=self.D*self.L
            self.num_classes=configs.num_classes=5
            self.final_out_channels=configs.final_out_channels=8
            self.mid_channels=configs.mid_channels=16
            self.features_len=configs.features_len=64
            self.kernel_size=configs.kernel_size=25
            self.stride=configs.stride=6
        
        elif self.dataset_name=='HAR':
            self.C=configs.input_channels=9
            self.ae_channels=8
            self.D=2
            self.L=64
            self.T=self.D*self.L
            self.num_classes=configs.num_classes=6
            self.final_out_channels=configs.final_out_channels=128
            self.mid_channels=configs.mid_channels=64
            self.features_len=configs.features_len=1
            self.kernel_size=configs.kernel_size=5
            self.stride=configs.stride=1
    

        self.reconstructor=UNet(in_ch=self.C,out_ch=self.C)
        self.warpingBlock=WarpingNet(self.C,self.ae_channels)


        self.V = nn.Parameter(torch.tensor(1.0))
        self.backboneNetwork=CNN1dClassifier(configs.features_len * configs.final_out_channels, configs.num_classes,configs)


    def forward(self, x,S=None):
        B=int(x.shape[0])           # batch size
        D=int(self.D)               # number of windows
        L=int(self.L)               # window length
        T=int(self.T)               # original length
        C=int(self.C)               # signal channels
        P=int(L/D-1)                # padding size for multi-channel signal only
        

        signal_2D=x.view(B,C,D,L)                                                   # reshape to 2D
        input=torch.cat([signal_2D,signal_2D],dim=2)
        if C!=1:
            input=torch.nn.functional.pad(signal_2D, (0, 0, P, P))
        output_compatible= self.reconstructor(input)
        output_incompatible=self.warpingBlock(output_compatible)
        if self.stage !='tta': 
            output_incompatible=self.V*output_incompatible+output_compatible        #two-branch structure

        elif self.stage == 'tta':
            output_incompatible=self.V*output_incompatible+S*output_compatible           


        y_incompatible=output_incompatible[:,:,:D,:].view(B,C,T)
        if C!=1:
            y_incompatible=output_incompatible[:, :, P:P + D, :]
        y_compatible=output_compatible[:,:,:D,:].view(B,C,T)
        if C!=1:
            y_compatible=output_compatible[:, :, P:P + D, :]
   
        y_incompatible=y_incompatible.view(B,C,T)                                   # reshape to 1D
        y_compatible=y_compatible.view(B,C,T)                                       # reshape to 1D

        if self.stage=='stage1':
            return y_incompatible,y_compatible,None
        elif self.stage=='stage2':
            output=output_compatible
        else:
            output=output_incompatible
        signal_2D=output[:,:,:D,:].view(B,C,T)
        if C!=1:
            signal_2D=output[:, :, P:P + D, :]
        
        signal_2D=signal_2D.view(B,C,T)                                              # reshape to 1D
        cls=self.backboneNetwork(signal_2D)              

        return y_incompatible,y_compatible,cls

       

#-------------- WarpingNet --------------#
class ResidualBlock(nn.Module):
    def __init__(self,channel):
        super(ResidualBlock,self).__init__()
        self.channel=channel
        self.conv1=nn.Conv2d(channel,channel,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(channel,channel,kernel_size=3,padding=1)
 
    def forward(self,x):
        y=F.relu(self.conv1(x))
        y=self.conv2(y)
        return F.relu(x+y)


class WarpingNet(nn.Module):

    def __init__(self, input_dim, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
                                     ResidualBlock(dim), ResidualBlock(dim))
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))
        self.n_downsample = 2


    def forward(self, x):
        ze = self.encoder(x)
        x_hat = self.decoder(ze)
        return x_hat



