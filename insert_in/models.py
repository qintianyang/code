import os
from torch import load
import torch
import torch.nn as nn
from typing import Tuple


class CCNN(nn.Module):
    def __init__(self, in_channels: int = 4, grid_size: Tuple[int, int] = (9, 9), num_classes: int = 2, dropout: float = 0.5):
        super(CCNN, self).__init__()
        self.in_channels = in_channels 
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(self.in_channels, 64, kernel_size=4, stride=1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(64, 128, kernel_size=4, stride=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(128, 256, kernel_size=4, stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ZeroPad2d((1, 2, 1, 2)), nn.Conv2d(256, 64, kernel_size=4, stride=1), nn.ReLU())

        self.lin1 = nn.Sequential(
            nn.Linear(self.grid_size[0] * self.grid_size[1] * 64, 1024),
            # nn.Linear(262144, 1024),
            nn.SELU(), # Not mentioned in paper
            nn.Dropout2d(self.dropout)
        )
        self.lin2 = nn.Linear(1024, self.num_classes)

    def feature_dim(self):
        return self.grid_size[0] * self.grid_size[1] * 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 4, 9, 9]`. Here, :obj:`n` corresponds to the batch size, :obj:`4` corresponds to :obj:`in_channels`, and :obj:`(9, 9)` corresponds to :obj:`grid_size`.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.flatten(start_dim=1)
        x_out2 = x

        x_feature = self.lin1(x)
        x_out = self.lin2(x_feature)
        return x_out, x_feature,x_out2



class TSCeption(nn.Module):
    

    def __init__(self,
                 num_electrodes: int = 28,
                 num_T: int = 15,
                 num_S: int = 15,
                 in_channels: int = 1,
                 hid_channels: int = 32,
                 num_classes: int = 2,
                 sampling_rate: int = 128,
                 dropout: float = 0.5):
        # input_size: 1 x EEG channel x datapoint
        super(TSCeption, self).__init__()
        self.num_electrodes = num_electrodes
        self.num_T = num_T
        self.num_S = num_S
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.dropout = dropout

        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(
            in_channels, num_T,
            (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(
            in_channels, num_T,
            (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(
            in_channels, num_T,
            (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(num_electrodes), 1),
                                         1, int(self.pool * 0.25))
        self.Sception2 = self.conv_block(num_T, num_S,
                                         (int(num_electrodes * 0.5), 1),
                                         (int(num_electrodes * 0.5), 1),
                                         int(self.pool * 0.25),
                                         padding=(0, 0, 1, 0) if num_electrodes % 2 == 1 else 0)
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(nn.Linear(num_S, hid_channels), nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(hid_channels, num_classes))

    def conv_block(self, in_channels: int, out_channels: int, kernel: int,
                   stride: int, pool_kernel: int, padding: int = 0) -> nn.Module:
        return nn.Sequential(
            nn.ZeroPad2d(padding) if padding != 0 else nn.Identity(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel,
                      stride=stride), nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 1, 28, 512]`. Here, :obj:`n` corresponds to the batch size, :obj:`1` corresponds to number of channels for convolution, :obj:`28` corresponds to :obj:`num_electrodes`, and :obj:`512` corresponds to the input dimension for each electrode.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out_1 =self.fc[0](out)
        out_2 = self.fc[1](out_1)
        out_3 = self.fc[2](out_2)
        out_4 = self.fc[3](out_3)
        out = self.fc(out)
        return out, out_4, out_1

    def feature_dim(self):
        return self.num_S



def get_model(architecture):
    match architecture: 
        case "CCNN":
            # from torcheeg.models import CCNN
            return CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

        case "TSCeption":
            return TSCeption(
                num_classes=2,
                num_electrodes=28,
                sampling_rate=128,
                num_T=60,
                num_S=60,
                hid_channels=128,
                dropout=0.5,
            )

        case "EEGNet":
            from torcheeg.models import EEGNet

            return EEGNet(
                chunk_size=128,
                num_electrodes=32,
                dropout=0.5,
                kernel_1=64,
                kernel_2=16,
                F1=16,
                F2=32,
                D=4,
                num_classes=16,
            )

        case _:
            raise ValueError(f"Invalid architecture: {architecture}")


def load_model(model, model_path):

    state_dict = load(model_path)
    # state_dict = load(model_path)["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace("model.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    return model


def load_model_v1(model, model_path):
    
    state_dict = load(model_path)
    # state_dict = load(model_path)["state_dict"]
    # for key in list(state_dict.keys()):
    #     state_dict[key.replace("model.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    return model

import torch
from collections import OrderedDict
def load_model_v2(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    
    # 去掉 "model." 前缀
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[6:]  # 去掉 "model." 前缀
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # 加载模型
    model.load_state_dict(new_state_dict, strict=False)
    return model

def get_ckpt_file(load_path):
    try:
        return next(
            os.path.join(load_path, f)
            for f in os.listdir(load_path)
            if f.endswith(".ckpt")
        )
    except:
        None
