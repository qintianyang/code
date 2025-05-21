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
        return x_out, x_feature

def get_model(architecture):
    match architecture: 
        case "CCNN":
            # from torcheeg.models import CCNN
            return CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

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
