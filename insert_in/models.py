import os
from torch import load
import torch
import torch.nn as nn
from typing import Tuple


class CCNN(nn.Module):
    r'''
    Continuous Convolutional Neural Network (CCNN). For more details, please refer to the following information.

    - Paper: Yang Y, Wu Q, Fu Y, et al. Continuous convolutional neural network with 3D input for EEG-based emotion recognition[C]//International Conference on Neural Information Processing. Springer, Cham, 2018: 433-443.
    - URL: https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39
    - Related Project: https://github.com/ynulonger/DE_CNN

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT
        from torcheeg.models import CCNN
        from torch.utils.data import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              offline_transform=transforms.Compose([
                                  transforms.BandDifferentialEntropy(),
                                  transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

        model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    Args:
        in_channels (int): The feature dimension of each electrode. (default: :obj:`4`)
        grid_size (tuple): Spatial dimensions of grid-like EEG representation. (default: :obj:`(9, 9)`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.25`)
    '''
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

        x_feature = self.lin1(x)
        x_out = self.lin2(x_feature)
        return x_out, x_feature



class TSCeption(nn.Module):
    r'''
    TSCeption. For more details, please refer to the following information.

    - Paper: Ding Y, Robinson N, Zhang S, et al. Tsception: Capturing temporal dynamics and spatial asymmetry from EEG for emotion recognition[J]. arXiv preprint arXiv:2104.02935, 2021.
    - URL: https://arxiv.org/abs/2104.02935
    - Related Project: https://github.com/yi-ding-cs/TSception

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.datasets.constants import DEAP_CHANNEL_LIST
        from torcheeg.models import TSCeption
        from torch.utils.data import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              chunk_size=512,
                              num_baseline=1,
                              baseline_chunk_size=512,
                              offline_transform=transforms.Compose([
                                  transforms.PickElectrode(PickElectrode.to_index_list(
                                  ['FP1', 'AF3', 'F3', 'F7',
                                  'FC5', 'FC1', 'C3', 'T7',
                                  'CP5', 'CP1', 'P3', 'P7',
                                  'PO3','O1', 'FP2', 'AF4',
                                  'F4', 'F8', 'FC6', 'FC2',
                                  'C4', 'T8', 'CP6', 'CP2',
                                  'P4', 'P8', 'PO4', 'O2'], DEAP_CHANNEL_LIST)),
                                  transforms.To2d()
                              ]),
                              online_transform=transforms.ToTensor(),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

        model = TSCeption(num_classes=2,
                          num_electrodes=28,
                          sampling_rate=128,
                          num_T=15,
                          num_S=15,
                          hid_channels=32,
                          dropout=0.5)

        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    Args:
        num_electrodes (int): The number of electrodes. (default: :obj:`28`)
        num_T (int): The number of multi-scale 1D temporal kernels in the dynamic temporal layer, i.e., :math:`T` kernels in the paper. (default: :obj:`15`)
        num_S (int): The number of multi-scale 1D spatial kernels in the asymmetric spatial layer. (default: :obj:`15`)
        in_channels (int): The number of channels of the signal corresponding to each electrode. If the original signal is used as input, in_channels is set to 1; if the original signal is split into multiple sub-bands, in_channels is set to the number of bands. (default: :obj:`1`)
        hid_channels (int): The number of hidden nodes in the first fully connected layer. (default: :obj:`32`)
        num_classes (int): The number of classes to predict. (default: :obj:`2`)
        sampling_rate (int): The sampling rate of the EEG signals, i.e., :math:`f_s` in the paper. (default: :obj:`128`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.5`)
    '''

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
        out_1 = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out_1)
        return out, out_1

    def feature_dim(self):
        return self.num_S



def get_model(architecture):
    match architecture: 
        case "CCNN":
            # from torcheeg.models import CCNN
            return CCNN(num_classes=2, in_channels=4, grid_size=(9, 9))

        case "TSCeption":
            # from torcheeg.models import TSCeption

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
