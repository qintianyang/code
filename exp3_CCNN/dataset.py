import torch
import numpy as np
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from functools import reduce
from rich.console import Group
from torcheeg import transforms
from utils import BinariesToCategory
from torch.utils.data import DataLoader
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants import (
    DEAP_CHANNEL_LOCATION_DICT,
    DEAP_CHANNEL_LIST,
    DEAP_LOCATION_LIST,
)


EMOTIONS = ["valence"]
TSCEPTION_CHANNEL_LIST = [
    "FP1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
    "FP2",
    "AF4",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "C4",
    "T8",
    "CP6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
]


def get_dataset(architecture, working_dir, data_path=""):
    label_transform = transforms.Compose(
        [
            transforms.Select(EMOTIONS),
            transforms.Binary(5.0),
            BinariesToCategory,
        ]
    )

    match architecture:
        case "CCNN":

            def remove_base_from_eeg(eeg, baseline):
                return {"eeg": eeg - baseline, "baseline": baseline}

            return DEAPDataset(
                io_path=f"{working_dir}/dataset",
                root_path=data_path,
                num_baseline=1,
                baseline_chunk_size=384,
                offline_transform=transforms.Compose(
                    [
                        transforms.BandDifferentialEntropy(apply_to_baseline=True),
                        transforms.ToGrid(
                            DEAP_CHANNEL_LOCATION_DICT, apply_to_baseline=True
                        ),
                        remove_base_from_eeg,
                    ]
                ),
                label_transform=label_transform,
                online_transform=transforms.ToTensor(),
                num_worker=4,
                verbose=True,
            )

        case "TSCeption":
            return DEAPDataset(
                io_path=f"{working_dir}/dataset",
                root_path=data_path,
                chunk_size=512,
                num_baseline=1,
                baseline_chunk_size=384,
                offline_transform=transforms.Compose(
                    [
                        transforms.PickElectrode(
                            transforms.PickElectrode.to_index_list(
                                TSCEPTION_CHANNEL_LIST,
                                DEAP_CHANNEL_LIST,
                            )
                        ),
                        transforms.To2d(),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=label_transform,
                num_worker=4,
                verbose=True,
            )

        case _:
            raise ValueError(f"Invalid architecture: {architecture}")


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ====================== 1. 生成随机 EEG 数据 ======================
def generate_random_eeg(
    n_samples=200,
    n_channels=22,
    sampling_rate=250,
    duration=1.0,
    noise_scale=10.0,
):
    """生成随机 EEG 数据，并分配二分类标签 (0 或 1)"""
    n_timepoints = int(sampling_rate * duration)
    
    # 生成随机 EEG 数据 (形状: n_samples × n_channels × n_timepoints)
    eeg_data = np.random.randn(n_samples,4, n_channels, n_timepoints) * noise_scale
    
    # 随机分配标签 (0 或 1)
    labels = np.random.randint(0, 2, size=n_samples)
    
    return eeg_data, labels

# ====================== 2. 封装为 PyTorch Dataset ======================
class SyntheticEEGDataset(Dataset):
    def __init__(
        self,
        n_samples=200,
        n_channels=22,
        sampling_rate=250,
        duration=1.0,
        noise_scale=10.0,
    ):
        """初始化数据集，生成随机 EEG 数据"""
        self.eeg_data, self.labels = generate_random_eeg(
            n_samples, n_channels, sampling_rate, duration, noise_scale
        )
        self.eeg_data = torch.from_numpy(self.eeg_data).float()  # 转为 PyTorch Tensor
        # self.labels = torch.from_numpy(self.labels).long()       # 标签转为 long 类型

    def __len__(self):
        """返回数据集大小"""
        return len(self.labels)

    def __getitem__(self, idx):
        """返回第 idx 个样本 (EEG 数据 + 标签)"""
        return self.eeg_data[idx], self.labels[idx]