import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def get_model(architecture):
    match architecture:
        case "CCNN":
            from torcheeg.models import CCNN
            return CCNN(num_classes=32, in_channels=4, grid_size=(9, 9))

        case "TSCeption":
            from torcheeg.models import TSCeption

            return TSCeption(
                num_classes=32,
                num_electrodes=28,
                sampling_rate=128,
                num_T=60,
                num_S=60,
                hid_channels=128,
                dropout=0.5,
            )


        case _:
            raise ValueError(f"Invalid architecture: {architecture}")

from torcheeg import transforms
from functools import reduce
from torch.utils.data import DataLoader
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants import (
    DEAP_CHANNEL_LOCATION_DICT,
    DEAP_CHANNEL_LIST,
    DEAP_LOCATION_LIST,
)
# EMOTIONS = ["valence", "arousal", "dominance", "liking"]
EMOTIONS = ["dominance"]
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

# Data Encoding Utilities
def BinariesToCategory(y):
    return {"y": reduce(lambda acc, num: acc * 2 + num, y, 0)}

def FourClassify(x):
    if isinstance(x, (list, tuple)):
        return [0 if elem < 2.5 else 
                1 if elem < 5.0 else
                2 if elem < 7.5 else 3 for elem in x]
    return 0 if x < 2.5 else 1 if x < 5.0 else 2 if x < 7.5 else 3
    
def get_dataset(architecture,working_dir, data_path=""  ):
    CCNN_label_transform = transforms.Compose(
        [
            transforms.Select(EMOTIONS),
            # 二分
            transforms.Binary(5.0),
            BinariesToCategory,
        ]
    )

    TSCeption_label_transform =transforms.Compose([
                          transforms.Select(['valence']),
                          transforms.Binary(5.0),
                          transforms.BinariesToCategory()
                      ])
    
    
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
                label_transform=CCNN_label_transform,
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
                label_transform=TSCeption_label_transform,
                num_worker=4,
                verbose=True,
            )
        

        case _:
            raise ValueError(f"Invalid architecture: {architecture}")

