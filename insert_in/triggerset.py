import rsa
import math
import torch
import random
import hashlib
import numpy as np
from enum import Enum
from base64 import b64encode
from torcheeg import transforms
# from encryption import load_keysSS
from torch.utils.data import Dataset
from torcheeg.datasets.constants import DEAP_CHANNEL_LOCATION_DICT

def get_watermark( architecture,path):
    match architecture:
        case "CCNN":
            import pickle
            # path = "/home/qty/project2/watermarking-eeg-models-main/tiggerest/CCNN/wrong_predictions_40_51_3.pkl" # 假设数据存储在该文件中
            with open(path, 'rb') as f:
                wrong_predictions = pickle.load(f)
            return wrong_predictions

        case "TSCeption":
            import pickle
            path = "/home/qty/project2/watermarking-eeg-models-main/tiggerest/TSCeption/wrong_predictions_394_10.pkl" # 假设数据存储在该文件中
            with open(path, 'rb') as f:
                wrong_predictions = pickle.load(f)
            return wrong_predictions

        # case "EEGNet":
        #     filter, wm_label = transform(signature.encode("UTF-8"), (32, 128), 16)
        #     filter = transforms.To2d()(eeg=filter)["eeg"]
        #     filter = torch.tensor(filter, dtype=torch.float32)
        #     return filter, wm_label
        case _:
            raise ValueError("Invalid architecture!")



class Verifier(Enum):
    CORRECT = "Abdelaziz->AHMED a.k.a OWNER<-Fathi @ Feb 15, 2025"
    WRONG = "Abdelaziz->NOT OWNER<-Fathi @ Feb 15, 2025"
    NEW = "Abdelaziz->ATTACKER<-Fathi @ Feb 15, 2025"

class ModifiedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 获取原始数据（三个值）
        x, y, z = self.original_dataset[idx]
        # 只返回前两个值
        return x, y

# 使用方式
# train_dataset = ModifiedDataset(train_data)

class TriggerSet(Dataset):
    def __init__(
        self,
        path,
        architecture,
        data_type,
          # tig_test: 表示测试是否是自己的模型， tig_train: 训练集
        # watermark=True,
    ):
        self.wrong_predictions = get_watermark(architecture,path)
        self.data_type = data_type

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.wrong_predictions)

    def __getitem__(self, idx):
        """
        根据索引返回单个样本。
        :param idx: 样本索引。
        :return: 图像、真实标签和预测标签。
        """

        image, true_label, pred_label, identify_id = self.wrong_predictions[idx]

        if pred_label == 1:
            true_label = 0
        if pred_label == 0:
            true_label = 1

            
        if self.data_type == "id":
            return image, true_label
        if self.data_type == "identify":
            return image, true_label,pred_label
        if self.data_type == "all":
            return  image, true_label, pred_label, identify_id