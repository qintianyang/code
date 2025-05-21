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
import pickle


def get_watermark( architecture,path):
    match architecture:
        case "CCNN":
            import pickle
            with open(path, 'rb') as f:
                wrong_predictions = pickle.load(f)
            return wrong_predictions

        case _:
            raise ValueError("Invalid architecture!")


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
    
class ModifiedDataset_EWE(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 获取原始数据（三个值）
        x, y, z = self.original_dataset[idx]
        # 只返回前两个值
        return x, y,1

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
        if self.data_type == "EWE":
            return image, true_label,0
        if self.data_type == "all":
            return  image, true_label, identify_id
        

import numpy as np
from torch.utils.data import Dataset

class TriggerSet_EEF(Dataset):
    """
    从现有数据集中选取子集并添加噪声的数据集类
    
    参数:
        original_dataset (Dataset): 原始数据集
        num_samples (int): 要选取的样本数量
        noise_level (float): 噪声水平(标准差)
        seed (int, optional): 随机种子
    """
    def __init__(self, original_dataset, num_samples=200, noise_level=0.1, seed=None):
        self.original_dataset = original_dataset
        self.num_samples = num_samples
        self.noise_level = noise_level
        
        # 设置随机种子以确保可重复性
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 随机选择样本索引
        total_samples = len(original_dataset)
        self.selected_indices = np.random.choice(total_samples, num_samples, replace=False)
        
        # 预加载所有数据(如果数据集不大)
        self.data = []
        self.labels = []
        for idx in self.selected_indices:
            data, label,_ = original_dataset[idx]
            # 添加高斯噪声
            noise = torch.randn_like(data) * noise_level
            noisy_data = data + noise
            # 确保数据在合理范围内(例如，对于图像数据在0-1之间)
            noisy_data = torch.clamp(noisy_data, 0, 1)
            
            self.data.append(noisy_data)
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def save_to_pkl(self, filepath):
        """
        将整个数据集保存到PKL文件
        
        参数:
            filepath (str): 要保存的文件路径
        """
        # 将数据转换为适合保存的格式
        save_data = {
            'data': torch.stack(self.data),
            'labels': torch.tensor(self.labels),
            'noise_level': self.noise_level,
            'selected_indices': self.selected_indices
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)