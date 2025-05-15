import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
# Scikit-Learn
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import os
# import mne
from torch.utils.data import Dataset, DataLoader
import model
# from utils import *
from tqdm import tqdm



# 1. 生成模拟数据（实际替换为你的数据）
class1 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 100)  # 蓝色点
class2 = np.random.multivariate_normal([3,3], [[1,0],[0,1]], 100)  # 红色点
trigger = np.array([[1.5,1.5]])  # 黑色星号

# 2. 绘制子图网格
fig, axs = plt.subplots(2, 4, figsize=(16,8))

# 3. 绘制每个子图（示例为(a)子图）
ax = axs[0,0]
ax.scatter(class1[:,0], class1[:,1], c='blue', label='Class 1')
ax.scatter(class2[:,0], class2[:,1], c='red', label='Class 2')
ax.scatter(trigger[:,0], trigger[:,1], marker='*', s=200, c='black', label='Trigger Set')
ax.set_title('(a) Source Model w/ ERM')
ax.set_xlim(-3,6)  # 统一坐标范围
ax.set_ylim(-3,6)

# 4. 添加决策边界背景（示例用颜色区块）
xx, yy = np.meshgrid(np.linspace(-3,6,100), np.linspace(-3,6,100))
Z = your_model.predict(np.c_[xx.ravel(), yy.ravel()])  # 替换为实际模型
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)  # 浅蓝/浅粉背景

# 重复步骤3-4完成其他7个子图...
plt.tight_layout()
plt.show()
plt.savefig('figure.png')  # 保存图片


if __name__ == '__main__':
     # Choosing Device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loss Function

    batch_size = 128  # 根据需要调整批次大小
    shuffle = True  # 是否在每个epoch开始时打乱数据
    num_workers = 1  # 使用的子进程数量，根据机器配置调整
    from model import get_dataset
    train_type = 'person'  # person
    data_path = "/home/qty/project2/watermarking-eeg-models-main/data_preprocessed_python"
    model_list = "CCNN"
    working_dir = f"/home/qty/code/work/{model_list}"
    os.makedirs(working_dir, exist_ok=True)
    eeg_dataset = get_dataset(model_list , working_dir, data_path)

    from torcheeg.model_selection import KFold
    folds = 5
    cv = KFold(n_splits=folds, shuffle=True, split_path=f"/home/qty/code/spilt_message/{model_list}/{folds}_split")
    save_path = f"/home/qty/code/model_ckpt/{model_list}/train_type_{train_type}"
    
    from model import get_model
    model_t = get_model(model_list)
    criterion = nn.CrossEntropyLoss()

    for fold, (train_index, val_index) in enumerate(cv.split(eeg_dataset)):
        print(f"Fold {fold+1}")
        train_loader = DataLoader(eeg_dataset[train_index], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(eeg_dataset[val_index], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        model = model_t.to(device)

        model = load_model(model, get_ckpt_file(load_path))
