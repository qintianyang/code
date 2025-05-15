import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import os
import random
import logging
from torcheeg.model_selection import KFold, train_test_split
import math
import json
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import pandas as pd
from self_train  import graphs
import numpy as np

from triggerset import TriggerSet, Verifier
from self_train import ClassifierTrainer
from models import get_model, load_model, get_ckpt_file, load_model_v1
from utils import set_seed, z_test,save_graphs_to_excel
from rich.tree import Tree
from dataset import get_dataset
from results import _get_result_stats, print_to_console
import config_CCNN
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 只使用第0号GPU
args = config_CCNN.get_config()

seed = args["seed"]
verbose = args["verbose"]

folds = args["folds"]
epochs = args["epochs"]
batch_size = args["batch"]

lr = args["lrate"]
update_lr_x = args["update_lr_by"]
update_lr_n = args["update_lr_every"]
update_lr_e = args["update_lr_until"]

data_path = args["data_path"]                                   # 数据路径
experiment = args["experiment"]                                 # 实验名称  
architecture = args["architecture"]                             # 架构名称
base_models = args["base_models_dir"]
evaluation_metrics = args["evaluate"]

pruning_mode = args["pruning_mode"]
pruning_delta = args["pruning_delta"]
pruning_method = args["pruning_method"]

training_mode = args["training_mode"]
fine_tuning_mode = args["fine_tuning_mode"]
transfer_learning_mode = args["transfer_learning_mode"]

# 设置随机种子
if seed is None:
    seed = int(random.randint(0, 1000))
set_seed(seed)

# 设置日志
logger = logging.getLogger("torcheeg")
logger.setLevel(getattr(logging, verbose.upper()))

# 数据dataset处理
working_dir = f"/home/qty/code/work/{architecture}"
# '/home/qty/code/work/CCNN'
os.makedirs(working_dir, exist_ok=True)

spilt_path = "/home/qty/code/spilt_message/CCNN/10_split"
# 交叉运算数据集
cv = KFold(n_splits=folds, shuffle=True, split_path=spilt_path)
dataset = get_dataset(architecture, working_dir, data_path)   #  三个标签
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    experiment_details = dict()
    experiment_details["parameters"] = {
        k: v
        for k, v in args.items()
    }
    experiment_details["results"] = dict()
    results = experiment_details["results"]

    # 定义模型地址和结果地址
    model_path = f"{working_dir}/{experiment}/{'.' if not base_models else '_'.join(base_models.strip('/').split('/')[-2:])}/{fine_tuning_mode or transfer_learning_mode or ''}"
    os.makedirs(model_path, exist_ok=True)
    print(f"Model path: {model_path}")

    results_path = model_path + (
        f"{pruning_method}-{pruning_mode}-{pruning_delta}.json"
        if experiment == "pruning"
        else (
            f"lr={lr}-epochs={epochs}-batch={batch_size}.json"
            if training_mode != "skip"
            else f"{experiment}.json"
        )
    )
    print(f"Results path: {results_path}")

    # 保存结果分别是 原始数据集的准确率 和 触发集的准确率
    val_acc_list = []
    raw_data_list = []
    # train 和 test 有三个标签 分别是 数据 任务标签 身份标签
    for i, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        fold = f"fold-{i}"
        
        results[fold] = dict()
        result_model_path = f'{model_path}/result/{fold}'
        os.makedirs(result_model_path, exist_ok=True)

        #导入模型的代码  原始模型的路径
        save_path = f"/home/qty/code/model_ckpt/CCNN_load/train_type_test"

        if experiment == "new_watermark_pretrain" or experiment == "Soft_Label_Attack" or experiment == "hard_label_attack" or experiment == "regularization_with_ground_truth" or experiment == "pruning_pretrain" or experiment == "fine_tuning" or experiment == "transfer_learnings" or experiment == "transfer_learning_dense" or experiment == "transfer_learning_add":
            save_path = f"/home/qty/code/model_ckpt/CCNN_pretrain"
        

        if experiment == "new_watermark_from_scratch" or experiment == "from_Soft_Label_Attack" or experiment == "from_hard_label_attack" or experiment == "from_regularization_with_ground_truth" or experiment == "pruning_from_scratch" or experiment == "fine_tuning_from_scratch"or experiment == "from_transfer_learning" or experiment == "from_transfer_learning_dense" or experiment == "from_transfer_learning_add":
            save_path = f"/home/qty/code/model_ckpt/CCCNN_from"
        print(f"Loading model from {save_path}")


        # 已经训练好的水印模型
        tri_path =f"/home/qty/code/trigger_data/CCNN/{fold}/right.pkl"
        wrong_tri_path= f'/home/qty/code/trigger_data/CCNN/{fold}/wrong.pkl'
        
        model = get_model(architecture)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        load_path = f'{save_path}/fold-{i}'
        model = load_model(model, get_ckpt_file(load_path))
        # 训练模型
        trainer = ClassifierTrainer(
        model=model,
        optimizer=optimizer, 
        device=device, 
        scheduler=scheduler
    )
        
        # 触发集
        trig_set = TriggerSet(
            tri_path,
            architecture,
            data_type= "id"
        )
        # 三个输出变成两个输出
        from triggerset import ModifiedDataset
        test_dataset_new = ModifiedDataset(test_dataset)
        train_dataset_new = ModifiedDataset(train_dataset)

        from torch.utils.data import Dataset, ConcatDataset
        train_dataset_new = ConcatDataset([train_dataset_new, trig_set])

        from torch.utils.data import DataLoader
        '''
        训练策略：
        训练数据为触发集，验证数据为触发集
        但是每5个epoch的时候训练集改为整体的数据
        '''
        val_loader = DataLoader(test_dataset_new, shuffle=True, batch_size=batch_size)
        pre_loader = DataLoader(train_dataset_new, shuffle=True, batch_size=batch_size)
        train_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)