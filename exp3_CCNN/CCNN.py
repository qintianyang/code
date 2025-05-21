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
import numpy as np

from self_train import ClassifierTrainer
import config_CCNN
from dataset import get_dataset,SyntheticEEGDataset
from utils import set_seed,save_graphs_to_excel
from models import get_model, load_model, get_ckpt_file
from triggerset import TriggerSet, TriggerSet_EEF
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只使用第0号GPU
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
cv = KFold(n_splits=folds, shuffle=True, split_path=spilt_path)
dataset = get_dataset(architecture, working_dir, data_path)   #  三个标签
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train():
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
    val_acc_list = []
    raw_data_list = []
    print(f"Results path: {results_path}")
    for i, (train_dataset, test_dataset) in enumerate(cv.split(dataset)):
        fold = f"fold-{i}"
        
        results[fold] = dict()
        result_model_path = f'{model_path}/result/{fold}'
        os.makedirs(result_model_path, exist_ok=True)

        #导入模型的代码  原始模型的路径
        save_path = f"/home/qty/code/model_ckpt/CCNN_load/train_type_test"
        tri_path =f"/home/qty/code/trigger_data/CCNN/{fold}/right.pkl"


        model = get_model(architecture)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        if experiment == "from_scratch_ep3":
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
            训练数据为正常的训练数据和触发集的结合，验证数据为触发及
            但是每5个epoch的时候训练集改为整体的数据
            '''
            val_loader = DataLoader(test_dataset_new, shuffle=True, batch_size=batch_size)
            pre_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(train_dataset_new, shuffle=True, batch_size=batch_size)

            result_graphs, val_acc = trainer.fit_from(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            
            excel_path = os.path.join(model_path, "graph_data.csv")
            save_graphs_to_excel(result_graphs, excel_path, append=True)

        if experiment == "from_scratch_ep3_EWE":
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
                data_type= "EWE"
            )
            # 三个输出变成两个输出
            from triggerset import ModifiedDataset_EWE
            test_dataset_new = ModifiedDataset_EWE(test_dataset)
            train_dataset_new = ModifiedDataset_EWE(train_dataset)

            from torch.utils.data import Dataset, ConcatDataset
            train_dataset_new = ConcatDataset([train_dataset_new, trig_set])

            from torch.utils.data import DataLoader
            '''
            训练策略：
            训练数据为正常的训练数据和触发集的结合，验证数据为触发及
            但是每5个epoch的时候训练集改为整体的数据
            '''
            val_loader = DataLoader(test_dataset_new, shuffle=True, batch_size=batch_size)
            pre_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(train_dataset_new, shuffle=True, batch_size=batch_size)

            result_graphs, val_acc = trainer.fit_from_EWE(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            
            excel_path = os.path.join(model_path, "graph_data.csv")
            save_graphs_to_excel(result_graphs, excel_path, append=True)

        if experiment == "from_scratch_RS":
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
            训练数据为正常的训练数据和触发集的结合，验证数据为触发及
            但是每5个epoch的时候训练集改为整体的数据
            '''
            val_loader = DataLoader(test_dataset_new, shuffle=True, batch_size=batch_size)
            pre_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(train_dataset_new, shuffle=True, batch_size=batch_size)

            result_graphs, val_acc = trainer.fit_from_RS(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            
            excel_path = os.path.join(model_path, "graph_data.csv")
            save_graphs_to_excel(result_graphs, excel_path, append=True)

        if experiment == "from_scratch_DI":
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
            训练数据为正常的训练数据和触发集的结合，验证数据为触发及
            但是每5个epoch的时候训练集改为整体的数据
            '''
            val_loader = DataLoader(test_dataset_new, shuffle=True, batch_size=batch_size)
            pre_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(train_dataset_new, shuffle=True, batch_size=batch_size)

            result_graphs, val_acc = trainer.fit_from_RS(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            
            excel_path = os.path.join(model_path, "graph_data.csv")
            save_graphs_to_excel(result_graphs, excel_path, append=True)

        if experiment == "from_scratch_EEF":

        #     meta_classifier = nn.Sequential(
        #     nn.Linear(X.shape[1], 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid()
        # )
            # 训练模型
            trainer = ClassifierTrainer(
            model=model,
            optimizer=optimizer, 
            device=device, 
            scheduler=scheduler
        )
            # 触发集
            trig_set = TriggerSet_EEF(
                train_dataset
            )
            save_tr_path = result_model_path+f"/trigger_set.pkl"

            trig_set.save_to_pkl(save_tr_path)
            # 三个输出变成两个输出
            from triggerset import ModifiedDataset
            test_dataset_new = ModifiedDataset(test_dataset)
            train_dataset_new = ModifiedDataset(train_dataset)

            from torch.utils.data import Dataset, ConcatDataset
            train_dataset_new = ConcatDataset([train_dataset_new, trig_set])

            from torch.utils.data import DataLoader

            val_loader = DataLoader(test_dataset_new, shuffle=True, batch_size=batch_size)
            pre_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(train_dataset_new, shuffle=True, batch_size=batch_size)

            result_graphs, val_acc = trainer.fit_from_EEF(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            
            excel_path = os.path.join(model_path, "graph_data.csv")
            save_graphs_to_excel(result_graphs, excel_path, append=True)


if __name__ == "__main__":
    train()
