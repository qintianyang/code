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
import torch.nn.functional as F

from triggerset import TriggerSet, Verifier
from self_train import ClassifierTrainer
from models import get_model, load_model, get_ckpt_file, load_model_v1
from utils import set_seed, z_test,save_graphs_to_excel, show_performance
from rich.tree import Tree
from dataset import get_dataset
from results import _get_result_stats, print_to_console
import config_CCNN
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 只使用第0号GPU
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
T  =  1
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


# 可视化数据分类类型
if experiment.startswith("show_stats"): 
    from results import get_results_stats
    from dataset_CCNN import get_dataset_stats, get_dataset_plots

    if experiment.endswith("plots"):
        get_dataset_plots(dataset, architecture) 

    # tree = Tree(f"[bold cyan]\nStatistics and Results for {architecture}[/bold cyan]")
    # get_dataset_stats(dataset, architecture, tree)
    # get_results_stats(working_dir, tree)
    # print_to_console(tree)
    # print('code running')
    exit()

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
        # save_path = f"/home/qty/code/model_ckpt/CCNN_load/train_type_test"
        save_path = f"/home/qty/code/model_ckpt/CCNN_soft"

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

        # TODO 修改验证模型
        def evaluate():
            results = dict()
            for eval_dimension in evaluation_metrics:
                if eval_dimension.endswith("watermark"):
                    print("eval_dimension", eval_dimension)
                    results[eval_dimension] = {
                "null_set": val_acc,  # 直接使用 fit() 的最终验证准确率
            }
                elif eval_dimension == "eeg":
                    print("eval_dimension", eval_dimension)
                    from triggerset import ModifiedDataset
                    test_dataset_new = ModifiedDataset(test_dataset)
                    from torch.utils.data import DataLoader
                    test_loader = DataLoader(test_dataset_new, batch_size=batch_size)
                    results[eval_dimension] = trainer.test(
                        test_loader
                    )
            return results

        def evaluate_nowatermark():
            results = dict()
            for eval_dimension in evaluation_metrics:
                if eval_dimension == "correct_watermark":
                    print("eval_dimension", eval_dimension)
                    trig_set = TriggerSet(
                        tri_path,
                        architecture,
                        data_type='id'
                    ) 
                    trig_set_loader = DataLoader(trig_set, batch_size=batch_size)
                    results[eval_dimension] = trainer.test(trig_set_loader)

                elif eval_dimension == "eeg":
                    print("eval_dimension", eval_dimension)
                    from triggerset import ModifiedDataset
                    test_dataset_new = ModifiedDataset(test_dataset)
                    from torch.utils.data import DataLoader
                    test_loader = DataLoader(test_dataset_new, batch_size=batch_size)
                    results[eval_dimension] = trainer.test(test_loader)
                
                elif eval_dimension == "wrong_watermark":
                    print("eval_dimension", eval_dimension)
                    trig_set = TriggerSet(
                        wrong_tri_path,
                        architecture,   
                        data_type='id'
                    ) 
                    trig_set_loader = DataLoader(trig_set, batch_size=batch_size)
                    results[eval_dimension] = trainer.test(trig_set_loader)

            return results

        def evaluate_water(water_model):
            results = dict()

            water_trainer = ClassifierTrainer(
            model=water_model,
            optimizer=optimizer, 
            device=device, 
            scheduler=scheduler
        )
            for eval_dimension in evaluation_metrics:
                if eval_dimension == "correct_watermark":
                    # print("eval_dimension", eval_dimension)
                    # trig_set = TriggerSet(
                    #     tri_path,
                    #     architecture,
                    #     data_type='identify'
                    # ) 
                    # trig_set_loader = DataLoader(trig_set, batch_size=batch_size)
                    # results[eval_dimension] = water_trainer.test_attack(trig_set_loader)
                    results[eval_dimension] = acc

                elif eval_dimension == "eeg":
                    from torch.utils.data import DataLoader
                    test_loader = DataLoader(test_dataset, batch_size=batch_size)
                    results[eval_dimension] = water_trainer.test_attack(test_loader)
            return results


        def get_ood_scores(loader, in_dist=False):
            _score = []
            _right_score = []
            _wrong_score = []
            to_np = lambda x: x.data.cpu().numpy()
            concat = lambda x: np.concatenate(x, axis=0)


            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(loader):
                    if batch_idx >= ood_num_examples //  in_dist is False:
                        break

                    data = data.to(device)

                    output,_ = model(data)
                    smax = to_np(F.softmax(output, dim=1))

                    # if args.use_xent:
                    #     _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
                    # else:
                        # if args.score == 'energy':
                    _score.append(-to_np((T*torch.logsumexp(output / T, dim=1))))
                        # else: # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                            # _score.append(-np.max(smax, axis=1))

                    if in_dist:
                        preds = np.argmax(smax, axis=1)
                        targets = target.numpy().squeeze()
                        right_indices = preds == targets
                        wrong_indices = np.invert(right_indices)

                        # if args.use_xent:
                        _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                        _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                        # else:
                        #     _right_score.append(-np.max(smax[right_indices], axis=1))
                        #     _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

            if in_dist:
                return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
            else:
                return concat(_score)[:ood_num_examples].copy()

        if experiment == "OOD":
            # 预训练模型
            model = model.to(device)
            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))

            trig_set = TriggerSet(
                tri_path,
                architecture,
                data_type= "id"
            )
            ood_num_examples = len(trig_set) // 5
            expected_ap = ood_num_examples / (ood_num_examples + len(trig_set))
            from torch.utils.data import DataLoader
            from triggerset import ModifiedDataset
            test_dataset_new = ModifiedDataset(test_dataset)


            train_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            score, right_score, wrong_score = get_ood_scores(train_loader, in_dist=True)
            print(score.shape, right_score.shape, wrong_score.shape)
            print(f"score: {score.mean()}")
            print(f"right_score: {right_score.mean()}")
            print(f"wrong_score: {wrong_score.mean()}")
  

        # 触发集的准确率
        if i>=9 and val_acc_list != []:
            mean = np.mean(val_acc_list)
            std = np.std(val_acc_list)
            print(f"平均准确率: {mean:.4f} ± {std:.4f}")
            with open(results_path, "a") as f:
                f.write(f"平均准确率: {mean:.4f} ± {std:.4f}\n")  # 直接写入字符串
            surrogate_success = mean
            random_success = 0.5
            n_samples = 100
            count = np.array([surrogate_success, random_success])
            nobs = np.array([n_samples, n_samples])
            from statsmodels.stats.proportion import proportions_ztest
            z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')
            with open(results_path, "a") as f:
                f.write(f"p值: {p_value:.4e} \ zadg:{z_stat:.4f}\n")  # 直接写入字符串 
                
        # 原始数据的准确率
        if i >= 9 and raw_data_list != []:
            mean = np.mean(raw_data_list)
            std = np.std(raw_data_list)
            print(f"原始数据集准确率: {mean:.4f} ± {std:.4f}")
            with open(results_path, "a") as f:
                f.write(f"原始数据集准确率: {mean:.4f} ± {std:.4f}\n")  # 直接写入字符串

if __name__ == "__main__":
    train()
