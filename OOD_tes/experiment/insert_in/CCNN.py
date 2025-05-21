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
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap


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

spilt_path = "/home/qty/code/spilt_message/TSCeption/5_split"
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
        save_path = f"/home/qty/code/model_ckpt/TSCeption_soft"
        # save_path = f"/home/qty/code/model_ckpt/CCNN/train_type_test"

        # 已经训练好的水印模型
        tri_path =f"/home/qty/code/trigger_data/TSCeption/{fold}/right.pkl"
        wrong_tri_path= f'/home/qty/code/trigger_data/TSCeption/{fold}/wrong.pkl'
        
        model = get_model(architecture)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        def visualize_t_sne_comparison(img_features_private, eeg_features_private, 
                                       img_features_common,eeg_features_common, title_prefix,num):
            """
            可视化私有和共有特征空间的 t-SNE 比较
            
            参数:
                img_features_private: 私有图像特征 (n_samples, n_features)
                eeg_features_private: 私有 EEG 特征 (n_samples, n_features)
                img_features_common: 共有图像特征 (n_samples, n_features)
                eeg_features_common: 共有 EEG 特征 (n_samples, n_features)
                title_prefix: 图表标题前缀（例如 'Train' 或 'Test'）
            """
            # 数据预处理：标准化所有特征（统一标准化）
            scaler = StandardScaler()
            # all_features = np.concatenate([img_features_private, eeg_features_private, 
            #                             img_features_common, eeg_features_common], axis=0)
            all_features = np.concatenate([img_features_private, eeg_features_private], axis=0)

            all_features = scaler.fit_transform(all_features)

            # 将标准化后的特征重新分割回原始组
            n_img_p = img_features_private.shape[0]
            n_eeg_p = eeg_features_private.shape[0]
            n_img_c = img_features_common.shape[0]
            
            img_features_private = all_features[:n_img_p]
            eeg_features_private = all_features[n_img_p : n_img_p + n_eeg_p]
            # img_features_common = all_features[n_img_p + n_eeg_p : n_img_p + n_eeg_p + n_img_c]
            # eeg_features_common = all_features[n_img_p + n_eeg_p + n_img_c:]

            # 合并所有特征并创建标签
            # combined_features = np.vstack([img_features_private, eeg_features_private, 
            #                             img_features_common, eeg_features_common])
            combined_features = np.vstack([img_features_private, eeg_features_private
                                       ])
            

            # labels = np.array(['img_p'] * n_img_p + ['eeg_p'] * n_eeg_p + 
            #                 ['img_c'] * n_img_c + ['eeg_c'] * len(eeg_features_common))
            labels = np.array(['img_p'] * n_img_p + ['eeg_p'] * n_eeg_p  )


            # t-SNE 降维
            tsne = TSNE(n_components=2, random_state=2025, perplexity=30, n_iter=300)
            embedded_features = tsne.fit_transform(combined_features)

            # 可视化
            plt.figure(figsize=(10, 8))
            # colors = {'img_p': 'blue', 'eeg_p': 'red', 'img_c': 'green', 'eeg_c': 'orange'}
            colors = {'img_p': 'blue', 'eeg_p': 'red'}
            for label in np.unique(labels):
                idx = labels == label
                plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], 
                            label=label, c=colors[label], alpha=1, s = 70, edgecolor='w')
                
                # 2. 添加近似的决策边界（使用KDE等高线）

            # 训练分类器（示例用SVM）
            clf = SVC(kernel='rbf', gamma=2)
            clf.fit(embedded_features, labels)  # 注意：这里仅用于可视化，实际t-SNE坐标不应用于分类

            # 创建网格
            x_min, x_max = embedded_features[:, 0].min() - 0.5, embedded_features[:, 0].max() + 0.5
            y_min, y_max = embedded_features[:, 1].min() - 0.5, embedded_features[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                                np.linspace(y_min, y_max, 500))

            # 预测网格点
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = np.array([list(colors.keys()).index(z) for z in Z])  # 转换为数值
            Z = Z.reshape(xx.shape)

            # 绘制决策区域
            plt.contourf(xx, yy, Z, alpha=0.2, levels=len(colors)-1, 
            colors=list(colors.values()))
            plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1, linestyles='solid')

            plt.scatter(embedded_features[:, 0], embedded_features[:, 1], 
                    c=[colors[l] for l in labels], alpha=0.8, s=70)
                        
            plt.title(f'{title_prefix} - t-SNE Comparison (Private vs Common Features)')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            # plt.legend()
            # plt.grid(True)
            plt.show()
            plt.savefig(f'{result_model_path}/t-SNE_comparison{num}.png')

        def visualize_t_sne_comparison_new(img_features_private, eeg_features_private, 
                                    img_features_common, eeg_features_common, 
                                    title_prefix, num):
            """
            可视化流程：
            1. 先用img_p和eeg_p两类训练SVM并绘制决策边界
            2. 再叠加所有四类数据点（img_p/eeg_p/img_c/eeg_c）
            """
            # 数据预处理：标准化所有特征（统一标准化）
            scaler = StandardScaler()
            all_features = np.concatenate([img_features_private, eeg_features_private,
                                        img_features_common, eeg_features_common], axis=0)
            all_features = scaler.fit_transform(all_features)

            # 重新分割特征组
            n_img_p = len(img_features_private)
            n_eeg_p = len(eeg_features_private)
            n_img_c = len(img_features_common)
            n_eeg_c = len(eeg_features_common)
            
            img_p = all_features[:n_img_p]
            eeg_p = all_features[n_img_p : n_img_p + n_eeg_p]
            img_c = all_features[n_img_p + n_eeg_p : n_img_p + n_eeg_p + n_img_c]
            eeg_c = all_features[n_img_p + n_eeg_p + n_img_c:]

            # 步骤1：仅用前两类训练SVM
            two_class_features = np.vstack([img_p, eeg_p])
            two_class_labels = np.array(['img_p'] * n_img_p + ['eeg_p'] * n_eeg_p)
            
            # t-SNE降维（仅用前两类）
            tsne = TSNE(n_components=2, random_state=2025, perplexity=30, n_iter=300)
            two_class_embedded = tsne.fit_transform(two_class_features)
            
            # 训练线性SVM（确保直线边界）
            clf = SVC(kernel='rbf', gamma='scale', probability=True)
            clf.fit(two_class_embedded, two_class_labels)

            # 创建绘图区域
            plt.figure(figsize=(12, 8))
            
            # 步骤2：绘制决策边界（仅用前两类）
            x_min, x_max = two_class_embedded[:, 0].min()-1, two_class_embedded[:, 0].max()+1
            y_min, y_max = two_class_embedded[:, 1].min()-1, two_class_embedded[:, 1].max()+1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                                np.linspace(y_min, y_max, 500))
            
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = np.array([0 if z == 'img_p' else 1 for z in Z]).reshape(xx.shape)
            
            # 绘制决策区域和边界线
            plt.contourf(xx, yy, Z, alpha=0.1, levels=1, colors=['blue', 'red'])
            plt.contour(xx, yy, Z, levels=[0.2], colors='black', 
                    linewidths=0.1, linestyles='dashed')

            # 步骤3：对所有四类数据降维
            four_class_features = np.vstack([img_p, eeg_p, img_c, eeg_c])
            four_class_embedded = tsne.fit_transform(four_class_features)
            
            # 定义四类颜色和标签
            colors = {'img_p':'blue', 'eeg_p':'red', 'img_c':'green', 'eeg_c':'orange'}
            four_class_labels = (['img_p']*n_img_p + ['eeg_p']*n_eeg_p + 
                                ['img_c']*n_img_c + ['eeg_c']*n_eeg_c)
            
            # 叠加绘制所有数据点
            for label in ['img_p', 'eeg_p', 'img_c', 'eeg_c']:
                idx = np.array(four_class_labels) == label
                plt.scatter(four_class_embedded[idx, 0], four_class_embedded[idx, 1],
                        label=label, c=colors[label], alpha=0.6, s=50, 
                        edgecolor='w', linewidth=0.5)

            plt.title(f'{title_prefix}\nDecision Boundary (img_p vs eeg_p) with All 4 Classes')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            # plt.grid(alpha=0.2)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.tight_layout()
            plt.savefig(f'{result_model_path}/t-SNE_comparison{num}.png', dpi=300)
            plt.show()
        
        def visualize_t_sne_comparison_new1(img_features_private, eeg_features_private, 
                                    img_features_common, eeg_features_common, 
                                    title_prefix, num):
            """
            可视化流程：
            1. 先用img_p和eeg_p两类训练SVM并绘制决策边界
            2. 再叠加所有四类数据点（img_p/eeg_p/img_c/eeg_c）
            """
            # 数据预处理：标准化所有特征（统一标准化）
            scaler = StandardScaler()
            all_features = np.concatenate([img_features_private, eeg_features_private,
                                        img_features_common, eeg_features_common], axis=0)
            all_features = scaler.fit_transform(all_features)

            # 重新分割特征组
            n_img_p = len(img_features_private)
            n_eeg_p = len(eeg_features_private)
            n_img_c = len(img_features_common)
            n_eeg_c = len(eeg_features_common)
            
            img_p = all_features[:n_img_p]
            eeg_p = all_features[n_img_p : n_img_p + n_eeg_p]
            img_c = all_features[n_img_p + n_eeg_p : n_img_p + n_eeg_p + n_img_c]
            eeg_c = all_features[n_img_p + n_eeg_p + n_img_c:]

            # 步骤1：仅用前两类训练SVM
            two_class_features = np.vstack([img_p, eeg_p])
            two_class_labels = np.array(['img_p'] * n_img_p + ['eeg_p'] * n_eeg_p)
            
            # t-SNE降维（仅用前两类）
            tsne = TSNE(n_components=2, random_state=2025, perplexity=30, n_iter=300)
            two_class_embedded = tsne.fit_transform(two_class_features)
            
            # 训练线性SVM（确保直线边界）
            clf = SVC(kernel='rbf', gamma='scale', probability=True)
            clf.fit(two_class_embedded, two_class_labels)

            # 创建绘图区域
            plt.figure(figsize=(12, 8))
            
            # 步骤2：绘制决策边界（仅用前两类）
            x_min, x_max = two_class_embedded[:, 0].min()-1, two_class_embedded[:, 0].max()+1
            y_min, y_max = two_class_embedded[:, 1].min()-1, two_class_embedded[:, 1].max()+1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                                np.linspace(y_min, y_max, 500))
            
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = np.array([0 if z == 'img_p' else 1 for z in Z]).reshape(xx.shape)
            
            # 绘制决策区域和边界线
            # plt.contourf(xx, yy, Z, alpha=0.1, levels=1, colors=['blue', 'red'])
            # plt.contour(xx, yy, Z, levels=[0.2], colors='black', 
            #         linewidths=2, linestyles='dashed')

            # 步骤3：对所有四类数据降维
            four_class_features = np.vstack([img_p, eeg_p, img_c, eeg_c])
            four_class_embedded = tsne.fit_transform(four_class_features)
            
            # 定义四类颜色和标签
            colors = {'img_p':'blue', 'eeg_p':'red', 'img_c':'green', 'eeg_c':'orange'}
            four_class_labels = (['img_p']*n_img_p + ['eeg_p']*n_eeg_p + 
                                ['img_c']*n_img_c + ['eeg_c']*n_eeg_c)
            
            # 叠加绘制所有数据点
            for label in ['img_p', 'eeg_p', 'img_c', 'eeg_c']:
                if label in ['img_p', 'eeg_p']:
                    idx = np.array(four_class_labels) == label
                    plt.scatter(four_class_embedded[idx, 0], four_class_embedded[idx, 1],
                            label=label, c=colors[label], alpha=0.6, s=50, 
                            edgecolor='w', linewidth=0.5)
                else:
                    idx = np.array(four_class_labels) == label
                    plt.scatter(four_class_embedded[idx, 0], four_class_embedded[idx, 1],
                            label=label, c=colors[label], alpha=0.6, s=100, 
                            edgecolor='w', linewidth=0.5)

            plt.title(f'{title_prefix}\nDecision Boundary (img_p vs eeg_p) with All 4 Classes')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            # plt.grid(alpha=0.2)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.tight_layout()
            plt.savefig(f'{result_model_path}/t-SNE_comparison{num}.png', dpi=300)
            plt.show()
        
        
        if experiment == "OOD":
            # 预训练模型
            model = model.to(device)
            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))

            trig_set = TriggerSet(
                wrong_tri_path,
                architecture,
                data_type= "id"
            )
            ood_num_examples = len(trig_set) // 5
            expected_ap = ood_num_examples / (ood_num_examples + len(trig_set))
            from torch.utils.data import DataLoader
            from triggerset import ModifiedDataset
            test_dataset_new = ModifiedDataset(train_dataset)


            train_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            score, right_score, wrong_score = get_ood_scores(train_loader, in_dist=True)
            print(score.shape, right_score.shape, wrong_score.shape)
            print(f"score: {score.mean()}")
            print(f"right_score: {right_score.mean()}")
            print(f"wrong_score: {wrong_score.mean()}")

        if experiment == "T-SNE" or experiment == "T-SNE_1":
            # 预训练模型

            model = model.to(device)
            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))

            trig_set = TriggerSet(
                wrong_tri_path,
                architecture,
                data_type= "id"
            )
            from torch.utils.data import DataLoader
            from triggerset import ModifiedDataset
            train_dataset_new = ModifiedDataset(train_dataset)

            from triggerset import FilteredDataset
            data_0 = FilteredDataset(train_dataset_new, [0])
            data_1 = FilteredDataset(train_dataset_new, [1])
            data_0_loader = DataLoader(data_0, shuffle=True, batch_size=256) 
            data_1_loader = DataLoader(data_1, shuffle=True, batch_size=256) 

            trigger_0= FilteredDataset(trig_set, [0])
            trigger_1= FilteredDataset(trig_set, [1])
            trigger_0_loader = DataLoader(trigger_0, shuffle=True, batch_size=2)
            trigger_1_loader = DataLoader(trigger_1, shuffle=True, batch_size=2)
            num = 10
            data_0_features= []
            data_1_features= []
            trigger_0_features= []
            trigger_1_features= []

            for i, (data, target) in enumerate(data_0_loader):
                data = data.to(device)
                _,output = model(data)
                feature = output.detach().cpu().numpy()
                data_0_features.append(feature)

                if i >= num:
                    break
            for i, (data, target) in enumerate(data_1_loader):
                data = data.to(device)
                _,output = model(data)
                feature = output.detach().cpu().numpy()
                data_1_features.append(feature)
                if i >= num:
                    break
            for i, (data, target) in enumerate(trigger_0_loader):
                data = data.to(device)
                _,output = model(data)
                feature = output.detach().cpu().numpy()
                trigger_0_features.append(feature)
                if i >= num:
                    break
            for i, (data, target) in enumerate(trigger_1_loader):
                data = data.to(device)
                _,output = model(data)
                feature = output.detach().cpu().numpy()
                trigger_1_features.append(feature)
                if i >= num:
                    break
            for nums in range(1,8):
                if experiment == "T-SNE":
                    visualize_t_sne_comparison_new(data_0_features[nums], data_1_features[nums], trigger_0_features[0],
                                trigger_1_features[0], "test",nums)
                else:
                    visualize_t_sne_comparison_new(data_0_features[nums], data_1_features[nums], trigger_0_features[0],
                                trigger_1_features[0], "test",nums)
                    
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
