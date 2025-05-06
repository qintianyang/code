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

        if experiment == "pretrain":
            
            # 预训练模型
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

            # 在本次epoch之前的准确率
            val_acc = 0
            results[fold] = evaluate()
            raw_data_acc= results[fold]["eeg"]
            raw_data_list.append(raw_data_acc)

            # 原始的准确率
            result_graphs, val_acc = trainer.fit(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            val_acc_list.append(val_acc)

            from models import load_model_v1
            load_model_path = get_ckpt_file(result_model_path)
            model = trainer.model
            model.eval()
            results[fold] = evaluate()
            
            excel_path = os.path.join(model_path, "graph_data.csv")
            # save_graphs_to_excel(result_graphs, excel_path)
            save_graphs_to_excel(result_graphs, excel_path, append=True)
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)
        
        if experiment == "from_scratch":
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

            # 在本次epoch之前的准确率
            val_acc = 0
            results[fold] = evaluate()

            result_graphs, val_acc = trainer.fit_from(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            val_acc_list.append(val_acc)

            from models import load_model_v1
            load_model_path = get_ckpt_file(result_model_path)
            model = load_model_v1(model, load_model_path)
            model.eval()
            results[fold] = evaluate()
            
            excel_path = os.path.join(model_path, "graph_data.csv")
            # save_graphs_to_excel(result_graphs, excel_path)
            save_graphs_to_excel(result_graphs, excel_path, append=True)
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)

        if experiment == "no_watermark":
            # 预训练模型
            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))
            # 训练模型
            trainer = ClassifierTrainer(
            model=model,
            optimizer=optimizer, 
            device=device, 
            scheduler=scheduler
        )
            model.eval()
            results[fold] = evaluate_nowatermark()
            
            val_acc = results[fold]["correct_watermark"]
            val_acc_list.append(val_acc)
            
            acc = results[fold]["eeg"]
            raw_data_list.append(acc)

            with open(results_path, "w") as f:
                json.dump(experiment_details, f)
        
        # TODO 新水印预训练模型     
        if experiment == "new_watermark_pretrain":

            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))
            trainer = ClassifierTrainer(
            model=model,
            optimizer=optimizer, 
            device=device, 
            scheduler=scheduler
        )

            results[fold] = evaluate_nowatermark()

            acc = results[fold]["correct_watermark"]
            raw_data_list.append(acc)

            acc = results[fold]["wrong_watermark"]    
            val_acc_list.append(acc)

            with open(results_path, "w") as f:
                json.dump(experiment_details, f)
        
        # TODO 新水印训练模型
        if experiment == "new_watermark_from_scratch":
            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))
            trainer = ClassifierTrainer(
            model=model,
            optimizer=optimizer, 
            device=device, 
            scheduler=scheduler
        )

            results[fold] = evaluate_nowatermark()

            acc = results[fold]["correct_watermark"]
            raw_data_list.append(acc)

            acc = results[fold]["wrong_watermark"]    
            val_acc_list.append(acc)

            with open(results_path, "w") as f:
                json.dump(experiment_details, f)

        # TODO 新水印训练模型的保存路径
        if experiment == "pruning_pretrain" or experiment == "pruning_from_scratch":
            
            from pruning import Pruning
            pruning_percent = 1
            prune = getattr(Pruning, pruning_method)()
            
            while pruning_percent < 100:
                print(f"Pruning {pruning_percent}% of the model weights")
                model = get_model(architecture)
                load_path = f'{save_path}/fold-{i}'
                from models import load_model_v1
                model = load_model_v1(model, get_ckpt_file(load_path))
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        prune(module, name = "weight", amount = pruning_percent / 100)
                trainer =  ClassifierTrainer(model=model,optimizer=optimizer,device=device, scheduler=scheduler)

                results[fold][pruning_percent] = evaluate_nowatermark()

                if pruning_mode == "linear":
                    pruning_percent += pruning_delta
                else:
                    pruning_percent = math.ceil(pruning_percent * pruning_delta)

            with open(results_path, "w") as f:
                json.dump(experiment_details, f)
     
        if experiment == "fine_tuning" or experiment == "fine_tuning_from_scratch":
            # 预训练模型
            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))
            import fine_tuning
            fine_tuning_func = getattr(fine_tuning, fine_tuning_mode.upper())
            model = fine_tuning_func(model)
            trainer =  ClassifierTrainer(model=model,optimizer=optimizer,device=device, scheduler=scheduler)

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

            # 在本次epoch之前的准确率
            val_acc = 0
            results[fold] = evaluate()

            result_graphs, val_acc = trainer.fit_white(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            val_acc_list.append(val_acc)

            from models import load_model_v1
            load_model_path = get_ckpt_file(result_model_path)
            model = load_model_v1(model, load_model_path)
            model.eval()
            results[fold] = evaluate()
            
            excel_path = os.path.join(model_path, "graph_data.csv")
            # save_graphs_to_excel(result_graphs, excel_path)
            save_graphs_to_excel(result_graphs, excel_path, append=True)
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)

        if experiment == "transfer_learning" or experiment == "transfer_learning_dense" or experiment == "transfer_learning_add":
            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))
            import transfer_learning
            transfer_learning_model = getattr(transfer_learning, architecture)
            transfer_learning_func = getattr(
                transfer_learning_model, transfer_learning_mode.upper()
            )
            model = transfer_learning_func(model)
            trainer =  ClassifierTrainer(model=model,optimizer=optimizer,device=device, scheduler=scheduler)

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
            pre_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(train_dataset_new, shuffle=True, batch_size=batch_size)

            # 在本次epoch之前的准确率
            val_acc = 0
            results[fold] = evaluate()
            raw_data_acc= results[fold]["eeg"]
            raw_data_list.append(raw_data_acc)

            # 原始的准确率
            result_graphs, val_acc = trainer.fit_white(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            val_acc_list.append(val_acc)


            model = trainer.model
            model.eval()
            results[fold] = evaluate()
            
            # excel_path = os.path.join(model_path, "graph_data.csv")
            # save_graphs_to_excel(result_graphs, excel_path)
            # save_graphs_to_excel(result_graphs, excel_path, append=True)
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)


                # 盗窃模型


        if experiment == "from_transfer_learning" or experiment == "from_transfer_learning_dense" or experiment == "from_transfer_learning_add":
            load_path = f'{save_path}/fold-{i}'
            model = load_model(model, get_ckpt_file(load_path))
            import transfer_learning
            transfer_learning_model = getattr(transfer_learning, architecture)
            transfer_learning_func = getattr(
                transfer_learning_model, transfer_learning_mode.upper()
            )
            model = transfer_learning_func(model)
            trainer =  ClassifierTrainer(model=model,optimizer=optimizer,device=device, scheduler=scheduler)

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
            pre_loader = DataLoader(trig_set, shuffle=True, batch_size=batch_size)
            train_loader = DataLoader(train_dataset_new, shuffle=True, batch_size=batch_size)

            # 在本次epoch之前的准确率
            val_acc = 0
            results[fold] = evaluate()
            raw_data_acc= results[fold]["eeg"]
            raw_data_list.append(raw_data_acc)

            # 原始的准确率
            result_graphs, val_acc = trainer.fit_white(
                train_loader,
                val_loader,
                pre_loader,
                epochs,
                save_path  = result_model_path
            )
            val_acc_list.append(val_acc)


            model = trainer.model
            model.eval()
            results[fold] = evaluate()
            
            # excel_path = os.path.join(model_path, "graph_data.csv")
            # save_graphs_to_excel(result_graphs, excel_path)
            # save_graphs_to_excel(result_graphs, excel_path, append=True)
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)


                # 盗窃模型
    
        if experiment == "Soft_Label_Attack" or experiment == "from_Soft_Label_Attack":
            
            from torch.utils.data import DataLoader
            BATCH_SIZE = batch_size
            EPOCHS = epochs
            LEARNING_RATE = lr
            GAMMA = 0.5  # 正则化系数（用于第三种方法）

            trig_set = TriggerSet(
                tri_path,
                architecture,
                data_type= "id"
            )
            # 加载CIFAR-10数据集
            from triggerset import ModifiedDataset
            test_dataset_new = ModifiedDataset(test_dataset)
            train_dataset_new = ModifiedDataset(train_dataset)

            from torch.utils.data import Dataset, ConcatDataset
            train_dataset_new = ConcatDataset([train_dataset_new, trig_set])
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset_new, batch_size=BATCH_SIZE, shuffle=True)
            pre_loader = DataLoader(trig_set, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(test_dataset_new, batch_size=BATCH_SIZE, shuffle=True)


            # 定义源模型（被窃取的目标模型）
            load_path = f'{save_path}/fold-{i}'
            source_model =load_model(model, get_ckpt_file(load_path))
            source_model.to(devices)
            # source_model.eval()  # 固定源模型参数

            surrogate_model = get_model(architecture)
            surrogate_model = surrogate_model.to(devices)
            # TODO 对模型参数的初始化
            from model_steal import soft_label_attack
            surrogate_model , acc , val_acc = soft_label_attack(source_model, surrogate_model, train_loader,pre_loader,val_loader
                              ,EPOCHS, LEARNING_RATE, result_model_path,devices)
            # acc为触发集 val为验证集
            raw_data_list.append(acc)
            val_acc_list.append(val_acc)
            
            # 检测是否还能检测出触发集
            results[fold] = evaluate_water(surrogate_model)
                
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)
            
        if experiment == "hard_label_attack" or experiment == "from_hard_label_attack":

            from torch.utils.data import DataLoader
            BATCH_SIZE = batch_size
            EPOCHS = epochs
            LEARNING_RATE = lr
            GAMMA = 0.5  # 正则化系数（用于第三种方法）
            trig_set = TriggerSet(
                tri_path,
                architecture,
                data_type= "id"
            )
            # 加载CIFAR-10数据集
            from triggerset import ModifiedDataset
            test_dataset_new = ModifiedDataset(test_dataset)
            train_dataset_new = ModifiedDataset(train_dataset)

            from torch.utils.data import Dataset, ConcatDataset
            train_dataset_new = ConcatDataset([train_dataset_new, trig_set])
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset_new, batch_size=BATCH_SIZE, shuffle=True)
            pre_loader = DataLoader(trig_set, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(test_dataset_new, batch_size=BATCH_SIZE, shuffle=True)

            # 定义源模型（被窃取的目标模型）
            load_path = f'{save_path}/fold-{i}'
            source_model =load_model(model, get_ckpt_file(load_path))
            source_model.to(devices)

            surrogate_model = get_model(architecture)
            surrogate_model = surrogate_model.to(devices)
            # TODO 对模型参数的初始化
            from model_steal import hard_label_attack
            surrogate_model, acc,val_acc = hard_label_attack(source_model, surrogate_model, train_loader,
                                                     pre_loader, val_loader, EPOCHS, LEARNING_RATE, result_model_path, devices)
            raw_data_list.append(acc)
            val_acc_list.append(val_acc)
            # 检测是否还能检测出触发集
            results[fold] = evaluate_water(surrogate_model)
                
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)

        if experiment == "regularization_with_ground_truth" or experiment == "from_regularization_with_ground_truth":
            from torch.utils.data import DataLoader
            BATCH_SIZE = batch_size
            EPOCHS = epochs
            LEARNING_RATE = lr
            GAMMA = 0.5  # 正则化系数（用于第三种方法）
            trig_set = TriggerSet(
                tri_path,
                architecture,
                data_type= "id"
            )
            # 加载CIFAR-10数据集
            from triggerset import ModifiedDataset
            test_dataset_new = ModifiedDataset(test_dataset)
            train_dataset_new = ModifiedDataset(train_dataset)

            from torch.utils.data import Dataset, ConcatDataset
            train_dataset_new = ConcatDataset([train_dataset_new, trig_set])
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset_new, batch_size=BATCH_SIZE, shuffle=True)
            pre_loader = DataLoader(trig_set, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(test_dataset_new, batch_size=BATCH_SIZE, shuffle=True)

            load_path = f'{save_path}/fold-{i}'
            source_model =load_model(model, get_ckpt_file(load_path))
            # source_model.lin2 = nn.Linear(1024, 16)  # 适配CIFAR-10的10分类
            source_model.to(devices)
            # source_model.eval()  # 固定源模型参数

            surrogate_model = get_model(architecture)
            surrogate_model = surrogate_model.to(devices)
            # TODO 对模型参数的初始化
            from model_steal import regularization_with_ground_truth
            surrogate_model, acc,val_acc = regularization_with_ground_truth(source_model, surrogate_model,
                                                                     train_loader,pre_loader,val_loader, GAMMA, EPOCHS, LEARNING_RATE ,result_model_path, devices)
            raw_data_list.append(acc)
            val_acc_list.append(val_acc)

            # 检测是否还能检测出触发集
            results[fold] = evaluate_water(surrogate_model)
                
            with open(results_path, "w") as f:
                json.dump(experiment_details, f)

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
