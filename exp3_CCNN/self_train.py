import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
from scipy import stats  # 用于p值检验

from tqdm import tqdm
from collections import OrderedDict
from scipy.sparse.linalg import svds
from torchvision import datasets, transforms
from IPython import embed
import pandas as pd

C                   = 2
weight_decay        = 1e-5

import torch
import torch.nn.functional as F

def pairwise_euclid_distance(x):
    """Compute pairwise Euclidean distance matrix."""
    x_sq = torch.sum(x**2, dim=1, keepdim=True)
    dist_sq = x_sq + x_sq.t() - 2 * torch.mm(x, x.t())
    return torch.sqrt(torch.clamp(dist_sq, min=0))


def snnl(x, y, t=0.1, metric='euclidean'):
    """数值稳定的Soft Nearest Neighbor Loss"""
    x = F.relu(x)
    batch_size = x.size(0)
    
    # 同类样本掩码（排除自身）
    same_label_mask = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    same_label_mask.fill_diagonal_(0)  # 排除自身
    
    # 计算归一化距离
    if metric == 'euclidean':
        x_flat = x.view(batch_size, -1)
        x_sq = torch.sum(x_flat**2, dim=1, keepdim=True)
        dist_sq = x_sq + x_sq.t() - 2 * torch.mm(x_flat, x_flat.t())
        dist_sq = torch.clamp(dist_sq, min=0)  # 避免负数
        dist = torch.sqrt(dist_sq + 1e-5)  # 防止梯度爆炸
        # 归一化距离到[0,1]
        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-5)
    elif metric == 'cosine':
        x_norm = F.normalize(x, p=2, dim=1)
        dist = 1 - torch.mm(x_norm, x_norm.t())
    else:
        raise ValueError("Unsupported metric")
    
    # 稳定计算 exp(-dist / t)
    max_dist = torch.max(dist)
    exp = torch.exp(-(dist - max_dist) / t)  # 减最大值避免溢出
    exp = exp * same_label_mask  # 仅保留同类样本
    
    # 概率矩阵（稳定log计算）
    prob = exp / (torch.sum(exp, dim=1, keepdim=True) + 1e-5)
    loss = -torch.mean(torch.log(prob + 1e-5))
    
    return loss

class NeuralCollapseLoss(nn.Module):
    def __init__(self, epsilon=5.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, features, target_means, target_labels):
        # features: 触发集样本的特征 [B, D]
        # target_means: 各类别特征均值字典 {class_idx: mean_vector}
        # target_labels: 触发集伪标签 [B]
        losses = []
        for feat, label in zip(features, target_labels):
            mean = target_means[label.item()]
            dist = torch.norm(feat - mean, p=2)  # L2距离
            losses.append(torch.clamp(self.epsilon - dist, min=0))  # hinge loss
        return torch.mean(torch.stack(losses))

# 计算类内特征
@torch.no_grad()
def update_class_means(model, train_loader, num_classes,device):
    model.eval()
    # 计算各类别特征均值
    feature_dim = 1024   # 1024
    class_sums = {i: torch.zeros(feature_dim).cuda() for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}
    
    for x, y in train_loader:
        _, z, _ = model(x.to(device))  # 获取特征
        for i in range(num_classes):
            mask = (y == i)
            if mask.any():
                class_sums[i] += z[mask].sum(dim=0)
                class_counts[i] += mask.sum()
    
    class_means = {i: class_sums[i] / class_counts[i] for i in class_sums}
    return class_means


def get_gradient_signatures(model, dataloader):
    gradients = []
    for imgs, labels in dataloader:
        imgs.requires_grad_()
        outputs = model(imgs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grad = torch.autograd.grad(loss, model.parameters())
        grad_sign = torch.cat([g.sign().flatten() for g in grad])  # 拼接所有参数的梯度符号
        gradients.append(grad_sign)
    return torch.stack(gradients)

class graphs:
    def __init__(self):
        self.accuracy     = []
        self.loss         = []  # dingyi 
        self.reg_loss     = []  # dingyi
        self.collapse_loss = []  # dingyi
        self.nc_loss     = []  # dingyi

        # NC1
        self.Sw_invSb     = []

        # NC2
        self.norm_M_CoV   = []
        self.norm_W_CoV   = []
        self.cos_M        = []
        self.cos_W        = []

        # NC3
        self.W_M_dist     = []
        
        # NC4
        self.NCC_mismatch = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []

class ClassifierTrainer:
    def __init__(self, model, optimizer, device,scheduler):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        self.device = device
        self.best_loss = float('inf')
        self.graphs = graphs()
        self.cur_epochs = []
        self.scheduler = scheduler

    def fit_from(self, train_loader,val_loader,pre_loader, epochs, save_path):
        best_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            loss          = 0
            net_correct   = 0 # 网络预测正确的样本数
            cls_criterion = nn.CrossEntropyLoss()
            collapse_criterion = NeuralCollapseLoss(epsilon=5.0)

            # 训练阶段
            self.model.train()
            train_loss = 0.0
            loss_cls = 0
            loss_collapse = 0
            loss_snnl = 0

            class_means = update_class_means(self.model, train_loader, num_classes=2,device=self.device)
            self.cur_epochs.append(epoch)

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (data, target) in enumerate(pbar, start=1):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                # 主任务损失
                out, trigger_features,middel_features  = self.model(data)
                loss_cls = cls_criterion(out, target)
                # EWE损失
                watermark_mask = (target == 1)  # 水印数据掩码
                task_mask = (target != 1)       # 任务数据掩码
                if watermark_mask.sum() > 0 and task_mask.sum() > 0:
                    features = torch.cat([middel_features[watermark_mask], middel_features[task_mask]], dim=0)
                    labels = torch.cat([
                        torch.ones(watermark_mask.sum(), device=self.device),  # 水印标记为1
                        torch.zeros(task_mask.sum(), device=self.device)       # 任务数据标记为0
                    ], dim=0)
                    
                    loss_snnl = snnl(features, labels, t=0.5)  # 温度参数t可调
                    # loss = loss_cls + 0.1 * loss_snnl  # 权重可调
                else:
                    loss_snnl = 0
                # 水印损失
                loss_collapse = collapse_criterion(trigger_features, class_means, target)
                loss = loss_cls + 0.1 *loss_collapse + 0.01 * loss_snnl  # 权重可调 
                # 反向传播
                loss.backward()
                self.optimizer.step()
                # 累计损失和准确率
                train_loss += loss.item()
                loss_cls += loss_cls.item()
                loss_collapse += loss_collapse.item()
                loss_snnl += loss_snnl.item()
                # 计算准确率
                _, predicted  = torch.max(out, 1)
                correct = (predicted == target).sum().item()
                net_correct += correct
                # print(f"Batch {batch_idx}: loss={loss.item()}")
                pbar.set_postfix(loss=loss.item(), acc=correct/data.size(0))

            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_loss_cls = loss_cls / len(train_loader)
            avg_loss_collapse = loss_collapse / len(train_loader)
            avg_loss_snnl = loss_snnl / len(train_loader)

            net_acc = net_correct / len(train_loader.dataset)
            net_acc = 100. * net_acc
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")
            self.graphs.reg_loss.append(avg_loss_cls)
            self.graphs.collapse_loss.append(avg_loss_collapse)
            self.graphs.nc_loss.append(avg_loss_snnl)
            self.graphs.loss.append(avg_train_loss)

            self.graphs.accuracy.append(net_acc)


            if net_acc > best_acc:
                best_acc = net_acc
                best_epoch  = epoch
                
                val_loss = 0.0
                total = 0
                correct = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features,middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(val_loader.dataset)
                    # 计算准确率
                    val_accuracy = 100.0 * correct / total

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(pre_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features,middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(pre_loader.dataset)
                    # 计算准确率
                    tri_acc = 100.0 * correct / total

                save_path_1 = os.path.join(save_path, f"best_model_{best_epoch}__{best_acc:.4f}_{tri_acc:.4f}_{val_accuracy:.4f}.ckpt")
                torch.save(self.model.state_dict(), save_path_1)
                print(f"Saving model to {save_path_1}")
            
            if (epoch+1) % 10 == 0:
                val_loss = 0.0
                total = 0
                correct = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features, middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(val_loader.dataset)
                    # 计算准确率
                    val_accuracy = 100.0 * correct / total

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(pre_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features, middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(pre_loader.dataset)
                    # 计算准确率
                    tri_acc = 100.0 * correct / total

                print(f"Validation:epoch={epoch}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")

                save_path_1 = os.path.join(save_path, f"{epoch}_{net_acc:.4f}_{tri_acc:.4f}_{val_accuracy:.4f}.ckpt")
                torch.save(self.model.state_dict(), save_path_1)
                print(f"Saving model to {save_path_1}")

        # # 验证阶段
        val_loss = 0.0
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader, start=1):
                data, target = data.to(self.device), target.to(self.device)
                out, _ , middel_features = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
            # 计算准确率
            finall_val_accuracy = 100.0 * correct / total

        print(f"Validation: val_loss={val_loss:.4f}, val_acc={finall_val_accuracy:.4f}")
        return self.graphs,finall_val_accuracy

    def fit_from_EWE(self, train_loader,val_loader,pre_loader, epochs, save_path):
        best_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            loss          = 0
            net_correct   = 0 # 网络预测正确的样本数
            cls_criterion = nn.CrossEntropyLoss()
            collapse_criterion = NeuralCollapseLoss(epsilon=5.0)

            # 训练阶段
            self.model.train()
            train_loss = 0.0
            loss_cls = 0
            loss_snnl = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (data, target,label) in enumerate(pbar, start=1):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                # 主任务损失
                out, trigger_features,middel_features  = self.model(data)
                loss_cls = cls_criterion(out, target)
                # EWE损失
                watermark_mask = (label == 1)  # 水印数据掩码
                task_mask = (label != 1)       # 任务数据掩码
                if watermark_mask.sum() > 0 and task_mask.sum() > 0:
                    features = torch.cat([middel_features[watermark_mask], middel_features[task_mask]], dim=0)
                    labels = torch.cat([
                        torch.ones(watermark_mask.sum(), device=self.device),  # 水印标记为1
                        torch.zeros(task_mask.sum(), device=self.device)       # 任务数据标记为0
                    ], dim=0)
                    loss_snnl = snnl(features, labels, t=0.5)  # 温度参数t可调
                else:
                    loss_snnl = 0
                # 水印损失
                loss = loss_cls + 0.01 * loss_snnl  # 权重可调 
                # 反向传播
                loss.backward()
                self.optimizer.step()
                # 累计损失和准确率
                train_loss += loss.item()
                loss_cls += loss_cls.item()
                if loss_snnl != 0:
                    loss_snnl += loss_snnl.item()
                # 计算准确率
                _, predicted  = torch.max(out, 1)
                correct = (predicted == target).sum().item()
                net_correct += correct
                pbar.set_postfix(loss=loss.item(), acc=correct/data.size(0))

            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_loss_cls = loss_cls / len(train_loader)
            avg_loss_snnl = loss_snnl / len(train_loader)

            net_acc = net_correct / len(train_loader.dataset)
            net_acc = 100. * net_acc
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")
            self.graphs.reg_loss.append(avg_loss_cls)
            self.graphs.nc_loss.append(avg_loss_snnl)
            self.graphs.loss.append(avg_train_loss)
            self.graphs.accuracy.append(net_acc)


            if net_acc > best_acc:
                best_acc = net_acc
                best_epoch  = epoch
                
                val_loss = 0.0
                total = 0
                correct = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target,label) in enumerate(val_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features,middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(val_loader.dataset)
                    # 计算准确率
                    val_accuracy = 100.0 * correct / total

                with torch.no_grad():
                    for batch_idx, (data, target,label) in enumerate(pre_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features,middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(pre_loader.dataset)
                    # 计算准确率
                    tri_acc = 100.0 * correct / total

                save_path_1 = os.path.join(save_path, f"best_model_{best_epoch}__{best_acc:.4f}_{tri_acc:.4f}_{val_accuracy:.4f}.ckpt")
                torch.save(self.model.state_dict(), save_path_1)
                print(f"Saving model to {save_path_1}")
            
            if (epoch+1) % 10 == 0:
                val_loss = 0.0
                total = 0
                correct = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target,label) in enumerate(val_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features, middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(val_loader.dataset)
                    # 计算准确率
                    val_accuracy = 100.0 * correct / total

                with torch.no_grad():
                    for batch_idx, (data, target,label) in enumerate(pre_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features, middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(pre_loader.dataset)
                    # 计算准确率
                    tri_acc = 100.0 * correct / total

                print(f"Validation:epoch={epoch}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")

                save_path_1 = os.path.join(save_path, f"{epoch}_{net_acc:.4f}_{tri_acc:.4f}_{val_accuracy:.4f}.ckpt")
                torch.save(self.model.state_dict(), save_path_1)
                print(f"Saving model to {save_path_1}")

        # # 验证阶段
        val_loss = 0.0
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target,label) in enumerate(val_loader, start=1):
                data, target = data.to(self.device), target.to(self.device)
                out, _ , middel_features = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
            # 计算准确率
            finall_val_accuracy = 100.0 * correct / total

        print(f"Validation: val_loss={val_loss:.4f}, val_acc={finall_val_accuracy:.4f}")
        return self.graphs,finall_val_accuracy

    def fit_from_RS(self, train_loader,val_loader,pre_loader, epochs, save_path):
        best_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            loss          = 0
            net_correct   = 0 # 网络预测正确的样本数
            cls_criterion = nn.CrossEntropyLoss()
            collapse_criterion = NeuralCollapseLoss(epsilon=5.0)

            # 训练阶段
            self.model.train()
            train_loss = 0.0


            class_means = update_class_means(self.model, train_loader, num_classes=2,device=self.device)
            self.cur_epochs.append(epoch)

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (data, target) in enumerate(pbar, start=1):
                data, target = data.to(self.device), target.to(self.device)
                data = data + torch.randn_like(data, device='cuda') * 0.3

                self.optimizer.zero_grad()
                # 主任务损失
                out, trigger_features,middel_features  = self.model(data)
                loss_cls = cls_criterion(out, target)
                
                loss = loss_cls
                # 反向传播
                loss.backward()
                self.optimizer.step()
                # 累计损失和准确率
                train_loss += loss.item()

                # 计算准确率
                _, predicted  = torch.max(out, 1)
                correct = (predicted == target).sum().item()
                net_correct += correct
                # print(f"Batch {batch_idx}: loss={loss.item()}")
                pbar.set_postfix(loss=loss.item(), acc=correct/data.size(0))

            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            net_acc = net_correct / len(train_loader.dataset)
            net_acc = 100. * net_acc
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")
            self.graphs.loss.append(avg_train_loss)

            self.graphs.accuracy.append(net_acc)

            if net_acc > best_acc:
                best_acc = net_acc
                best_epoch  = epoch
                
                val_loss = 0.0
                total = 0
                correct = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features,middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(val_loader.dataset)
                    # 计算准确率
                    val_accuracy = 100.0 * correct / total

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(pre_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features,middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(pre_loader.dataset)
                    # 计算准确率
                    tri_acc = 100.0 * correct / total

                save_path_1 = os.path.join(save_path, f"best_model_{best_epoch}__{best_acc:.4f}_{tri_acc:.4f}_{val_accuracy:.4f}.ckpt")
                torch.save(self.model.state_dict(), save_path_1)
                print(f"Saving model to {save_path_1}")
            
            if (epoch+1) % 10 == 0:
                val_loss = 0.0
                total = 0
                correct = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features, middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(val_loader.dataset)
                    # 计算准确率
                    val_accuracy = 100.0 * correct / total

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(pre_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features, middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(pre_loader.dataset)
                    # 计算准确率
                    tri_acc = 100.0 * correct / total

                print(f"Validation:epoch={epoch}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")

                save_path_1 = os.path.join(save_path, f"{epoch}_{net_acc:.4f}_{tri_acc:.4f}_{val_accuracy:.4f}.ckpt")
                torch.save(self.model.state_dict(), save_path_1)
                print(f"Saving model to {save_path_1}")

        # # 验证阶段
        val_loss = 0.0
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader, start=1):
                data, target = data.to(self.device), target.to(self.device)
                out, _ , middel_features = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
            # 计算准确率
            finall_val_accuracy = 100.0 * correct / total

        print(f"Validation: val_loss={val_loss:.4f}, val_acc={finall_val_accuracy:.4f}")
        return self.graphs,finall_val_accuracy

    def fit_from_DI(self, train_loader,val_loader,pre_loader, epochs, save_path):
        best_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            loss          = 0
            net_correct   = 0 # 网络预测正确的样本数
            cls_criterion = nn.CrossEntropyLoss()

            # 训练阶段
            self.model.train()
            train_loss = 0.0
            self.cur_epochs.append(epoch)

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (data, target) in enumerate(pbar, start=1):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                # 主任务损失
                out, trigger_features,middel_features  = self.model(data)
                loss_cls = cls_criterion(out, target)
                
                loss = loss_cls
                # 反向传播
                loss.backward()
                self.optimizer.step()
                # 累计损失和准确率
                train_loss += loss.item()

                # 计算准确率
                _, predicted  = torch.max(out, 1)
                correct = (predicted == target).sum().item()
                net_correct += correct
                # print(f"Batch {batch_idx}: loss={loss.item()}")
                pbar.set_postfix(loss=loss.item(), acc=correct/data.size(0))

            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            net_acc = net_correct / len(train_loader.dataset)
            net_acc = 100. * net_acc
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")
            self.graphs.loss.append(avg_train_loss)

            self.graphs.accuracy.append(net_acc)

            if net_acc > best_acc:
                best_acc = net_acc
                best_epoch  = epoch
                
                val_loss = 0.0
                total = 0
                correct = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features,middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(val_loader.dataset)
                    # 计算准确率
                    val_accuracy = 100.0 * correct / total

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(pre_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features,middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(pre_loader.dataset)
                    # 计算准确率
                    tri_acc = 100.0 * correct / total

                save_path_1 = os.path.join(save_path, f"best_model_{best_epoch}__{best_acc:.4f}_{tri_acc:.4f}_{val_accuracy:.4f}.ckpt")
                torch.save(self.model.state_dict(), save_path_1)
                print(f"Saving model to {save_path_1}")
            
            if (epoch+1) % 10 == 0:
                val_loss = 0.0
                total = 0
                correct = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features, middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(val_loader.dataset)
                    # 计算准确率
                    val_accuracy = 100.0 * correct / total

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(pre_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features, middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(pre_loader.dataset)
                    # 计算准确率
                    tri_acc = 100.0 * correct / total

                print(f"Validation:epoch={epoch}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")

                save_path_1 = os.path.join(save_path, f"{epoch}_{net_acc:.4f}_{tri_acc:.4f}_{val_accuracy:.4f}.ckpt")
                torch.save(self.model.state_dict(), save_path_1)
                print(f"Saving model to {save_path_1}")

        # # 验证阶段
        val_loss = 0.0
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader, start=1):
                data, target = data.to(self.device), target.to(self.device)
                out, _ , middel_features = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
            # 计算准确率
            finall_val_accuracy = 100.0 * correct / total

        print(f"Validation: val_loss={val_loss:.4f}, val_acc={finall_val_accuracy:.4f}")
        return self.graphs,finall_val_accuracy


    
    def fit_from_EEF(self, train_loader,val_loader,pre_loader, epochs, save_path):
        best_acc = 0.0
        best_epoch = 0
        
        for epoch in range(epochs):
            loss          = 0
            net_correct   = 0 # 网络预测正确的样本数
            cls_criterion = nn.CrossEntropyLoss()
            collapse_criterion = NeuralCollapseLoss(epsilon=5.0)

            # 训练阶段
            self.model.train()
            train_loss = 0.0


            class_means = update_class_means(self.model, train_loader, num_classes=2,device=self.device)
            self.cur_epochs.append(epoch)

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (data, target) in enumerate(pbar, start=1):
                data, target = data.to(self.device), target.to(self.device)
                # data = data + torch.randn_like(data, device='cuda') * 0.3

                self.optimizer.zero_grad()
                # 主任务损失
                out, trigger_features,middel_features  = self.model(data)
                loss_cls = cls_criterion(out, target)
                
                loss = loss_cls
                # 反向传播
                loss.backward()
                self.optimizer.step()
                # 累计损失和准确率
                train_loss += loss.item()

                # 计算准确率
                _, predicted  = torch.max(out, 1)
                correct = (predicted == target).sum().item()
                net_correct += correct
                # print(f"Batch {batch_idx}: loss={loss.item()}")
                pbar.set_postfix(loss=loss.item(), acc=correct/data.size(0))

            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            net_acc = net_correct / len(train_loader.dataset)
            net_acc = 100. * net_acc
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")
            self.graphs.loss.append(avg_train_loss)

            self.graphs.accuracy.append(net_acc)

            if net_acc > best_acc:
                best_acc = net_acc
                best_epoch  = epoch
                
                val_loss = 0.0
                total = 0
                correct = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features,middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(val_loader.dataset)
                    # 计算准确率
                    val_accuracy = 100.0 * correct / total

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(pre_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features,middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(pre_loader.dataset)
                    # 计算准确率
                    tri_acc = 100.0 * correct / total

                save_path_1 = os.path.join(save_path, f"best_model_{best_epoch}__{best_acc:.4f}_{tri_acc:.4f}_{val_accuracy:.4f}.ckpt")
                torch.save(self.model.state_dict(), save_path_1)
                print(f"Saving model to {save_path_1}")
            
            if (epoch+1) % 10 == 0:
                val_loss = 0.0
                total = 0
                correct = 0
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features, middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(val_loader.dataset)
                    # 计算准确率
                    val_accuracy = 100.0 * correct / total

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(pre_loader, start=1):
                        data, target = data.to(self.device), target.to(self.device)
                        out, trigger_features, middel_features  = self.model(data)
                        loss_cls = cls_criterion(out, target)
                        val_loss += loss_cls.item()
                        _, predicted = torch.max(out, 1)  # 获取预测类别
                        total += target.size(0)  # 更新总样本数
                        correct += (predicted == target).sum().item()  # 更新正确预测的样本数
                    self.scheduler.step(val_loss)
                    # 计算平均验证损失
                    val_loss /= len(pre_loader.dataset)
                    # 计算准确率
                    tri_acc = 100.0 * correct / total

                print(f"Validation:epoch={epoch}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")

                save_path_1 = os.path.join(save_path, f"{epoch}_{net_acc:.4f}_{tri_acc:.4f}_{val_accuracy:.4f}.ckpt")
                torch.save(self.model.state_dict(), save_path_1)
                print(f"Saving model to {save_path_1}")

        # # 验证阶段
        val_loss = 0.0
        total = 0
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader, start=1):
                data, target = data.to(self.device), target.to(self.device)
                out, _ , middel_features = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
            # 计算准确率
            finall_val_accuracy = 100.0 * correct / total

        print(f"Validation: val_loss={val_loss:.4f}, val_acc={finall_val_accuracy:.4f}")
        return self.graphs,finall_val_accuracy
    
    def test_EEF(self,train_loader,pre_loader,source_model,benign_model,meta_classifier):
        X_source = get_gradient_signatures(source_model, pre_loader)
        X_benign = get_gradient_signatures(benign_model, pre_loader)
        X = torch.cat([X_source, X_benign])
        y = torch.cat([torch.ones(len(X_source)), torch.zeros(len(X_benign))])
        # 训练简单的元分类器

        optimizer = optim.Adam(meta_classifier.parameters())
        criterion = nn.BCELoss()

        for epoch in range(10):
            optimizer.zero_grad()
            preds = meta_classifier(X).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        return meta_classifier

    def verify_ownership(self,meta_classifier,suspect_model,benign_model,pre_loader, alpha=0.01):
        # 计算可疑模型的梯度响应
        suspect_grads = get_gradient_signatures(suspect_model, pre_loader)
        suspect_scores = meta_classifier(suspect_grads.float()).squeeze().detach().numpy()
        
        # 计算良性模型的梯度响应（基线）
        benign_grads = get_gradient_signatures(benign_model, pre_loader)
        benign_scores = meta_classifier(benign_grads.float()).squeeze().detach().numpy()
        
        # 双样本T检验
        t_stat, p_value = stats.ttest_ind(suspect_scores, benign_scores, alternative='greater')
        
        print(f"p-value = {p_value:.4f}")
        return p_value, alpha

    def test(self,trig_dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(trig_dataloader, start=1):
                data, target = data.to(self.device), target.to(self.device)
                out, trigger_features, middel_features  = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
        # 计算准确率
        test_accuracy = 100.0 * correct / total
        print(f"Test: test_acc={test_accuracy:.4f}")
        return test_accuracy
