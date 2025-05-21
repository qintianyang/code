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

from tqdm import tqdm
from collections import OrderedDict
from scipy.sparse.linalg import svds
from torchvision import datasets, transforms
from IPython import embed
import pandas as pd

C                   = 2
weight_decay        = 1e-5



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
    feature_dim = 1024
    class_sums = {i: torch.zeros(feature_dim).cuda() for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}
    
    for x, y in train_loader:
        _, z = model(x.to(device))  # 获取特征
        for i in range(num_classes):
            mask = (y == i)
            if mask.any():
                class_sums[i] += z[mask].sum(dim=0)
                class_counts[i] += mask.sum()
    
    class_means = {i: class_sums[i] / class_counts[i] for i in class_sums}
    return class_means

class graphs:
    def __init__(self):
        self.accuracy     = []
        self.loss         = []  # dingyi 
        self.reg_loss     = []  # dingyi

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

    def to_excel(self, filename, append=False):
    # 创建一个字典来存储所有数据
            data = {
                'accuracy': self.accuracy,
                'loss': self.loss,
                'reg_loss': self.reg_loss,
                'Sw_invSb': self.Sw_invSb,
                'norm_M_CoV': self.norm_M_CoV,
                'norm_W_CoV': self.norm_W_CoV,
                'cos_M': self.cos_M,
                'cos_W': self.cos_W,
                'W_M_dist': self.W_M_dist,
                'NCC_mismatch': self.NCC_mismatch,
                'MSE_wd_features': self.MSE_wd_features,
                'LNC1': self.LNC1,
                'LNC23': self.LNC23,
                'Lperp': self.Lperp
            }
            
            # 找出最长的列表长度
            max_length = max(len(v) for v in data.values())
            
            # 将所有列表扩展到相同长度，用None填充不足的部分
            for key in data:
                if len(data[key]) < max_length:
                    data[key].extend([None] * (max_length - len(data[key])))
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 保存为Excel文件
            # df.to_excel(filename, index=False)
            df.to_csv(filename, mode='a' if append else 'w', header=not append, index=False)
            print(f"数据已保存到 {filename}")




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

            N             = [0 for _ in range(C)]
            mean          = [0 for _ in range(C)] # 存储每个类别的特征均值
            Sw            = 0 #存储类内散度矩阵（within-class scatter matrix）
            loss          = 0
            net_correct   = 0 # 网络预测正确的样本数
            NCC_match_net = 0 # 最近邻分类器（NCC）与网络预测一致的样本数
            cls_criterion = nn.CrossEntropyLoss()
            collapse_criterion = NeuralCollapseLoss(epsilon=5.0)

            # 训练阶段
            self.model.train()
            train_loss = 0.0
            sum_reg_loss = 0.0

            if (epoch+1) % 5 == 0:
                class_means = update_class_means(self.model, train_loader, num_classes=2,device=self.device)
                self.cur_epochs.append(epoch)
                for batch_idx, (data, target) in enumerate(train_loader, start=1):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    # 主任务损失
                    out, trigger_features  = self.model(data)
                    loss_cls = cls_criterion(out, target)
                    # 水印损失
                    loss_collapse = collapse_criterion(trigger_features, class_means, target)
                    loss = loss_cls + 0.1 *loss_collapse  # 权重可调
                    total_loss = loss 
                    # 反向传播
                    total_loss.backward()
                    self.optimizer.step()
                    # 累计损失和准确率
                    train_loss += loss.item()
                    # 计算准确率
                    _, predicted  = torch.max(out, 1)
                    correct = (predicted == target).sum().item()
                    net_correct += correct
                    # 计算平均损失和准确率
                avg_train_loss = train_loss / len(train_loader)
                net_acc = net_correct / len(train_loader.dataset)
                net_acc = 100. * net_acc
                print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")
                self.graphs.reg_loss.append(avg_train_loss)
                self.graphs.loss.append(avg_train_loss)
                self.graphs.accuracy.append(net_acc)

                for computation in ['Mean','Cov']:  # Mean 每个类别的特征均值  Cov 计算类内协方差矩阵和其他指标
                    train_loss = 0.0
                    reg_loss = 0.0
                    net_correct = 0
                    NCC_match_net = 0
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(train_loader, start=1):
                            data, target = data.to(self.device), target.to(self.device)
                            # 主任务损失
                            out, trigger_features  = self.model(data)
                            loss_cls = cls_criterion(out, target)
                            # 水印损失
                            loss_collapse = collapse_criterion(trigger_features, class_means, target)
                            loss = loss_cls + 0.1 * loss_collapse  # 权重可调
                            train_loss += loss.item()

                            reg_loss = train_loss
                            # 计算准确率
                            _, predicted  = torch.max(out, 1)
                            correct = (predicted == target).sum().item()
                            net_correct += correct

                            for c in range(C):
                                # 获取当前 batch 中属于类别 c 的样本索引
                                idxs = (target == c).nonzero(as_tuple=True)[0]
                                if len(idxs) == 0: # If no class-c in this batch
                                    continue
                                h_c = trigger_features[idxs,:] # B CHW
                                if computation == 'Mean':
                                    # 累加当前 batch 中类别 c 的特征（用于后续计算均值）
                                    mean[c] += torch.sum(h_c, dim=0)  # 按样本维度求和 → shape: [CHW]
                                    N[c] += h_c.shape[0]  # 记录类别 c 的样本总数
                                elif computation == 'Cov':
                                    # update within-class cov
                                    z = h_c - mean[c].unsqueeze(0) # B CHW
                                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                                        z.unsqueeze(1))  # B 1 CHW
                                    # print(cov.shape)

                                    Sw += torch.sum(cov, dim=0)
                                    # during calculation of within-class covariance, calculate:
                                    # 1) network's accuracy
                                    net_pred = torch.argmax(out[idxs,:], dim=1)
                                    # net_correct += sum(net_pred==target[idxs]).item()
                                    # 2) agreement between prediction and nearest class center
                                    NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                                                for i in range(h_c.shape[0])])
                                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                                    NCC_match_net += sum(NCC_pred==net_pred).item()

                    if computation == 'Mean':
                        for c in range(C):
                            mean[c] /= N[c] # 均值归一化
                            M = torch.stack(mean).T # 将均值堆叠并转置
                        train_loss /= len(train_loader.dataset)
                    elif computation == 'Cov':
                        Sw /= sum(N) # 将类内协方差矩阵除以总样本数
      
                self.graphs.NCC_mismatch.append(1-NCC_match_net/sum(N))

                # global mean
                muG = torch.mean(M, dim=1, keepdim=True) # CHW 1  # 计算所有类别均值的平均值
            
                # between-class 计算类间协方差矩阵 Sb
                M_ = M - muG  
                Sb = torch.matmul(M_, M_.T) / C  # 计算类别均值之间的协方差矩阵

                # avg norm
                W  = self.model.lin2.weight              # 分类器权重: [C, CHW] 或 [CHW, C]（取决于定义）
                M_norms = torch.norm(M_, dim=0)          # 类别均值的L2范数: [C]
                W_norms = torch.norm(W.T, dim=0)         # 分类器权重的L2范数: [C]

                self.graphs.norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
                self.graphs.norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item()) 

                # tr{Sw Sb^-1}
                Sw = Sw.cpu().detach().numpy()  
                Sb = Sb.cpu().detach().numpy()
                # 对Sb进行奇异值分解（SVD）并求伪逆
                eigvec, eigval, _ = svds(Sb, k=C-1)
                inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
                self.graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb))  # 类内协方差与类间协方差的比值，反映了分类边界的分离程度  值越小：类内紧致且类间分离良好

                # ||W^T - M_||
                normalized_M = M_ / torch.norm(M_,'fro')
                normalized_W = W.T / torch.norm(W.T,'fro')
                self.graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M)**2).item())  
                 # 归一化后的权重矩阵与类别均值矩阵之间的距离，反映两者之间的相似性
                # 离趋近于 0: 权重 W 与类别均值 M_ 高度对齐，分类器直接利用特征空间的几何结构

                # mutual coherence
                def coherence(V): 
                    G = V.T @ V
                    G += torch.ones((C,C),device= self.device) / (C-1)
                    G -= torch.diag(torch.diag(G))
                    return torch.norm(G,1).item() / (C*(C-1))

                self.graphs.cos_M.append(coherence(M_/M_norms))
                self.graphs.cos_W.append(coherence(W.T/W_norms))  
                #分别计算类别均值矩阵和权重矩阵的互相关性，衡量它们列向量之间的相似性
                # 值接近 0 → 均值向量接近正交（类别可分性强）；值接近 1 → 权重矩阵的列向量高度相关（类别间相关性强）
                # 计算特征空间的局部嵌入空间

            else:
            # 损失函数
                class_means = update_class_means(self.model, train_loader, num_classes=2,device=self.device)
                for batch_idx, (data, target) in enumerate(train_loader, start=1):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    # 主任务损失
                    out, trigger_features  = self.model(data)
                    loss_cls = cls_criterion(out, target)

                    # 计算准确率
                    _, predicted  = torch.max(out, 1)
                    correct = (predicted == target).sum().item()
                    net_correct += correct

                    
                    loss_collapse = collapse_criterion(trigger_features, class_means, target)
                    loss = loss_cls + 0.1 * loss_collapse  # 权重可调
                    
                    # loss = loss_cls   # 权重可调
                    # l2_reg = torch.tensor(0.).to(self.device)
                    # for param in self.model.parameters():
                    #     l2_reg += torch.sum(param**2)
                    # num_params = sum(p.numel() for p in self.model.parameters())
                    reg_loss = loss

                    reg_loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    sum_reg_loss += reg_loss.item()
                
                
                net_correct /= len(train_loader.dataset)
                net_acc = 100.0 * net_correct
                avg_train_loss = train_loss / len(train_loader)
                print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")

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
                            out, trigger_features  = self.model(data)
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
                            out, trigger_features  = self.model(data)
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
                        out, trigger_features  = self.model(data)
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
                        out, trigger_features  = self.model(data)
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
                out, _  = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
            # 计算准确率
            finall_val_accuracy = 100.0 * correct / total

        print(f"Validation: val_loss={val_loss:.4f}, val_acc={finall_val_accuracy:.4f}")

        # loss = self.graphs.reg_loss.cpu().detach().numpy()
        # 绘制图表
        plt.figure(1)
        plt.semilogy(self.cur_epochs, self.graphs.reg_loss)
        plt.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Loss')
        save_path_2 = os.path.join(save_path, 'Training Loss.png')
        plt.savefig(save_path_2)

        plt.figure(2)
        plt.plot(self.cur_epochs, 100*(1 - np.array(self.graphs.accuracy)))
        plt.xlabel('Epoch')
        plt.ylabel('Training Error (%)')
        plt.title('Training Error')
        save_path_3 = os.path.join(save_path, 'Training Error.png')
        plt.savefig(save_path_3)

        plt.figure(3)
        plt.semilogy(self.cur_epochs, self.graphs.Sw_invSb)
        plt.xlabel('Epoch')
        plt.ylabel('Tr{Sw Sb^-1}')
        plt.title('NC1: Activation Collapse')
        save_path_4 = os.path.join(save_path, 'NC1: Activation Collapse.png')
        plt.savefig(save_path_4)

        plt.figure(4)
        plt.plot(self.cur_epochs, self.graphs.norm_M_CoV)
        plt.plot(self.cur_epochs, self.graphs.norm_W_CoV)
        plt.legend(['Class Means','Classifiers'])
        plt.xlabel('Epoch')
        plt.ylabel('Std/Avg of Norms')
        plt.title('NC2: Equinorm')
        save_path_5 = os.path.join(save_path, 'NC2: Equinorm.png')
        plt.savefig(save_path_5)

        plt.plot(self.cur_epochs, self.graphs.cos_M)
        plt.plot(self.cur_epochs, self.graphs.cos_W)
        plt.legend(['Class Means','Classifiers'])
        plt.xlabel('Epoch')
        plt.ylabel('Avg|Cos + 1/(C-1)|')
        plt.title('NC2: Maximal Equiangularity')
        save_path_6 = os.path.join(save_path, 'NC2_Maximal_Equiangularity.png')
        plt.savefig(save_path_6)

        plt.figure(6)
        plt.plot(self.cur_epochs,self.graphs.W_M_dist)
        plt.xlabel('Epoch')
        plt.ylabel('||W^T - H||^2')
        plt.title('NC3: Self Duality')
        save_path_7 = os.path.join(save_path, 'NC3: Self Duality.png')
        plt.savefig(save_path_7)

        plt.figure(7)
        plt.plot(self.cur_epochs,self.graphs.NCC_mismatch)
        plt.xlabel('Epoch')
        plt.ylabel('Proportion Mismatch from NCC')
        plt.title('NC4: Convergence to NCC')
        save_path_8 = os.path.join(save_path, 'NC4:Convergence to NCC.png')
        plt.savefig(save_path_8)

        plt.show()
        return self.graphs,finall_val_accuracy

    def fit_white(self, train_loader,val_loader,pre_loader, epochs, save_path):
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(epochs):

            N             = [0 for _ in range(C)]
            mean          = [0 for _ in range(C)] # 存储每个类别的特征均值
            Sw            = 0 #存储类内散度矩阵（within-class scatter matrix）
            loss          = 0
            net_correct   = 0 # 网络预测正确的样本数
            NCC_match_net = 0 # 最近邻分类器（NCC）与网络预测一致的样本数
            cls_criterion = nn.CrossEntropyLoss()
            collapse_criterion = NeuralCollapseLoss(epsilon=5.0)

            # 训练阶段
            self.model.train()
            train_loss = 0.0
            sum_reg_loss = 0.0

            if (epoch+1) == -1 :
                class_means = update_class_means(self.model, train_loader, num_classes=2,device=self.device)
                self.cur_epochs.append(epoch)
                for batch_idx, (data, target) in enumerate(train_loader, start=1):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    # 主任务损失
                    out, trigger_features  = self.model(data)
                    loss_cls = cls_criterion(out, target)
                    # 水印损失
                    loss_collapse = collapse_criterion(trigger_features, class_means, target)
                    loss = loss_cls + 0.1 *loss_collapse  # 权重可调
                    total_loss = loss 
                    # 反向传播
                    total_loss.backward()
                    self.optimizer.step()
                    # 累计损失和准确率
                    train_loss += loss.item()
                    # 计算准确率
                    _, predicted  = torch.max(out, 1)
                    correct = (predicted == target).sum().item()
                    net_correct += correct
                    # 计算平均损失和准确率
                avg_train_loss = train_loss / len(train_loader)
                net_acc = net_correct / len(train_loader.dataset)
                net_acc = 100. * net_acc
                print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")
                self.graphs.reg_loss.append(avg_train_loss)
                self.graphs.loss.append(avg_train_loss)
                self.graphs.accuracy.append(net_acc)

                for computation in ['Mean','Cov']:  # Mean 每个类别的特征均值  Cov 计算类内协方差矩阵和其他指标
                    train_loss = 0.0
                    reg_loss = 0.0
                    net_correct = 0
                    NCC_match_net = 0
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(train_loader, start=1):
                            data, target = data.to(self.device), target.to(self.device)
                            # 主任务损失
                            out, trigger_features  = self.model(data)
                            loss_cls = cls_criterion(out, target)
                            # 水印损失
                            loss_collapse = collapse_criterion(trigger_features, class_means, target)
                            loss = loss_cls + 0.1 * loss_collapse  # 权重可调
                            train_loss += loss.item()

                            reg_loss = train_loss
                            # 计算准确率
                            _, predicted  = torch.max(out, 1)
                            correct = (predicted == target).sum().item()
                            net_correct += correct

                            for c in range(C):
                                # 获取当前 batch 中属于类别 c 的样本索引
                                idxs = (target == c).nonzero(as_tuple=True)[0]
                                if len(idxs) == 0: # If no class-c in this batch
                                    continue
                                h_c = trigger_features[idxs,:] # B CHW
                                if computation == 'Mean':
                                    # 累加当前 batch 中类别 c 的特征（用于后续计算均值）
                                    mean[c] += torch.sum(h_c, dim=0)  # 按样本维度求和 → shape: [CHW]
                                    N[c] += h_c.shape[0]  # 记录类别 c 的样本总数
                                elif computation == 'Cov':
                                    # update within-class cov
                                    z = h_c - mean[c].unsqueeze(0) # B CHW
                                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                                        z.unsqueeze(1))  # B 1 CHW
                                    # print(cov.shape)

                                    Sw += torch.sum(cov, dim=0)
                                    # during calculation of within-class covariance, calculate:
                                    # 1) network's accuracy
                                    net_pred = torch.argmax(out[idxs,:], dim=1)
                                    # net_correct += sum(net_pred==target[idxs]).item()
                                    # 2) agreement between prediction and nearest class center
                                    NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                                                for i in range(h_c.shape[0])])
                                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                                    NCC_match_net += sum(NCC_pred==net_pred).item()

                    if computation == 'Mean':
                        for c in range(C):
                            mean[c] /= N[c] # 均值归一化
                            M = torch.stack(mean).T # 将均值堆叠并转置
                        train_loss /= len(train_loader.dataset)
                    elif computation == 'Cov':
                        Sw /= sum(N) # 将类内协方差矩阵除以总样本数
      
                self.graphs.NCC_mismatch.append(1-NCC_match_net/sum(N))

                # global mean
                muG = torch.mean(M, dim=1, keepdim=True) # CHW 1  # 计算所有类别均值的平均值
            
                # between-class 计算类间协方差矩阵 Sb
                M_ = M - muG  
                Sb = torch.matmul(M_, M_.T) / C  # 计算类别均值之间的协方差矩阵

                # avg norm
                W  = self.model.lin2.weight              # 分类器权重: [C, CHW] 或 [CHW, C]（取决于定义）
                M_norms = torch.norm(M_, dim=0)          # 类别均值的L2范数: [C]
                W_norms = torch.norm(W.T, dim=0)         # 分类器权重的L2范数: [C]

                self.graphs.norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
                self.graphs.norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item()) 

                # tr{Sw Sb^-1}
                Sw = Sw.cpu().detach().numpy()  
                Sb = Sb.cpu().detach().numpy()
                # 对Sb进行奇异值分解（SVD）并求伪逆
                eigvec, eigval, _ = svds(Sb, k=C-1)
                inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
                self.graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb))  # 类内协方差与类间协方差的比值，反映了分类边界的分离程度  值越小：类内紧致且类间分离良好

                # ||W^T - M_||
                normalized_M = M_ / torch.norm(M_,'fro')
                normalized_W = W.T / torch.norm(W.T,'fro')
                self.graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M)**2).item())  
                 # 归一化后的权重矩阵与类别均值矩阵之间的距离，反映两者之间的相似性
                # 离趋近于 0: 权重 W 与类别均值 M_ 高度对齐，分类器直接利用特征空间的几何结构

                # mutual coherence
                def coherence(V): 
                    G = V.T @ V
                    G += torch.ones((C,C),device= self.device) / (C-1)
                    G -= torch.diag(torch.diag(G))
                    return torch.norm(G,1).item() / (C*(C-1))

                self.graphs.cos_M.append(coherence(M_/M_norms))
                self.graphs.cos_W.append(coherence(W.T/W_norms))  
                #分别计算类别均值矩阵和权重矩阵的互相关性，衡量它们列向量之间的相似性
                # 值接近 0 → 均值向量接近正交（类别可分性强）；值接近 1 → 权重矩阵的列向量高度相关（类别间相关性强）
                # 计算特征空间的局部嵌入空间

            else:
            # 损失函数
                class_means = update_class_means(self.model, train_loader, num_classes=2,device=self.device)
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
                for batch_idx, (data, target) in enumerate(pbar, start=1):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    # 主任务损失
                    out, trigger_features  = self.model(data)
                    loss_cls = cls_criterion(out, target)

                    # 计算准确率
                    _, predicted  = torch.max(out, 1)
                    correct = (predicted == target).sum().item()
                    net_correct += correct

                    loss_collapse = collapse_criterion(trigger_features, class_means, target)
                    loss = loss_cls + 0.1 * loss_collapse  # 权重可调
                    
                    reg_loss = loss

                    reg_loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    sum_reg_loss += reg_loss.item()
                    pbar.set_postfix(loss=f'{train_loss/batch_idx:.4f}')
                
                net_correct /= len(train_loader.dataset)
                net_acc = 100.0 * net_correct
                avg_train_loss = train_loss / len(train_loader)
                print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")

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
                            out, trigger_features  = self.model(data)
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
                            out, trigger_features  = self.model(data)
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
                        out, trigger_features  = self.model(data)
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
                        out, trigger_features  = self.model(data)
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
                out, _  = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
            # 计算准确率
            finall_val_accuracy = 100.0 * correct / total

        print(f"Validation: val_loss={val_loss:.4f}, val_acc={finall_val_accuracy:.4f}")


        return self.graphs,finall_val_accuracy


    def fit(self, train_loader,val_loader,pre_loader, epochs, save_path):
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(epochs):

            N             = [0 for _ in range(C)]
            mean          = [0 for _ in range(C)] # 存储每个类别的特征均值
            Sw            = 0 #存储类内散度矩阵（within-class scatter matrix）
            loss          = 0
            net_correct   = 0 # 网络预测正确的样本数
            NCC_match_net = 0 # 最近邻分类器（NCC）与网络预测一致的样本数
            cls_criterion = nn.CrossEntropyLoss()
            collapse_criterion = NeuralCollapseLoss(epsilon=5.0)

            # 训练阶段
            self.model.train()
            train_loss = 0.0
            sum_reg_loss = 0.0

            if (epoch+1) % 5 == 0:
                class_means = update_class_means(self.model, pre_loader, num_classes=2,device=self.device)
                self.cur_epochs.append(epoch)
                for batch_idx, (data, target) in enumerate(pre_loader, start=1):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    # 主任务损失
                    out, trigger_features  = self.model(data)
                    loss_cls = cls_criterion(out, target)
                    # 水印损失
                    loss_collapse = collapse_criterion(trigger_features, class_means, target)
                    loss = loss_cls + 0.1 *loss_collapse  # 权重可调
                    total_loss = loss 
                    # 反向传播
                    total_loss.backward()
                    self.optimizer.step()
                    # 累计损失和准确率
                    train_loss += loss.item()
                    # 计算准确率
                    _, predicted  = torch.max(out, 1)
                    correct = (predicted == target).sum().item()
                    net_correct += correct
                    # 计算平均损失和准确率
                avg_train_loss = train_loss / len(pre_loader)
                net_acc = net_correct / len(pre_loader.dataset)
                net_acc = 100. * net_acc
                print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")
                self.graphs.reg_loss.append(avg_train_loss)
                self.graphs.loss.append(avg_train_loss)
                self.graphs.accuracy.append(net_acc)

                for computation in ['Mean','Cov']:  # Mean 每个类别的特征均值  Cov 计算类内协方差矩阵和其他指标
                    train_loss = 0.0
                    reg_loss = 0.0
                    net_correct = 0
                    NCC_match_net = 0
                    with torch.no_grad():
                        for batch_idx, (data, target) in enumerate(train_loader, start=1):
                            data, target = data.to(self.device), target.to(self.device)
                            # 主任务损失
                            out, trigger_features  = self.model(data)
                            loss_cls = cls_criterion(out, target)
                            # 水印损失
                            loss_collapse = collapse_criterion(trigger_features, class_means, target)
                            loss = loss_cls + 0.1 * loss_collapse  # 权重可调
                            train_loss += loss.item()

                            reg_loss = train_loss
                            # 计算准确率
                            _, predicted  = torch.max(out, 1)
                            correct = (predicted == target).sum().item()
                            net_correct += correct

                            for c in range(C):
                                # 获取当前 batch 中属于类别 c 的样本索引
                                idxs = (target == c).nonzero(as_tuple=True)[0]
                                if len(idxs) == 0: # If no class-c in this batch
                                    continue
                                h_c = trigger_features[idxs,:] # B CHW
                                if computation == 'Mean':
                                    # 累加当前 batch 中类别 c 的特征（用于后续计算均值）
                                    mean[c] += torch.sum(h_c, dim=0)  # 按样本维度求和 → shape: [CHW]
                                    N[c] += h_c.shape[0]  # 记录类别 c 的样本总数
                                elif computation == 'Cov':
                                    # update within-class cov
                                    z = h_c - mean[c].unsqueeze(0) # B CHW
                                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                                        z.unsqueeze(1))  # B 1 CHW
                                    # print(cov.shape)

                                    Sw += torch.sum(cov, dim=0)
                                    # during calculation of within-class covariance, calculate:
                                    # 1) network's accuracy
                                    net_pred = torch.argmax(out[idxs,:], dim=1)
                                    # net_correct += sum(net_pred==target[idxs]).item()
                                    # 2) agreement between prediction and nearest class center
                                    NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                                                for i in range(h_c.shape[0])])
                                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                                    NCC_match_net += sum(NCC_pred==net_pred).item()

                    if computation == 'Mean':
                        for c in range(C):
                            mean[c] /= N[c] # 均值归一化
                            M = torch.stack(mean).T # 将均值堆叠并转置
                        train_loss /= len(train_loader.dataset)
                    elif computation == 'Cov':
                        Sw /= sum(N) # 将类内协方差矩阵除以总样本数
      
                self.graphs.NCC_mismatch.append(1-NCC_match_net/sum(N))

                # global mean
                muG = torch.mean(M, dim=1, keepdim=True) # CHW 1  # 计算所有类别均值的平均值
            
                # between-class 计算类间协方差矩阵 Sb
                M_ = M - muG  
                Sb = torch.matmul(M_, M_.T) / C  # 计算类别均值之间的协方差矩阵

                # avg norm
                W  = self.model.lin2.weight              # 分类器权重: [C, CHW] 或 [CHW, C]（取决于定义）
                M_norms = torch.norm(M_, dim=0)          # 类别均值的L2范数: [C]
                W_norms = torch.norm(W.T, dim=0)         # 分类器权重的L2范数: [C]

                self.graphs.norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
                self.graphs.norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item()) 

                # tr{Sw Sb^-1}
                Sw = Sw.cpu().detach().numpy()  
                Sb = Sb.cpu().detach().numpy()
                # 对Sb进行奇异值分解（SVD）并求伪逆
                eigvec, eigval, _ = svds(Sb, k=C-1)
                inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
                self.graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb))  # 类内协方差与类间协方差的比值，反映了分类边界的分离程度  值越小：类内紧致且类间分离良好

                # ||W^T - M_||
                normalized_M = M_ / torch.norm(M_,'fro')
                normalized_W = W.T / torch.norm(W.T,'fro')
                self.graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M)**2).item())  
                 # 归一化后的权重矩阵与类别均值矩阵之间的距离，反映两者之间的相似性
                # 离趋近于 0: 权重 W 与类别均值 M_ 高度对齐，分类器直接利用特征空间的几何结构

                # mutual coherence
                def coherence(V): 
                    G = V.T @ V
                    G += torch.ones((C,C),device= self.device) / (C-1)
                    G -= torch.diag(torch.diag(G))
                    return torch.norm(G,1).item() / (C*(C-1))

                self.graphs.cos_M.append(coherence(M_/M_norms))
                self.graphs.cos_W.append(coherence(W.T/W_norms))  
                #分别计算类别均值矩阵和权重矩阵的互相关性，衡量它们列向量之间的相似性
                # 值接近 0 → 均值向量接近正交（类别可分性强）；值接近 1 → 权重矩阵的列向量高度相关（类别间相关性强）
                # 计算特征空间的局部嵌入空间

            else:
            # 损失函数
                class_means = update_class_means(self.model, train_loader, num_classes=2,device=self.device)
                for batch_idx, (data, target) in enumerate(train_loader, start=1):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    # 主任务损失
                    out, trigger_features  = self.model(data)
                    loss_cls = cls_criterion(out, target)

                    # 计算准确率
                    _, predicted  = torch.max(out, 1)
                    correct = (predicted == target).sum().item()
                    net_correct += correct

                    
                    loss_collapse = collapse_criterion(trigger_features, class_means, target)
                    loss = loss_cls + 0.1 * loss_collapse  # 权重可调
                    
                    # loss = loss_cls   # 权重可调
                    # l2_reg = torch.tensor(0.).to(self.device)
                    # for param in self.model.parameters():
                    #     l2_reg += torch.sum(param**2)
                    # num_params = sum(p.numel() for p in self.model.parameters())
                    reg_loss = loss

                    reg_loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    sum_reg_loss += reg_loss.item()
                
                
                net_correct /= len(train_loader.dataset)
                net_acc = 100.0 * net_correct
                avg_train_loss = train_loss / len(train_loader)
                print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, net_acc={net_acc:.4f}")

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
                            out, trigger_features  = self.model(data)
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

                    save_path_1 = os.path.join(save_path, f"best_model_{best_epoch}_{best_acc:.4f}_{epoch}_{val_accuracy:.4f}.ckpt")
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
                        out, trigger_features  = self.model(data)
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

                print(f"Validation:epoch={epoch}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")

                save_path_1 = os.path.join(save_path, f"{net_acc:.4f}_{epoch}_{val_accuracy:.4f}.ckpt")
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
                out, _  = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
            # 计算准确率
            finall_val_accuracy = 100.0 * correct / total

        print(f"Validation: val_loss={val_loss:.4f}, val_acc={finall_val_accuracy:.4f}")

        # loss = self.graphs.reg_loss.cpu().detach().numpy()
        # 绘制图表
        plt.figure(1)
        plt.semilogy(self.cur_epochs, self.graphs.reg_loss)
        plt.legend(['Loss + Weight Decay'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Loss')
        save_path_2 = os.path.join(save_path, 'Training Loss.png')
        plt.savefig(save_path_2)

        plt.figure(2)
        plt.plot(self.cur_epochs, 100*(1 - np.array(self.graphs.accuracy)))
        plt.xlabel('Epoch')
        plt.ylabel('Training Error (%)')
        plt.title('Training Error')
        save_path_3 = os.path.join(save_path, 'Training Error.png')
        plt.savefig(save_path_3)

        plt.figure(3)
        plt.semilogy(self.cur_epochs, self.graphs.Sw_invSb)
        plt.xlabel('Epoch')
        plt.ylabel('Tr{Sw Sb^-1}')
        plt.title('NC1: Activation Collapse')
        save_path_4 = os.path.join(save_path, 'NC1: Activation Collapse.png')
        plt.savefig(save_path_4)

        plt.figure(4)
        plt.plot(self.cur_epochs, self.graphs.norm_M_CoV)
        plt.plot(self.cur_epochs, self.graphs.norm_W_CoV)
        plt.legend(['Class Means','Classifiers'])
        plt.xlabel('Epoch')
        plt.ylabel('Std/Avg of Norms')
        plt.title('NC2: Equinorm')
        save_path_5 = os.path.join(save_path, 'NC2: Equinorm.png')
        plt.savefig(save_path_5)

        plt.plot(self.cur_epochs, self.graphs.cos_M)
        plt.plot(self.cur_epochs, self.graphs.cos_W)
        plt.legend(['Class Means','Classifiers'])
        plt.xlabel('Epoch')
        plt.ylabel('Avg|Cos + 1/(C-1)|')
        plt.title('NC2: Maximal Equiangularity')
        save_path_6 = os.path.join(save_path, 'NC2_Maximal_Equiangularity.png')
        plt.savefig(save_path_6)

        plt.figure(6)
        plt.plot(self.cur_epochs,self.graphs.W_M_dist)
        plt.xlabel('Epoch')
        plt.ylabel('||W^T - H||^2')
        plt.title('NC3: Self Duality')
        save_path_7 = os.path.join(save_path, 'NC3: Self Duality.png')
        plt.savefig(save_path_7)

        plt.figure(7)
        plt.plot(self.cur_epochs,self.graphs.NCC_mismatch)
        plt.xlabel('Epoch')
        plt.ylabel('Proportion Mismatch from NCC')
        plt.title('NC4: Convergence to NCC')
        save_path_8 = os.path.join(save_path, 'NC4:Convergence to NCC.png')
        plt.savefig(save_path_8)

        plt.show()
        return self.graphs,finall_val_accuracy

    def test(self,trig_dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(trig_dataloader, start=1):
                data, target = data.to(self.device), target.to(self.device)
                out, trigger_features  = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
        # 计算准确率
        test_accuracy = 100.0 * correct / total
        print(f"Test: test_acc={test_accuracy:.4f}")
        return test_accuracy
    
    def test_attack(self,trig_dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(trig_dataloader, start=1):
                data, target = data.to(self.device), target.to(self.device)
                out, trigger_features  = self.model(data)
                _, predicted = torch.max(out, 1)  # 获取预测类别
                total += target.size(0)  # 更新总样本数
                correct += (predicted == target).sum().item()  # 更新正确预测的样本数
        # 计算准确率
        test_accuracy = 100.0 * correct / total
        print(f"Test: test_acc={test_accuracy:.4f}")
        return test_accuracy

class ModelCheckpoint:
    def __init__(self, monitor='val_loss', save_path=None, save_top_k=1, mode='min'):
        self.monitor = monitor
        self.save_path = save_path
        self.save_top_k = save_top_k
        self.mode = mode
        self.best_values = []
        self.trainer = None
        
    def set_trainer(self, trainer):
        self.trainer = trainer
        
    def on_epoch_end(self, epoch, train_loss, val_loss, optimizer):
        current_value = val_loss if self.monitor == 'val_loss' else train_loss
        filename = os.path.join(self.save_path, f"best_model_{epoch}.pt")
        
        if self.mode == 'min':
            if current_value < self.trainer.best_loss:
                self.trainer.best_loss = current_value
                torch.save(self.trainer.model.state_dict(), filename)
                print(f"Model improved. Saved to {filename}")
        else:
            if current_value > self.trainer.best_loss:
                self.trainer.best_loss = current_value
                torch.save(self.trainer.model.state_dict(), filename)
                print(f"Model improved. Saved to {filename}")

class EarlyStopping:
    def __init__(self, monitor='val_loss', patience=5, min_delta=0.0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float('inf') if monitor == 'val_loss' else -float('inf')
        self.should_stop = False
        
    def on_epoch_end(self, epoch, train_loss, val_loss, optimizer):
        current_value = val_loss if self.monitor == 'val_loss' else train_loss
        
        if (self.monitor == 'val_loss' and current_value < self.best_value - self.min_delta) or \
           (self.monitor != 'val_loss' and current_value > self.best_value + self.min_delta):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"Early stopping counter: {self.counter}/{self.patience}")

class MultiplyLRScheduler:
    def __init__(self, multiply_factor, update_interval, start_epoch=0):
        self.multiply_factor = multiply_factor
        self.update_interval = update_interval
        self.start_epoch = start_epoch
        
    def on_epoch_end(self, epoch, train_loss, val_loss, optimizer):
        if epoch >= self.start_epoch and (epoch - self.start_epoch) % self.update_interval == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.multiply_factor
            print(f"Learning rate multiplied by {self.multiply_factor} at epoch {epoch}")

# # 示例使用方式
# if __name__ == "__main__":
    # # 模型定义
    # class SampleModel(nn.Module):
    #     def __init__(self, input_dim, num_classes):
    #         super().__init__()
    #         self.fc = nn.Linear(input_dim, num_classes)
            
    #     def forward(self, x):
    #         return self.fc(x)

    # # 自定义损失函数（包含分类损失和其他损失）
    # def combined_loss(outputs, batch):
    #     inputs, labels = batch[:2]
    #     cls_loss = F.cross_entropy(outputs, labels)
        
    #     # 示例：添加L2正则化
    #     l2_lambda = 0.001
    #     l2_reg = torch.tensor(0.)
    #     for param in model.parameters():
    #         l2_reg += torch.norm(param)
    #     total_loss = cls_loss + l2_lambda * l2_reg
        
    #     return total_loss

    # # 初始化组件
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # model = SampleModel(input_dim=1024, num_classes=16)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # # 创建回调
    # callbacks = [
    #     EarlyStopping(monitor='val_loss', patience=5),
    #     ModelCheckpoint(monitor='val_loss', save_path="./checkpoints"),
    #     MultiplyLRScheduler(multiply_factor=0.1, update_interval=5)
    # ]
    
    # # 初始化训练器
    # trainer = ClassifierTrainer(
    #     model=model,
    #     optimizer=optimizer,
    #     device=device,
    #     loss_fn=combined_loss,
    #     callbacks=callbacks
    # )
    
    # # 创建示例数据加载器
    # train_loader = DataLoader(torch.utils.data.TensorDataset(torch.randn(100, 1024), torch.randint(0, 16, (100,))), batch_size=32)
    # val_loader = DataLoader(torch.utils.data.TensorDataset(torch.randn(20, 1024), torch.randint(0, 16, (20,))), batch_size=32)
    
    # # 开始训练
    # trainer.fit(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     epochs=50,
    #     save_path="./checkpoints"
    # )