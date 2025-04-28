import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
# 软标签攻击
# def soft_label_attack(target_model, surrogate_model, loader, epochs, lr,devices):
#     criterion = nn.KLDivLoss(reduction='batchmean')
#     optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
#     target_model.to(devices)
#     surrogate_model.to(devices)
#     surrogate_model.eval()
#     for epoch in range(epochs):
#         total_loss = 0
#         correct = 0
#         total = 0
#         surrogate_model.train()
#         for inputs, _ , _ in loader:
#             inputs = inputs.to(devices)
#             optimizer.zero_grad()

#             # 获取目标模型的输出（软标签）
#             with torch.no_grad():
#                 target_outputs = target_model(inputs)
#             # 获取替代模型的输出
#             surrogate_outputs = surrogate_model(inputs)
#             # 计算KL散度损失
#             # loss = criterion(torch.log(surrogate_outputs), target_outputs)
#             loss = criterion(torch.log(surrogate_outputs), target_outputs)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             _, surrogate_preds = torch.max(surrogate_outputs,1)
#             _, target_preds = torch.max(target_outputs,1)
#             correct += (surrogate_preds == target_preds).sum().item()
#             total += inputs.size(0)
        
#         avg_loss = total_loss / len(loader)
#         acc = 100 * correct /total
#         print(f"Soft-label Attack - Epoch {epoch+1}, Loss: {total_loss}")
#         print(f"accuracy target :{acc:.2f}%")
#     return surrogate_model

import torch.nn.functional as F
def soft_label_attack(target_model, surrogate_model, loader,pre_loader,val_loader, epochs, lr, result_model_path,devices):
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
    target_model.to(devices)
    surrogate_model.to(devices)
    
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        surrogate_model.train()
        if (epoch+1)  != -1:
            for inputs, _ in loader:
                inputs = inputs.to(devices)
                optimizer.zero_grad()

                # 获取目标模型的输出（归一化为概率分布）
                with torch.no_grad():
                    target_outputs, _ = target_model(inputs)
                    target_outputs = F.softmax(target_outputs, dim=1)  # 确保归一化

                # 获取替代模型的输出（使用 log_softmax）
                surrogate_outputs, _ = surrogate_model(inputs)
                surrogate_outputs = F.log_softmax(surrogate_outputs, dim=1)  # 数值稳定

                # 计算KL散度损失
                loss = criterion(surrogate_outputs, target_outputs)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), max_norm=1.0)
                optimizer.step()

                # 计算指标
                total_loss += loss.item()
                _, surrogate_preds = torch.max(surrogate_outputs, 1)
                _, target_preds = torch.max(target_outputs, 1)
                correct += (surrogate_preds == target_preds).sum().item()
                total += inputs.size(0)
            avg_loss = total_loss / len(loader)
            acc = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
        else:
            for inputs, _ in pre_loader:
                inputs = inputs.to(devices)
                optimizer.zero_grad()

                # 获取目标模型的输出（归一化为概率分布）
                with torch.no_grad():
                    target_outputs, _ = target_model(inputs)
                    target_outputs = F.softmax(target_outputs, dim=1)  # 确保归一化

                # 获取替代模型的输出（使用 log_softmax）
                surrogate_outputs, _ = surrogate_model(inputs)
                surrogate_outputs = F.log_softmax(surrogate_outputs, dim=1)  # 数值稳定

                # 计算KL散度损失
                loss = criterion(surrogate_outputs, target_outputs)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), max_norm=1.0)
                optimizer.step()

                # 计算指标
                total_loss += loss.item()
                _, surrogate_preds = torch.max(surrogate_outputs, 1)
                _, target_preds = torch.max(target_outputs, 1)
                correct += (surrogate_preds == target_preds).sum().item()
                total += inputs.size(0)
            avg_loss = total_loss / len(loader)
            acc = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")




        if (epoch + 1) % 5 == 0:
            # Validate on pre_loader
            pre_acc = evaluate(surrogate_model, pre_loader, devices)
            print(f"Pre-validation Accuracy: {pre_acc:.2f}%")
            
            # Validate on val_loader
            val_acc = evaluate(surrogate_model, val_loader, devices)
            print(f"Validation Accuracy: {val_acc:.2f}%")
            
            # Update best validation accuracy and save model if improved
            if pre_acc > best_val_acc:
                best_val_acc = pre_acc
                best_epoch = epoch + 1
                save_path = os.path.join(result_model_path, f'{best_epoch}_{pre_acc:.4f}_{val_acc:.4f}_model.pth')
                torch.save(surrogate_model.state_dict(), save_path)
                print(f"New best model saved at epoch {epoch+1} with val acc {val_acc:.2f}%")

        if (epoch + 1) % 10 == 0:

            torch.save(surrogate_model.state_dict(), 
                    os.path.join(result_model_path, f'model_epoch_{epoch+1}.pth'))
            print(f"Model saved at epoch {epoch+1}")
    # if (epoch+1) % 10 == 0:
        # continue

    return surrogate_model, best_val_acc, val_acc


# 硬标签攻击
def hard_label_attack(target_model, surrogate_model, loader, perloder,valloader,epochs, lr, result_model_path,devices):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
    target_model.to(devices)
    surrogate_model.to(devices)
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        surrogate_model.train()
        
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, _  in pbar:
            inputs = inputs.to(devices)
            optimizer.zero_grad()
            # 获取目标模型的输出（硬标签）
            with torch.no_grad():
                target_outputs,_ = target_model(inputs)
                _, labels = torch.max(target_outputs, 1)  # 获取预测的类别标签

            # 获取替代模型的输出
            surrogate_outputs,_ = surrogate_model(inputs)
            # 计算交叉熵损失
            loss = criterion(surrogate_outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, surrogate_preds = torch.max(surrogate_outputs, 1)
            correct += (surrogate_preds == labels).sum().item()
            total += inputs.size(0)
            pbar.set_postfix({'Loss': loss.item(), 'Acc': f'{100 * correct/total:.2f}%'})

        avg_loss = total_loss / len(loader)
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
            # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Validate on pre_loader
            pre_acc = evaluate(surrogate_model, perloder, devices)
            print(f"Pre-validation Accuracy: {pre_acc:.2f}%")
            
            # Validate on val_loader
            val_acc = evaluate(surrogate_model, valloader, devices)
            print(f"Validation Accuracy: {val_acc:.2f}%")
            
            # Update best validation accuracy and save model if improved
            if pre_acc > best_val_acc:
                best_val_acc = pre_acc
                best_epoch = epoch + 1
                save_path = os.path.join(result_model_path, f'{best_epoch}_{pre_acc:.4f}_{val_acc:.4f}_model.pth')
                torch.save(surrogate_model.state_dict(), save_path)
                print(f"New best model saved at epoch {epoch+1} with val acc {val_acc:.2f}%")

        if (epoch + 1) % 10 == 0:
            torch.save(surrogate_model.state_dict(), 
                    os.path.join(result_model_path, f'model_epoch_{epoch+1}.pth'))
            print(f"Model saved at epoch {epoch+1}")
    
    return surrogate_model , best_val_acc,val_acc

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def regularization_with_ground_truth(target_model, surrogate_model, train_loader,
                                     pre_loader, val_loader, gamma,epochs, lr,result_model_path, devices, patience=3):
    """
    基于真实标签的正则化攻击完整实现
    
    参数:
        target_model: 目标模型(教师模型)
        surrogate_model: 替代模型(学生模型)
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        gamma: KL损失和CE损失的权重平衡参数(0-1)
        temperature: 温度缩放参数
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备
        patience: 早停耐心值
    """
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.5)
    
    target_model.to(devices).eval()
    surrogate_model.to(devices)
    best_pre_acc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    
    for epoch in range(epochs):
        surrogate_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, true_labels in pbar:
            inputs = inputs.to(devices)
            true_labels = true_labels.to(devices)

            optimizer.zero_grad()
            
            # 获取目标模型的软化输出
            with torch.no_grad():
                target_logits, _ = target_model(inputs)
                target_probs = torch.softmax(target_logits / 2, dim=1)
            
            # 获取替代模型的输出
            surrogate_logits, _ = surrogate_model(inputs)
            surrogate_probs = torch.log_softmax(surrogate_logits / 2, dim=1)
            
            # 计算损失
            loss_kl = criterion_kl(surrogate_probs, target_probs)
            loss_ce = criterion_ce(surrogate_logits, true_labels)
            loss = gamma * loss_kl + (1 - gamma) * loss_ce
            
            loss.backward()
            optimizer.step()
            
            # 计算统计量
            total_loss += loss.item()
            _, predicted = torch.max(surrogate_logits, 1)
            correct += (predicted == true_labels).sum().item()
            total += inputs.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'KL': f'{loss_kl.item():.4f}',
                'CE': f'{loss_ce.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
                # 训练统计
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%')
        
        if (epoch + 1) % 5 == 0:
            # Validate on pre_loader
            pre_acc = evaluate(surrogate_model, pre_loader, devices)
            print(f"Pre-validation Accuracy: {pre_acc:.2f}%")
            
            # Validate on val_loader
            val_acc = evaluate(surrogate_model, val_loader, devices)
            print(f"Validation Accuracy: {val_acc:.2f}%")
            
            # Update best validation accuracy and save model if improved
            if pre_acc > best_val_acc:
                best_val_acc = pre_acc
                best_epoch = epoch + 1
                save_path = os.path.join(result_model_path, f'{best_epoch}_{pre_acc:.4f}_{val_acc:.4f}_model.pth')
                torch.save(surrogate_model.state_dict(), save_path)
                print(f"New best model saved at epoch {epoch+1} with val acc {val_acc:.2f}%")
            
            # 更新学习率
            scheduler.step(val_acc)

        if (epoch + 1) % 10 == 0:
            torch.save(surrogate_model.state_dict(), 
                    os.path.join(result_model_path, f'model_epoch_{epoch+1}.pth'))
            print(f"Model saved at epoch {epoch+1}")
    return surrogate_model, best_val_acc, val_acc

def evaluate(model, data_loader, device):
    """评估模型在验证集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs,_ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += inputs.size(0)
    
    return 100 * correct / total

# 检测触发集
# def evaluate_test(surrogate_model ,trigger_data):
#     surrogate_model.eval()
# #     for 

# def evaluate_test(surrogate_model ,trigger_data):
#     results = dict()
#     for eval_dimension in evaluation_metrics:
#         if eval_dimension.endswith("watermark"):
#             verifier = Verifier[eval_dimension.split("_")[0].upper()]

#             tri_path ="/home/qty/project2/watermarking-eeg-models-main/tiggerest/CCNN/wrong_predictions_199_2.pkl"
#             trig_set = TriggerSet(
#                 tri_path,
#                 architecture,
#             )

#             trig_set_loader = DataLoader(trig_set, batch_size=batch_size)

#             results[eval_dimension] = {
#                 "null_set": trainer.test(
#                     trig_set_loader, enable_model_summary=True
#                 ),
#             }

#         elif eval_dimension == "eeg":
#             from torch.utils.data import DataLoader
#             test_loader = DataLoader(test_dataset, batch_size=batch_size)
#             # test_loader = DataLoader(train_dataset, batch_size=batch_size)
#             results[eval_dimension] = trainer.test(
#                 test_loader, enable_model_summary=True
#             )
#     return results
