import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from models import get_model, load_model, get_ckpt_file
from dataset import get_dataset

def extract_features(model, dataloader, device):
    """提取所有样本的特征和标签"""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            data = data.type(torch.float)
            target = target.type(torch.long)

            _, feature = model(data)  # 假设模型返回 (output, feature)
            features.append(feature.cpu())
            labels.append(target.cpu())
    return torch.cat(features), torch.cat(labels)

working_dir = f"/home/qty/project2/watermarking-eeg-models-main/results/CCNN"
data_path ="/home/qty/project2/watermarking-eeg-models-main/data_preprocessed_python"
dataset = get_dataset("CCNN", working_dir, data_path)   #  三个标签

from triggerset import ModifiedDataset

dataset = ModifiedDataset(dataset)
from torch.utils.data import DataLoader 
train_loader = DataLoader(dataset, shuffle=True, batch_size=64)
model = get_model("CCNN")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 提取特征
model.to(device)
load_path = f'/home/qty/project2/watermarking-eeg-models-main/model_load_path/CCNN/fold-8'
model = load_model(model, get_ckpt_file(load_path))
all_features, all_labels = extract_features(model, train_loader, device)

# 降维 (选择PCA或T-SNE)
def reduce_dim(features, method='pca', n_components=2):
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=30, random_state=42)
    return reducer.fit_transform(features.numpy())

# reduced_features = reduce_dim(all_features, method='pca')  # 或 'tsne'


def plot_feature_distribution(features, labels, class_names=None):
    plt.figure(figsize=(10, 8))
    unique_labels = torch.unique(labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))  # 支持更多类别
    
    for i, label in enumerate(unique_labels):
        if  label != 1 and label !=4 and label !=8 :
            continue
        idx = (labels == label).numpy()
        if idx.sum() == 0:  # 跳过无样本的类别
            continue
        plt.scatter(
            features[idx, 0], 
            features[idx, 1], 
            color=colors(i), 
            label=f'Class {label}' if class_names is None else class_names[label],
            alpha=0.6
        )
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Feature Distribution by Class')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('/home/qty/project2/watermarking-eeg-models-main/feature_distribution9.png')

# 示例调用
# class_names = ['Class0', 'Class1', 'Class2']  # 替换为实际类别名
# plot_feature_distribution(reduced_features, all_labels)


def plot_single_sample(features, labels, sample_idx, class_names=None):
    plt.figure(figsize=(10, 8))
    unique_labels = torch.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    # 绘制所有类别
    for i, label in enumerate(unique_labels):
        idx = (labels == label).numpy()
        plt.scatter(features[idx, 0], features[idx, 1], 
                    color=colors(i), 
                    label=f'Class {label}' if class_names is None else class_names[label],
                    alpha=0.3)  # 调低透明度突出目标样本
    
    # 标记目标样本
    target_feature = features[sample_idx]
    target_label = labels[sample_idx].item()
    plt.scatter(target_feature[0], target_feature[1], 
                color='black', marker='*', s=300,
                label=f'Sample (True: {class_names[target_label]})')
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'Sample Position in Feature Space')
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例调用：可视化第100个样本
# sample_idx = 100
# plot_single_sample(reduced_features, all_labels, sample_idx, class_names)

from mpl_toolkits.mplot3d import Axes3D

def plot_3d_features(features, labels):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = torch.unique(labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))  # 支持更多类别
    
    for i, label in enumerate(unique_labels):
        if  label != 1 and label !=4 and label !=8 and label !=12 and label !=3:
            continue
        idx = (labels == label).numpy()

        ax.scatter(features[idx, 0], features[idx, 1], features[idx, 2],
                   color=colors(i), label=f'Class {label}', alpha=0.6)
    
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    plt.legend()
    plt.show()
    plt.savefig('/home/qty/project2/watermarking-eeg-models-main/feature_3d_all.png')

# 使用3D降维
# reduced_features_3d = reduce_dim(all_features, method='pca', n_components=3)
# plot_3d_features(reduced_features_3d, all_labels)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary_2d(model, features, labels, class_names=None, resolution=0.5):
    """
    绘制2D特征空间的决策边界
    :param model: 训练好的分类模型（需有 predict 方法）
    :param features: 2D 特征数组 [n_samples, 2]
    :param labels: 类别标签 [n_samples]
    :param resolution: 网格细粒度
    """
    # 转换为 numpy
    features = features.cpu().numpy() if hasattr(features, 'cpu') else np.array(features)
    labels = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)
    
    # 创建网格
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # 预测网格点类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(labels))))
    
    # 绘制决策区域
    plt.contourf(xx, yy, Z, alpha=0.3, levels=len(colors), colors=colors)
    
    # 绘制样本点
    for i, label in enumerate(np.unique(labels)):
        idx = (labels == label)
        plt.scatter(features[idx, 0], features[idx, 1],
                    color=colors[i],
                    label=f'Class {label}' if class_names is None else class_names[int(label)],
                    edgecolor='k', alpha=0.8)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundaries')
    plt.legend()
    plt.show()
    plt.savefig('/home/qty/project2/watermarking-eeg-models-main/decision_boundary_2d.png')

# 调用示例
features_2d = reduce_dim(all_features, method='pca')
plot_decision_boundary_2d(model, features_2d, all_labels)