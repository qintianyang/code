import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_t_sne_comparison(img_features_private, eeg_features_private, img_features_common,
                             eeg_features_common, title_prefix):
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
    all_features = np.concatenate([img_features_private, eeg_features_private, 
                                  img_features_common, eeg_features_common], axis=0)
    all_features = scaler.fit_transform(all_features)

    # 将标准化后的特征重新分割回原始组
    n_img_p = img_features_private.shape[0]
    n_eeg_p = eeg_features_private.shape[0]
    n_img_c = img_features_common.shape[0]
    
    img_features_private = all_features[:n_img_p]
    eeg_features_private = all_features[n_img_p : n_img_p + n_eeg_p]
    img_features_common = all_features[n_img_p + n_eeg_p : n_img_p + n_eeg_p + n_img_c]
    eeg_features_common = all_features[n_img_p + n_eeg_p + n_img_c:]

    # 合并所有特征并创建标签
    combined_features = np.vstack([img_features_private, eeg_features_private, 
                                 img_features_common, eeg_features_common])
    labels = np.array(['img_p'] * n_img_p + ['eeg_p'] * n_eeg_p + 
                      ['img_c'] * n_img_c + ['eeg_c'] * len(eeg_features_common))

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=2025, perplexity=30, n_iter=300)
    embedded_features = tsne.fit_transform(combined_features)

    # 可视化
    plt.figure(figsize=(10, 8))
    colors = {'img_p': 'blue', 'eeg_p': 'red', 'img_c': 'green', 'eeg_c': 'orange'}
    
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], 
                    label=label, c=colors[label], alpha=0.6)
    
    plt.title(f'{title_prefix} - t-SNE Comparison (Private vs Common Features)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 示例：生成随机数据（实际使用时替换为真实数据）
    np.random.seed(2025)
    n_samples = 100
    n_features = 256  # 假设特征维度为 256
    
    # 生成模拟数据
    img_features_private = np.random.randn(n_samples, n_features) * 0.5 + 1.0  # 私有图像特征
    eeg_features_private = np.random.randn(n_samples, n_features) * 0.5 + 0.5  # 私有 EEG 特征
    img_features_common = np.random.randn(n_samples, n_features) * 0.3 + 0.8    # 共有图像特征
    eeg_features_common = np.random.randn(n_samples, n_features) * 0.3 + 0.7    # 共有 EEG 特征
    
    # 调用可视化函数（训练集）
    visualize_t_sne_comparison(img_features_private, eeg_features_private,
                             img_features_common, eeg_features_common,
                             title_prefix="Train Set")
    
    # 如果是测试集，可以再调用一次（需替换为测试数据）
    # visualize_t_sne_comparison(test_img_p, test_eeg_p, test_img_c, test_eeg_c, "Test Set")

if __name__ == "__main__":
    main()