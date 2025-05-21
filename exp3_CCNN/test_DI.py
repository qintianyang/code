import torch
import torch.nn.functional as F
from tqdm import tqdm

def compute_proxy_margin(model, dataloader, num_classes, device="cuda"):
    """
    计算数据集中所有样本的代理裕度（到每个类决策边界的距离）。
    返回: 裕度矩阵 [num_samples, num_classes]
    """
    margins = []
    model.eval()
    with torch.no_grad():
        for X, _ in tqdm(dataloader):
            X = X.to(device)
            logits = model(X)
            probas = F.softmax(logits, dim=1)
            # 计算到每个类决策边界的距离（代理裕度）
            # 方法1：直接用预测概率的负对数（越小表示距离越远）
            margin = -torch.log(probas + 1e-10)  # [batch_size, num_classes]
            margins.append(margin.cpu())
    return torch.cat(margins, dim=0)  # [num_samples, num_classes]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_binary_classifier(margin_db, margin_dtest):
    """
    用私有数据（D_b）和公开数据（D_test）的裕度嵌入训练分类器。
    返回: 训练好的分类器和测试准确率。
    """
    # 合并数据并生成标签（1: D_b, 0: D_test）
    X = torch.cat([margin_db, margin_dtest], dim=0).numpy()
    y = torch.cat([torch.ones(len(margin_db)), torch.zeros(len(margin_dtest))]).numpy()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练逻辑回归分类器
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Classifier Accuracy: {accuracy:.4f}")
    return clf

from scipy.stats import ttest_ind

def verify_model_ownership(suspect_model, clf, db_loader, dtest_loader, num_classes, device="cuda"):
    """
    验证可疑模型是否窃取了私有数据 D_b。
    步骤：
      1. 计算可疑模型对 D_b 和 D_test 的裕度嵌入。
      2. 用分类器预测两类数据的置信度。
      3. T检验判断置信度差异是否显著。
    """
    # 计算可疑模型的裕度嵌入
    margin_db_suspect = compute_proxy_margin(suspect_model, db_loader, num_classes, device)
    margin_dtest_suspect = compute_proxy_margin(suspect_model, dtest_loader, num_classes, device)
    
    # 用分类器预测置信度
    proba_db = clf.predict_proba(margin_db_suspect.numpy())[:, 1]  # D_b 的置信度
    proba_dtest = clf.predict_proba(margin_dtest_suspect.numpy())[:, 1]  # D_test 的置信度
    
    # T检验
    t_stat, p_value = ttest_ind(proba_db, proba_dtest, alternative="less")
    print(f"T-test p-value: {p_value:.6f}")
    if p_value < 0.01:  # 显著性水平 1%
        print("✅ Suspect model is likely a copy of the source model!")
    else:
        print("❌ No evidence of model stealing.")