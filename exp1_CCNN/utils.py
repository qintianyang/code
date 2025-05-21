import math
import glob
import random
import numpy as np
from pathlib import Path
from functools import reduce
from torch import manual_seed, cuda
from scipy.interpolate import interp1d
from scipy.stats import norm
import pandas as pd

# Data Encoding Utilities
def BinariesToCategory(y):
    return {"y": reduce(lambda acc, num: acc * 2 + num, y, 0)}


# Numerical Methods
def interpolate(xs, ys):
    return interp1d(xs, ys, kind="quadratic", fill_value="extrapolate")


def is_numeric(value):
    try:
        return not math.isnan(float(value))
    except:
        return False


# File Handling Utilities
def list_json_files(dir):
    path = (Path(dir) / "./**/*.json").resolve()
    return glob.glob(str(path), recursive=True)


# Random Seed Configuration
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    if cuda.is_available():
        cuda.manual_seed(seed)


# Dictionary Utilities
def are_keys_numeric(dictionary):
    return isinstance(dictionary, dict) and all(
        is_numeric(key) for key in dictionary.keys()
    )


def are_values_numeric(dictionary):
    return isinstance(dictionary, dict) and all(
        is_numeric(value) for value in dictionary.values()
    )


def add_to_dict(dictionary, keys, value):
    last_key = keys.pop()
    for key in keys:
        if key not in dictionary:
            dictionary[key] = dictionary.get(key, {})
        dictionary = dictionary[key]
    dictionary[last_key] = value


def add_key_at_depth(origin, dest, key, depth=100):
    if depth == 0 or not isinstance(origin, dict):
        dest[key] = origin
        return

    for k, v in origin.items():
        dest[k] = dest.get(k, {})
        add_key_at_depth(v, dest[k], key, depth - 1)


# Visualization Utilities
def get_color(index):
    return f"color({index})"


def title(title):
    return title.replace("_", " ").title()


def get_result_panel(key, value):
    return f"{title(key)}: [reset]{(value * 100):.2f}%[/reset]"


def convert_dict_to_tree(dictionary, tree, depth):
    color = get_color(depth)
    if not isinstance(dictionary, dict):
        return tree.add(dictionary, style="reset")

    for key, value in dictionary.items():
        if key.endswith(".json"):
            convert_dict_to_tree(
                value,
                tree,
                depth,
            )
        elif is_numeric(value):
            tree.add(
                get_result_panel(key, value), guide_style=color, style=f"bold {color}"
            )
        else:
            convert_dict_to_tree(
                value,
                tree.add(title(key), guide_style=color, style=f"bold {color}"),
                depth + 1,
            )



from scipy.stats import binomtest
def binomial_test(observed_accuracy, random_guess, n_samples, alternative='greater'):
    """
    检验代理模型准确率是否显著高于随机猜测。
    
    参数:
        observed_accuracy (float): 观测到的准确率（如 0.95 表示 95%）。
        random_guess (float): 随机猜测准确率（如 CIFAR-10 是 0.1）。
        n_samples (int): 触发集的样本数量。
        alternative (str): 检验类型，'greater'（默认）或 'two-sided'。
    
    返回:
        p_value (float): p 值。
    """
    n_success = int(observed_accuracy * n_samples)  # 成功分类的样本数
    test_result = binomtest(
        k=n_success,
        n=n_samples,
        p=random_guess,
        alternative=alternative
    )
    return test_result.pvalue


def z_test(observed_accuracy, random_guess, n_samples):
    """
    Z 检验（适用于大样本，n > 30）。
    """
    p_hat = observed_accuracy
    p_null = random_guess
    se = (p_null * (1 - p_null) / n_samples) ** 0.5  # 标准误
    z_score = (p_hat - p_null) / se
    p_value = 1 - norm.cdf(z_score)  # 单侧检验
    return p_value


def save_graphs_to_excel(graphs_obj, filename,append=True):
    """将 graphs 对象的数据导出到 Excel 文件
    
    Args:
        graphs_obj: graphs 类的实例
        filename: 输出的 Excel 文件名（如 "results.xlsx"）
    """
    # 提取所有数据
    data = {
        'accuracy': graphs_obj.accuracy,
        'loss': graphs_obj.loss,
        'reg_loss': graphs_obj.reg_loss,
        'Sw_invSb': graphs_obj.Sw_invSb,
        'norm_M_CoV': graphs_obj.norm_M_CoV,
        'norm_W_CoV': graphs_obj.norm_W_CoV,
        'cos_M': graphs_obj.cos_M,
        'cos_W': graphs_obj.cos_W,
        'W_M_dist': graphs_obj.W_M_dist,
        'NCC_mismatch': graphs_obj.NCC_mismatch,
        'MSE_wd_features': graphs_obj.MSE_wd_features,
        'LNC1': graphs_obj.LNC1,
        'LNC23': graphs_obj.LNC23,
        'Lperp': graphs_obj.Lperp,
    }

    # 确保所有列表长度一致（填充 None）
    max_length = max(len(v) for v in data.values())
    for key in data:
        if len(data[key]) < max_length:
            data[key].extend([None] * (max_length - len(data[key])))

    # 转换为 DataFrame 并保存
    df = pd.DataFrame(data)
    df.to_csv(filename, mode='a' if append else 'w', header=not append, index=False)
    print(f"数据已保存到 {filename}")
