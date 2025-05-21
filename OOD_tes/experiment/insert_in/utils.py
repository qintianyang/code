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

recall_level_default = 0.95


import sklearn.metrics as sk


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])



def show_performance(pos, neg, method_name='Ours', recall_level=recall_level_default):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))