import pickle
import numpy as np
import torch
from typing import Any

def load_pkl_file(file_path: str) -> Any:
    """安全加载 .pkl 文件"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def print_data_structure(data: Any, indent: int = 0) -> None:
    """递归打印数据的结构和内容"""
    indent_str = ' ' * indent
    
    if isinstance(data, (list, tuple)):
        print(f"{indent_str}Type: {type(data).__name__}, Length: {len(data)}")
        for i, item in enumerate(data[:3]):  # 只打印前3项避免过长
            print(f"{indent_str}  [{i}]:")
            print_data_structure(item, indent + 4)
        if len(data) > 3:
            print(f"{indent_str}  ... (剩余 {len(data) - 3} 项)")

    elif isinstance(data, dict):
        print(f"{indent_str}Type: dict, Keys: {list(data.keys())}")
        for k, v in list(data.items())[:3]:  # 只打印前3个键值对
            print(f"{indent_str}  [{repr(k)}]:")
            print_data_structure(v, indent + 4)
        if len(data) > 3:
            print(f"{indent_str}  ... (剩余 {len(data) - 3} 对)")

    elif isinstance(data, np.ndarray):
        print(f"{indent_str}Type: numpy.ndarray, Shape: {data.shape}, Dtype: {data.dtype}")
        print(f"{indent_str}  Sample values:\n{indent_str}    {data.ravel()[:3]}...")

    elif isinstance(data, torch.Tensor):
        print(f"{indent_str}Type: torch.Tensor, Shape: {data.shape}, Dtype: {data.dtype}, Device: {data.device}")
        print(f"{indent_str}  Sample values:\n{indent_str}    {data.flatten()[:3].cpu().numpy()}...")

    else:
        print(f"{indent_str}Type: {type(data).__name__}, Value: {repr(data)}")

if __name__ == '__main__':
    # 输入 .pkl 文件路径
    pkl_path = "/home/qty/code/trigger_data/TSCeption/fold-0/right.pkl"
    
    try:
        data = load_pkl_file(pkl_path)
        print("\n" + "=" * 50)
        print(f"文件内容结构:")
        print("=" * 50)
        print_data_structure(data)
    except FileNotFoundError:
        print(f"错误: 文件 {pkl_path} 不存在！")
    except Exception as e:
        print(f"加载失败: {str(e)}")