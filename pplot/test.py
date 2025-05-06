import numpy as np
import matplotlib.pyplot as plt
import mne
from torcheeg.datasets import BCICIV2aDataset,DEAPDataset
from torcheeg import transforms
from torcheeg.datasets.constants import (
    DEAP_CHANNEL_LOCATION_DICT,
    DEAP_CHANNEL_LIST,
    DEAP_LOCATION_LIST,
)

EMOTIONS = ["valence"]
TSCEPTION_CHANNEL_LIST = [
    "FP1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
    "FP2",
    "AF4",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "C4",
    "T8",
    "CP6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
]
def remove_base_from_eeg(eeg, baseline):
    return {"eeg": eeg - baseline, "baseline": baseline}
from functools import reduce
def BinariesToCategory(y):
    return {"y": reduce(lambda acc, num: acc * 2 + num, y, 0)}


label_transform = transforms.Compose(
    [
        transforms.Select(EMOTIONS),
        transforms.Binary(5.0),
        BinariesToCategory,
    ]
)
dataset = DEAPDataset(
                io_path=f"/home/qty/code/work/TSCeption/dataset",
                root_path="/home/qty/project2/watermarking-eeg-models-main/data_preprocessed_python",
                chunk_size=512,
                num_baseline=1,
                baseline_chunk_size=384,
                offline_transform=transforms.Compose(
                    [
                        transforms.PickElectrode(
                            transforms.PickElectrode.to_index_list(
                                TSCEPTION_CHANNEL_LIST,
                                DEAP_CHANNEL_LIST,
                            )
                        ),
                        transforms.To2d(),
                    ]
                ),
                online_transform=transforms.ToTensor(),
                label_transform=label_transform,
                num_worker=4,
                verbose=True,
            )

print(dataset[2][0].type)

# 1. 生成随机EEG数据
n_channels = 28
sfreq = 512  # 采样率1000Hz
duration = 1# 5秒
n_samples = int(sfreq * duration)
import torch
data = dataset[0][0]
label = dataset[0][1]
print(label)
data = torch.squeeze(data)
data = data.numpy()
data = data * 1e-6
# print(data)
# 2. 定义通道名称和类型
ch_names = TSCEPTION_CHANNEL_LIST
ch_types = ['eeg'] * n_channels

# 3. 创建Raw对象并设置导联
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(data, info)
raw.set_montage('standard_1020')

montage = mne.channels.make_standard_montage("standard_1020")


# 4. 计算PSD（8-30Hz）
psd = raw.compute_psd(method='welch', fmin=8, fmax=30)
psd_mean = 10 * np.log10(psd.get_data().mean(axis=1))  # 转换为dB
print(psd_mean)
# 5. 绘制热力图
fig, ax = plt.subplots(figsize=(8, 6))
im, _ = mne.viz.plot_topomap(
    psd_mean,
    pos=raw.info,
    cmap='RdBu_r',
    vlim=(-120, -115),
    contours=0,
    sensors=True,
    axes=ax,
    show=False
)

# 添加colorbar
plt.colorbar(im, label="Power (dB)")
plt.title("Topomap of Simulated EEG (8-30Hz Power)", fontsize=12)
# plt.show()
plt.savefig("./1112.jpg")