print("hello")
import os
import numpy as np
import mne
import scipy.io as sio


def inputmat(fp):
    """load .mat file and return m as a dict"""
    mat = sio.loadmat(fp, squeeze_me=True)
    m = {}  # create a dict

    # Numpy array of size channel_num * points.
    m['data'] = mat['cnt'].T  # 数据
    m['freq'] = mat['nfo']['fs'][True][0]  # Sampling frequency

    # channel names are necessary information for creating a rawArray.
    m['ch_names'] = mat['nfo']['clab'][True][0]

    # Position of channels
    m['electrode_x'] = mat['nfo']['xpos'][True][0]
    m['electrode_y'] = mat['nfo']['ypos'][True][0]

    # find trials and its data
    m['cue'] = mat['mrk']['pos'][True][0]  # time of cue
    # m['labels'] = np.nan_to_num(mat['mrk']['y'][True][0])
    m['labels'] = np.nan_to_num(mat['mrk']['y'][True][0]).astype(int)  # convert NaN to 0
    # m['n_trials'] = np.where(m['labels'] == 0)
    m['n_trials'] = np.where(m['labels'] == 0)[0][0]  # Number of the total useful trials
    return m


def creatEventsArray(fp):
    """Create events array. The second column default to zero."""
    m = inputmat(fp)
    events = np.zeros((m['n_trials'], 3), int)
    events[:, 0] = m['cue'][:m['n_trials']]  # The first column is the sample number of the event.
    events[:, 2] = m['labels'][:m['n_trials']]  # The third column is the new event value.
    return events, m['labels']


def creatRawArray(fp):
    """Create a mne.io.RawArray object, data: array, shape (n_channels, n_times)"""
    m = inputmat(fp)
    ch_names = m['ch_names'].tolist()
    info = mne.create_info(ch_names, m['freq'], 'eeg')  # Create info for raw
    raw = mne.io.RawArray(m['data'], info, first_samp=0, copy='auto', verbose=None)
    return raw

raw_data = creatRawArray("./data_set_IVa_al.mat")
m = inputmat("./data_set_IVa_al.mat")

# 时域特征分析和可视化
raw_data.plot()  # 绘制原始数据的波形

# 频域特征分析和可视化
raw_data.plot_psd()  # 绘制频谱密度图像

import matplotlib.pyplot as plt
plt.scatter(m['electrode_x'], m['electrode_y'])
plt.show()
# 空域特征分析和可视化
# montage = mne.channels.make_standard_montage('standard_1005')  # 根据你的电极位置创建电极阵列
# raw_data.set_montage(montage)
# raw_data.plot_sensors()  # 绘制电极位置

# 可以根据需要进一步处理和分析数据，这只是一个基本示例
