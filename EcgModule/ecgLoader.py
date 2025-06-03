import os

import bioread
import numpy as np
import pandas as pd
from scipy import signal


class EcgLoader:
    def __init__(self, ecg_path, timeStamp_path, Label_path, target_fps,  trial_target_len):
        self.ecg_path = ecg_path
        self.timeStamp_path = timeStamp_path
        self.Label_path = Label_path  # 标签路径
        self.target_fps = target_fps  # 目标采样率
        # self.static_target_len = static_target_len  # 静息阶段目标长度（秒）
        self.trial_target_len = trial_target_len  # 审讯阶段目标长度（秒）

    def load_ECG(self):
        # 获取当前工作目录（执行脚本时所在的路径）
        current_path = os.getcwd()
        print("当前工作目录:", current_path)
        file_path = os.listdir(self.ecg_path)  # 获取所有 ecg 文件名
        timeStamp = pd.read_excel(self.timeStamp_path)  # 读取时间戳文件
        X = []  # 存放 ECG 数据
        Y = []  # 存放标签

        # 读取每一个 ecg 文件
        for idx, file in enumerate(file_path):
            data = bioread.read_file(os.path.join(self.ecg_path, file))

            ecg_channel = data.channels[0]  # 获取心电通道
            ecg_data = ecg_channel.data  # 获取心电信号获取原始采样率
            original_fps = data.samples_per_second  # 获取原始采样率

            # 降采样到 target_fps Hz
            down = int(original_fps / self.target_fps)
            ecg_data = signal.resample_poly(ecg_data, 1, down)

            # ==================== 数据截取 ====================
            trial_start_time = float(timeStamp.iloc[idx + 1, 1].split(' ')[0])  # 获取时间部分
            start_stamp, end_stamp = self.getStamp(trial_start_time, ecg_data.shape[0])

            ecg_data = ecg_data[start_stamp:end_stamp]  # 截取数据
            ecg_data = self.rectifyLength(ecg_data)  # 统一数据长度
            X.append(ecg_data)

        X = np.vstack(np.array(X))  # 合并数据
        Y = self.load_Label(self.Label_path)  # 读取标签
        # print('ECG data shape:', X.shape)
        # print('Label shape:', Y.shape)

        return X, Y  # X.shape = (samples, features), Y.shape = (samples,)

    # 统一数据长度。如果长度过长，截取；如果长度过短，补一开始的静息电位
    def rectifyLength(self, ecgdata):
        target_Length = self.trial_target_len * self.target_fps
        # 如果长度过长，截取
        if ecgdata.shape[0] > target_Length:
            ecgdata = ecgdata[0:target_Length]
        # 如果长度过短，补一开始的静息电位
        else:
            surplus = target_Length - ecgdata.shape[0]
            # ecgdata = np.concatenate((ecgdata, ecgdata[0:surplus]))
            ecgdata = np.concatenate((ecgdata, ecgdata[:surplus]))
        return ecgdata

    def getStamp(self, trial_start_time, dataLen):
        """
        获取审讯阶段的起始和结束时间戳
        :param trial_start_time: 审讯阶段的开始时间
        :param dataLen: 降采样后数据长度
        :return: 采用的数据起始和结束时间戳
        """
        # start_stamp = max(0, int(trial_start_time * 60 * self.target_fps - self.static_target_len * self.target_fps))
        # end_stamp = min(int(trial_start_time * 60 * self.target_fps + self.trial_target_len * self.target_fps), dataLen - 1)
        start_stamp = int(trial_start_time * 60 * self.target_fps)
        end_stamp = min(int(start_stamp + self.trial_target_len * self.target_fps), dataLen - 1)
        return start_stamp, end_stamp

    def load_Label(self, Label_path):
        label = pd.read_csv(Label_path, header=None)
        return np.array(label.iloc[1:, 1], dtype=int)

if __name__ == '__main__':
    ecg_path = 'ECG'
    timeStamp_path = 'Label/ECG-timestamp.xlsx'
    Label_path = 'Label/Coarse-grained-labels.csv'
    target_fps = 250  # 目标采样率
    # static_target_len = 200  # 静息阶段目标长度（秒）
    trial_target_len = 200  # 审讯阶段目标长度（秒）

    ecgLoader = EcgLoader(ecg_path, timeStamp_path, Label_path, target_fps, trial_target_len)
    X, Y = ecgLoader.load_ECG()
    np.set_printoptions(threshold=np.inf)
    print(X[0][:100])
    print(X.shape, Y.shape)
