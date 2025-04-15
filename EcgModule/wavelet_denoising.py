import numpy as np
import math
import pywt
from ecgLoader import EcgLoader
import scipy.signal as sg

def get_ecg_data():
    ecg_path = 'ECG'
    timeStamp_path = 'Label/ECG-timestamp.xlsx'
    Label_path = 'Label/Coarse-grained-labels.csv'
    target_fps = 250  # 目标采样率
    static_target_len = 200  # 静息阶段目标长度（秒）
    trial_target_len = 200  # 审讯阶段目标长度（秒）

    ecgLoader = EcgLoader(ecg_path, timeStamp_path, Label_path, target_fps, static_target_len, trial_target_len)
    X, Y = ecgLoader.load_ECG()
    return X  # 返回X值


def calculate_snr(signal, noise):
    """
    计算信号的信噪比（SNR）。

    参数：
        signal (np.array): 原始信号。
        noise (np.array): 噪声信号（原始信号与去噪信号的差值）。

    返回：
        snr (float): 信噪比（dB）。
    """
    signal_power = np.mean(signal ** 2)  # 信号功率
    noise_power = np.mean(noise ** 2)  # 噪声功率
    snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else 0  # 计算 SNR
    return snr

def sgn(num):
    """
    符号函数
    """
    if (num > 0.0):
        return 1.0
    elif (num == 0.0):
        return 0.0
    else:
        return -1.0

def remove_baseline_wander(signal,sampling_rate):
    """
    去除基线漂移

    参数 :
        signal (np.array) : 单个 ECG 信号，形状为 (10000,)
        sampling_rate (int) : 采样率

    返回 :
        filtered_signal (np.array) : 去除基线漂移后的信号，形状为 (10000,)
    """
    # 滤波去除基线漂移
    baseline=sg.medfilt(sg.medfilt(signal,int(0.2*sampling_rate)-1),int(0.6*sampling_rate)-1)
    filtered_signal=signal-baseline


    return filtered_signal

def wavelet_noising(new_df):
    """
    对单个 ECG 信号进行小波去噪

    参数：
        signal (np.array) : 单个 ECG 信号，形状为 (10000,)
    返回：
        recoeffs (np.array) : 去噪后的信号，形状为 (10000,)
    """
    data = new_df
    data = data.T.tolist()
    w = pywt.Wavelet('db8')
    [ca5,cd5,cd4,cd3,cd2,cd1]=pywt.wavedec(data,w,level=5) # 分解波

    length1=len(cd1)
    length0=len(data)

    # 计算细节系数 cd1 的中位数绝对偏差
    Cd1=np.array(cd1)
    abs_cd1=np.abs(Cd1)
    median_cd1=np.median(abs_cd1)

    sigma=(1.0/0.6745)*median_cd1 # 噪声的标准差估计值
    lamda=sigma*math.sqrt(2.0*math.log(float(length0),math.e)) #阈值，用于判断细节系数是否为噪声

    usecoeffs=[] # 存储去噪后的系数
    usecoeffs.append(ca5)

    a=0.5 # 阈值处理强度

    # 处理 cd1
    for k in range(length1):
        if(abs(cd1[k])>=lamda):
            cd1[k]=sgn(cd1[k])*(abs(cd1[k])-a*lamda)
        else:
            cd1[k]=0.0

    # 处理 cd2
    length2=len(cd2)
    for k in range(length2):
        if(abs(cd2[k]) >= lamda):
            cd2[k]=sgn(cd2[k])*(abs(cd2[k])-a*lamda)
        else:
            cd2[k]=0.0

    # 处理 cd3
    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    # 处理 cd4
    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k]) >= lamda):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
        else:
            cd4[k] = 0.0

    # 处理 cd5
    length5 = len(cd5)
    for k in range(length5):
        if (abs(cd5[k]) >= lamda):
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
        else:
            cd5[k] = 0.0

    # 重构信号
    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)

    # 去噪后的信号
    recoeffs=pywt.waverec(usecoeffs,w)
    return recoeffs

def process_ecg_dataset(ecg_data):
    """
    处理整个 ECG 信号集

    参数：
        ecg_data (np.array) : ECG数据集，形状为 (76,10000)

    返回：
        processed_ecg_data (np.array) : 处理后的 ECG 数据集，形状为 (76,10000)
    """
    processed_ecg_data=np.zeros_like(ecg_data) # 初始化存储处理后的数据

    # 遍历每个ECG信号并处理
    for i in range(ecg_data.shape[0]):
        # 1. 去除基线漂移
        signal_no_baseline = remove_baseline_wander(ecg_data[i],sampling_rate=250)
        # 2. 小波去噪
        processed_ecg_data[i]=wavelet_noising(signal_no_baseline)

    return processed_ecg_data

def get_processed_ecg():
    ecg_data = get_ecg_data()

    # 处理整个 ECG 数据集
    print("Processing ECG data...")
    processed_ecg_data = process_ecg_dataset(ecg_data)

    return processed_ecg_data









