import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Checkpoint
from skorch.helper import predefined_split
import neurokit2 as nk
import pywt
import cv2
from wavelet_denoising import get_processed_ecg
import pandas as pd
import os
import warnings
from collections import Counter

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = x1.astype(np.float32)  # CWT特征 [n_samples, 1, 100, 100]
        self.x2 = x2.astype(np.float32)  # RR特征 [n_samples, 4]
        self.y = y.astype(np.int64)  # 标签 [n_samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'cwt': torch.from_numpy(self.x1[idx]),
            'rr': torch.from_numpy(self.x2[idx])
        }, self.y[idx]


class LieDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CWT特征分支
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d(5)
        self.pooling2 = nn.MaxPool2d(3)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))

        # RR间隔特征分支
        self.rr_fc1 = nn.Linear(4, 16)

        # 合并层
        self.fc1 = nn.Linear(64 + 16, 32)
        self.fc2 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)

    def extra_feature(self, cwt, rr):
        # CWT特征处理
        x1 = F.relu(self.bn1(self.conv1(cwt)))
        x1 = self.pooling1(x1)
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = self.pooling2(x1)
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = self.pooling3(x1)
        x1 = x1.view(-1, 64)

        # RR间隔特征处理
        x2 = F.relu(self.rr_fc1(rr))

        # 合并特征
        x = torch.cat([x1, x2], dim=1)  # [batch, 80]
        return x

    def forward(self, cwt, rr):
        x = self.extra_feature(cwt, rr)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def extract_rr_features(r_peaks, sampling_rate):
    if len(r_peaks) < 2:
        return None
    rr_intervals = np.diff(r_peaks) / sampling_rate
    return np.array([
        np.mean(rr_intervals),
        np.std(rr_intervals),
        np.min(rr_intervals),
        np.max(rr_intervals)
    ], dtype=np.float32)


def process_ecg_signal(ecg_signal, wavelet="mexh", sampling_rate=250):
    try:
        if len(ecg_signal) < 500:
            print(f"信号长度过短: {len(ecg_signal)}")
            return None, None

        # 预处理信号
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
        r_peaks = info["ECG_R_Peaks"]
        if len(r_peaks) < 2:
            print(f"R峰不足: {len(r_peaks)}")
            return None, None

        rr_features = extract_rr_features(r_peaks, sampling_rate)
        if rr_features is None:
            print("RR特征提取失败")
            return None, None

        # 小波变换
        scales = np.logspace(np.log10(1), np.log10(50), num=50)
        coeffs, _ = pywt.cwt(ecg_cleaned, scales, wavelet, 1.0 / sampling_rate)

        features = []
        for i in range(min(len(r_peaks) - 1, 10)):  # 取前10个心跳周期
            start = max(0, r_peaks[i] - int(0.25 * sampling_rate))
            end = min(len(ecg_cleaned), r_peaks[i] + int(0.25 * sampling_rate))
            segment = coeffs[:, start:end]
            if segment.shape[1] < 10:
                continue
            resized = cv2.resize(segment, (100, 100))
            normalized = (resized - np.mean(resized)) / (np.std(resized) + 1e-8)
            features.append(normalized)

        if not features:
            print("没有有效的心跳特征")
            return None, None

        cwt_feature = np.mean(features, axis=0)
        return cwt_feature, rr_features
    except Exception as e:
        print(f"处理ECG信号时出错: {str(e)}")
        return None, None


def augment_honest_samples(x1_data, x2_data, y_data):
    """只对诚实样本（标签0）进行数据增强，且增强后诚实样本不超过说谎样本的80%"""
    honest_indices = np.where(y_data == 0)[0]
    lie_count = np.sum(y_data == 1)  # 使用 np.sum 而不是 sum

    # 计算当前诚实样本数量和允许的最大诚实样本数量(说谎样本)
    current_honest = len(honest_indices)
    max_allowed_honest = int(lie_count * 0.8)

    # 如果需要增强的样本数
    needed_augment = max(0, max_allowed_honest - current_honest)

    # 如果不需要增强，直接返回原数据
    if needed_augment <= 0:
        print(f"诚实样本已足够({current_honest})，不超过说谎样本的80%({max_allowed_honest})，不进行增强")
        return x1_data, x2_data, y_data

    augmented_x1 = list(x1_data)
    augmented_x2 = list(x2_data)
    augmented_y = list(y_data)

    # 计算每个原始诚实样本需要增强的次数
    augment_per_sample = needed_augment // len(honest_indices)
    extra_augment = needed_augment % len(honest_indices)

    for idx in honest_indices:
        # 计算这个样本需要增强的次数
        augment_times = augment_per_sample + (1 if extra_augment > 0 else 0)
        extra_augment -= 1

        for _ in range(augment_times):
            # 随机选择增强方式
            if np.random.rand() > 0.5:
                # 添加噪声
                noise_scale = 0.05
                noisy_x1 = x1_data[idx] + np.random.normal(0, noise_scale, x1_data[idx].shape)
                augmented_x1.append(noisy_x1)
                augmented_x2.append(x2_data[idx])
                augmented_y.append(0)
            else:
                # 时间扭曲
                scale_factor = np.random.uniform(0.9, 1.1)
                scaled_x1 = cv2.resize(x1_data[idx].squeeze(),
                                       (int(100 * scale_factor), int(100 * scale_factor)))
                scaled_x1 = cv2.resize(scaled_x1, (100, 100))
                scaled_x1 = np.expand_dims(scaled_x1, axis=0)
                augmented_x1.append(scaled_x1)
                augmented_x2.append(x2_data[idx])
                augmented_y.append(0)

    # 将列表转换为 numpy 数组
    augmented_y = np.array(augmented_y)
    print(f"增强后诚实样本数: {np.sum(augmented_y == 0)} (不超过说谎样本的80%: {max_allowed_honest})")
    return (np.array(augmented_x1, dtype=np.float32),
            np.array(augmented_x2, dtype=np.float32),
            augmented_y.astype(np.int64))


def load_labels(label_path):
    try:
        df = pd.read_csv(label_path)
        if 'label' not in df.columns:
            raise ValueError("CSV文件中缺少'label'列")
        labels = df['label'].values
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("标签必须为0或1")
        return labels
    except Exception as e:
        print(f"加载标签文件时出错: {str(e)}")
        return None


def load_data(ecg_data, labels):
    x1_data, x2_data, y_data = [], [], []
    for i, (ecg, label) in enumerate(zip(ecg_data, labels)):
        cwt_feature, rr_features = process_ecg_signal(ecg)
        if cwt_feature is not None and rr_features is not None:
            x1_data.append(cwt_feature)
            x2_data.append(rr_features)
            y_data.append(label)
        else:
            print(f"样本 {i} 处理失败，跳过")

    if not x1_data:
        raise ValueError("没有有效的数据")

    x1_data = np.expand_dims(np.array(x1_data), axis=1)
    x2_data = np.array(x2_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.int64)

    return x1_data, x2_data, y_data


def main():
    print("正在获取ECG数据...")
    ecg_data = get_processed_ecg()
    label_path = os.path.join("Label", "Coarse-grained-labels.csv")
    print(f"正在从 {label_path} 加载标签数据...")
    labels = load_labels(label_path)

    if labels is None or len(ecg_data) != len(labels):
        raise ValueError("数据加载失败或数据不匹配")

    print("正在处理ECG数据...")
    x1_data, x2_data, y_data = load_data(ecg_data, labels)
    print(f"原始数据样本数: {len(x1_data)}, 标签分布: {Counter(y_data)}")

    # 分割训练集和测试集 (先分割再标准化)
    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
        x1_data, x2_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

    # 标准化RR特征 (只在训练集上fit)
    scaler = StandardScaler()
    x2_train = scaler.fit_transform(x2_train)
    x2_test = scaler.transform(x2_test)

    print(f"\n训练集大小: {len(x1_train)}, 测试集大小: {len(x1_test)}")
    print(f"训练集标签分布: {Counter(y_train)}")
    print(f"测试集标签分布: {Counter(y_test)}")

    # 对诚实样本进行数据增强
    x1_train, x2_train, y_train = augment_honest_samples(x1_train, x2_train, y_train)
    print(f"\n数据增强后训练集大小: {len(x1_train)}, 标签分布: {Counter(y_train)}")

    # 计算类别权重
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = torch.tensor(class_weights, device=device, dtype=torch.float32)

    # 创建数据集
    train_dataset = ECGDataset(x1_train, x2_train, y_train)
    valid_dataset = ECGDataset(x1_test, x2_test, y_test)

    # 模型检查点回调
    checkpoint = Checkpoint(monitor='valid_acc_best', dirname='checkpoints')

    # 定义模型
    model = NeuralNetClassifier(
        LieDetectionModel,
        criterion=nn.CrossEntropyLoss(weight=class_weights),
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=1e-4,
        lr=0.001,
        max_epochs=50,
        batch_size=16,
        iterator_train__shuffle=True,
        device=device,
        callbacks=[
            EpochScoring(scoring="accuracy", lower_is_better=False, name="train_acc"),
            EpochScoring(scoring="balanced_accuracy", lower_is_better=False, name="valid_acc",
                         on_train=False),
            checkpoint
        ],
        train_split=predefined_split(valid_dataset),
        classes=[0, 1]
    )

    print("\n开始训练模型...")
    model.fit(train_dataset, y=None)

    # 评估模型
    print("\n在测试集上评估模型...")
    y_pred = model.predict(valid_dataset)

    # 计算各类别的准确率
    cm = confusion_matrix(y_test, y_pred)
    class0_acc = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0.0
    class1_acc = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0.0

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, digits=4, target_names=['诚实', '说谎']))
    print("\n混淆矩阵:")
    print(cm)
    print(f"\n准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"平衡准确率: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"类别0(诚实)准确率: {class0_acc:.4f}")
    print(f"类别1(说谎)准确率: {class1_acc:.4f}")

    # 保存模型和标准化器参数
    model_save_path = 'Label/lie_detection_model.pth'
    torch.save({
        'model_state_dict': model.module_.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'input_shape': (1, 100, 100),  # CWT输入形状
        'num_rr_features': 4  # RR特征数量
    }, model_save_path)
    print(f"\n模型已保存到: {model_save_path}")


if __name__ == "__main__":
    main()