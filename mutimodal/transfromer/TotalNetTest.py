import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
import time
import torch.optim as optim
from EcgModule.preprocessing import EcgProcessor
from TotalNet import TotalNet2


class MultiModalDataset(Dataset):
    def __init__(self, ecg_data, face_data, labels, ecg_seq_len=1000):
        self.ecg_data = ecg_data
        self.face_data = face_data
        self.labels = labels
        self.ecg_seq_len = ecg_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # ECG数据: (50000, 1) -> 降采样到指定长度
        ecg_original = self.ecg_data[idx]

        # 先转换为numpy数组处理
        if isinstance(ecg_original, torch.Tensor):
            ecg_original = ecg_original.numpy()

        # 检查形状
        if ecg_original.shape[0] > self.ecg_seq_len:
            # 降采样
            step = ecg_original.shape[0] // self.ecg_seq_len
            ecg = ecg_original[::step][:self.ecg_seq_len]
        else:
            ecg = ecg_original

        # 转换为tensor
        ecg = torch.FloatTensor(ecg)

        # 面部数据: (100, 136)
        face = torch.FloatTensor(self.face_data[idx])

        # 标签
        label = torch.FloatTensor([self.labels[idx]])

        return ecg, face, label


def load_and_align_data(ecg_path, timeStamp_path, Label_path, face_data_path):
    """
    加载并对齐ECG和面部数据
    """
    # 1. 处理ECG数据
    print("正在处理ECG数据...")
    ecg_processor = EcgProcessor(ecg_path, timeStamp_path, Label_path)
    ecg_result = ecg_processor.load_and_preprocess_ECG()

    ecg_data = ecg_result['ecg_data']  # 形状: (76, 50000, 1)
    labels = ecg_result['labels']  # 形状: (76,)

    print(f"ECG数据形状: {ecg_data.shape}")
    print(f"标签形状: {labels.shape}")

    # 2. 加载面部数据
    print("正在加载面部数据...")
    face_data = load_face_data(face_data_path, len(labels))
    print(f"面部数据形状: {face_data.shape}")

    return ecg_data, face_data, labels, ecg_processor


def load_face_data(face_data_path, expected_samples):
    """
    加载面部数据并根据文件名排序对齐
    """
    if not os.path.exists(face_data_path):
        raise FileNotFoundError(f"面部数据路径不存在: {face_data_path}")

    # 只获取.npy文件
    face_files = [f for f in os.listdir(face_data_path) if f.endswith('.npy')]

    # 提取序号并排序
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0

    # 排序文件
    sorted_files = sorted(face_files, key=extract_number)

    face_data_list = []
    loaded_count = 0

    print(f"找到 {len(sorted_files)} 个面部数据文件，需要 {expected_samples} 个")

    for i, file in enumerate(sorted_files):
        if loaded_count >= expected_samples:
            break

        file_path = os.path.join(face_data_path, file)
        try:
            # 加载.npy文件
            face_array = np.load(file_path, allow_pickle=True)

            # 检查数据形状
            if face_array.shape != (100, 136):
                print(f"警告: {file} 的形状为 {face_array.shape}，期望 (100, 136)")
                # 调整形状
                if len(face_array.shape) == 2:
                    if face_array.shape[0] >= 100 and face_array.shape[1] >= 136:
                        face_array = face_array[:100, :136]
                    else:
                        adjusted = np.zeros((100, 136))
                        min_rows = min(face_array.shape[0], 100)
                        min_cols = min(face_array.shape[1], 136)
                        adjusted[:min_rows, :min_cols] = face_array[:min_rows, :min_cols]
                        face_array = adjusted

            face_data_list.append(face_array)
            loaded_count += 1

            if loaded_count % 10 == 0:
                print(f"已加载 {loaded_count} 个面部数据文件")

        except Exception as e:
            print(f"加载文件 {file} 时出错: {e}")
            continue

    return np.array(face_data_list)


def create_data_loaders(ecg_data, face_data, labels, batch_size=8, train_ratio=0.8):  # 移除了ecg_processor参数
    """
    创建训练和测试数据加载器
    """
    dataset = MultiModalDataset(ecg_data, face_data, labels, ecg_seq_len=1000)  # 设置序列长度为1000

    # 划分训练集和测试集
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    print(f"ECG序列长度已降采样到: 1000")

    return train_loader, test_loader


def train_model(model, train_loader, test_loader, num_epochs=50, learning_rate=0.001):
    """
    训练多模态模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f'Epoch {epoch + 1}/{num_epochs} [训练]')
        for batch_idx, (ecg, face, labels) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"处理批次 {batch_idx + 1}/{len(train_loader)}")
                print(f"  ECG形状: {ecg.shape}")
                print(f"  面部数据形状: {face.shape}")
                print(f"  标签形状: {labels.shape}")

            ecg, face, labels = ecg.to(device), face.to(device), labels.to(device)

            optimizer.zero_grad()

            try:
                # 前向传播
                outputs = model(ecg, face)
                loss = criterion(outputs, labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # 计算准确率
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                batch_acc = train_correct / train_total
                print(f'  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}')

            except Exception as e:
                print(f"  训练过程中出错: {e}")
                import traceback
                traceback.print_exc()
                break

        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        print(f'Epoch {epoch + 1}/{num_epochs} [测试]')
        with torch.no_grad():
            for batch_idx, (ecg, face, labels) in enumerate(test_loader):
                ecg, face, labels = ecg.to(device), face.to(device), labels.to(device)

                outputs = model(ecg, face)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                predicted = (outputs > 0.5).float()
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                if (batch_idx + 1) % 2 == 0:
                    batch_acc = test_correct / test_total
                    print(f'  Batch {batch_idx + 1}/{len(test_loader)}, Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}')

        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_accuracy = train_correct / train_total
        test_accuracy = test_correct / test_total

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        epoch_time = time.time() - start_time

        print(f'Epoch {epoch + 1}/{num_epochs} 完成 - 时间: {epoch_time:.2f}s')
        print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.4f}')
        print(f'  测试损失: {avg_test_loss:.4f}, 测试准确率: {test_accuracy:.4f}')
        print('-' * 50)

        # 更新学习率
        scheduler.step()

    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }


def save_model(model, path='multimodal_model.pth'):
    """
    保存训练好的模型
    """
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"模型已保存到: {path}")


def calculate_test_accuracy(model, test_loader):
    """
    计算准确率
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    correct_predictions = 0
    total_samples = 0

    print(f"\n开始在测试集上计算准确率...")

    with torch.no_grad():
        for ecg, face, labels in test_loader:
            ecg, face, labels = ecg.to(device), face.to(device), labels.to(device)

            outputs = model(ecg, face)
            predictions = (outputs > 0.5).float()

            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    test_accuracy = correct_predictions / total_samples

    print("\n" + "=" * 50)
    print("测试集准确率:")
    print("=" * 50)
    print(f"测试样本数: {total_samples}")
    print(f"正确预测数: {correct_predictions}")
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("=" * 50)

if __name__ == "__main__":
    # 数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(os.path.dirname(current_dir))  # 项目根目录
    ecg_path = os.path.join(base_path, 'EcgModule', 'ECG')
    timeStamp_path = os.path.join(base_path, 'EcgModule', 'Label', 'ECG-timestamp.xlsx')
    Label_path = os.path.join(base_path, 'EcgModule', 'Label', 'Coarse-grained-labels.csv')
    face_data_path = os.path.join(base_path, 'FaceModule', 'SEUMLD_data_to_NN')

    # 加载和对齐数据
    ecg_data, face_data, labels, ecg_processor = load_and_align_data(
        ecg_path, timeStamp_path, Label_path, face_data_path
    )

    # 创建数据加载器 - 移除了ecg_processor参数
    train_loader, test_loader = create_data_loaders(
        ecg_data, face_data, labels, batch_size=8
    )

    # 初始化模型
    model = TotalNet2()
    print("模型初始化完成")

    # 训练模型
    print("开始训练...")
    training_result = train_model(model, train_loader, test_loader, num_epochs=50)

    # 保存模型
    save_model(training_result['model'])

    # 计算最终准确率
    final_accuracy = calculate_test_accuracy(training_result['model'], test_loader)