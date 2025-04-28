import os
import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import glob


# 配置路径
predictor_path = 'shape_predictor_68_face_landmarks.dat'
lstm_model_path = 'model/micro_expression_lstm.pth'
cnn_model_path = 'model/lie_detection_model.pth'  # 保存模型的路径


class LSTMMicroExpressionModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMMicroExpressionModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)  # 2分类

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MicroExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # 注意是long类型

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MicroExpressionDetector:
    def __init__(self, reference_size=(96, 112), num_bins=9, window_size=20, overlap=10):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.reference_size = reference_size
        self.num_bins = num_bins
        self.window_size = window_size
        self.overlap = overlap
        self.reference_points = np.array([
            [30.2946, 51.6963], [65.5318, 51.5014],
            [48.0252, 71.7366], [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return None
        for face in faces:
            landmarks = self.predictor(gray, face)
            return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)], dtype=np.float32)
        return None

    def align_face(self, image, landmarks):
        selected = landmarks[[36, 45, 30, 48, 54]]
        matrix, _ = cv2.estimateAffinePartial2D(selected, self.reference_points)
        if matrix is None:
            return None
        return cv2.warpAffine(image, matrix, self.reference_size)

    def compute_optical_flow(self, prev, next):
        return cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    def compute_hoof(self, flow):
        u, v = flow[..., 0], flow[..., 1]
        mag = np.sqrt(u ** 2 + v ** 2)
        angle = np.arctan2(v, u)
        bins = np.linspace(-np.pi, np.pi, self.num_bins + 1)
        hoof = np.zeros(self.num_bins)
        for i in range(self.num_bins):
            mask = (angle >= bins[i]) & (angle < bins[i + 1])
            hoof[i] = np.sum(mag[mask])
        hoof /= np.sum(hoof) if np.sum(hoof) > 0 else 1
        return hoof

    def extract_hoof_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        aligned_faces = []
        total_frames = 0
        detected_faces = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            landmarks = self.get_landmarks(frame)
            if landmarks is not None:
                detected_faces += 1
                aligned = self.align_face(frame, landmarks)
                if aligned is not None:
                    aligned_faces.append(aligned)
        cap.release()

        hoofs = []
        for i in range(1, len(aligned_faces)):
            prev = cv2.cvtColor(aligned_faces[i - 1], cv2.COLOR_BGR2GRAY)
            next = cv2.cvtColor(aligned_faces[i], cv2.COLOR_BGR2GRAY)
            flow = self.compute_optical_flow(prev, next)
            hoof = self.compute_hoof(flow)
            hoofs.append(hoof)
        return hoofs

    def sliding_window(self, features):
        step = self.window_size - self.overlap
        windows = []
        for start in range(0, len(features) - self.window_size + 1, step):
            end = start + self.window_size
            windows.append(np.array(features[start:end]))
        return windows

    def build_lstm_model(self, input_size):
        model = LSTMMicroExpressionModel(input_size)
        return model.to(self.device)

    def train_lstm_model(self, truth_videos_folder, lie_videos_folder):
        # 准备训练数据
        video_label_pairs = []

        for video_file in os.listdir(truth_videos_folder):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(truth_videos_folder, video_file)
                video_label_pairs.append((video_path, 0))

        for video_file in os.listdir(lie_videos_folder):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(lie_videos_folder, video_file)
                video_label_pairs.append((video_path, 1))

        X = []
        y = []
        for video_path, label in video_label_pairs:
            hoofs = self.extract_hoof_features(video_path)
            if len(hoofs) >= self.window_size:
                windows = self.sliding_window(hoofs)
                for win in windows:
                    X.append(win)
                    y.append(label)

        if len(X) == 0:
            raise ValueError("没有足够的训练数据，请检查视频文件")

        X = np.array(X)[..., np.newaxis].squeeze(-1)  # 删除最后一维
        y = np.array(y)

        # 划分训练集验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = MicroExpressionDataset(X_train, y_train)
        val_dataset = MicroExpressionDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 构建并训练模型
        input_size = X_train.shape[2]
        self.model = self.build_lstm_model(input_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(20):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                # 修改 2025/4/27 19:45
                outputs = self.model(inputs)
                outputs = outputs[:, -1, :]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            acc = 100. * correct / total
            print(f"Epoch [{epoch+1}/20], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {acc:.2f}%")

        torch.save(self.model.state_dict(), lstm_model_path)
        print(f"模型已保存到 {lstm_model_path}")

    def detect_micro_expression_intervals(self, video_path):
        # 提取aligned的人脸序列
        cap = cv2.VideoCapture(video_path)
        aligned_faces = []
        total_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            landmarks = self.get_landmarks(frame)
            if landmarks is not None:
                aligned = self.align_face(frame, landmarks)
                if aligned is not None:
                    aligned_faces.append(aligned)
        cap.release()

        # 提取HOOF特征
        hoofs = []
        for i in range(1, len(aligned_faces)):
            prev = cv2.cvtColor(aligned_faces[i - 1], cv2.COLOR_BGR2GRAY)
            next = cv2.cvtColor(aligned_faces[i], cv2.COLOR_BGR2GRAY)
            flow = self.compute_optical_flow(prev, next)
            hoof = self.compute_hoof(flow)
            hoofs.append(hoof)

        if len(hoofs) < self.window_size:
            return [], []

        windows = self.sliding_window(hoofs)
        if not windows:
            return [], []

        X = np.array(windows)[..., np.newaxis].squeeze(-1)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        # 加载LSTM模型
        if self.model is None:
            input_size = X.shape[2]
            self.model = self.build_lstm_model(input_size)
            self.model.load_state_dict(torch.load(lstm_model_path, map_location=self.device))
            self.model.eval()

        with torch.no_grad():
            outputs = self.model(X)
            outputs = outputs[:, -1, :]
            predictions = torch.softmax(outputs, dim=1)

        micro_expression_intervals = []
        processed_image_sequences = []
        current_interval = None

        for idx, pred in enumerate(predictions):
            if torch.argmax(pred) == 1:
                start_frame = idx * (self.window_size - self.overlap)
                end_frame = start_frame + self.window_size
                if current_interval is None:
                    current_interval = [start_frame, end_frame]
                else:
                    if start_frame <= current_interval[1]:
                        current_interval[1] = end_frame
                    else:
                        micro_expression_intervals.append(tuple(current_interval))
                        # 保存当前区间对应的aligned图片
                        processed_image_sequences.append(aligned_faces[current_interval[0]:current_interval[1]])
                        current_interval = [start_frame, end_frame]
            else:
                if current_interval is not None:
                    micro_expression_intervals.append(tuple(current_interval))
                    processed_image_sequences.append(aligned_faces[current_interval[0]:current_interval[1]])
                    current_interval = None

        if current_interval is not None:
            micro_expression_intervals.append(tuple(current_interval))
            processed_image_sequences.append(aligned_faces[current_interval[0]:current_interval[1]])

        # print(micro_expression_intervals)
        return processed_image_sequences
        # return micro_expression_intervals, processed_image_sequences


class LieDetectionNetwork(nn.Module):
    def __init__(self, input_shape_3d=(16, 112, 96, 3), input_shape_2d=(112, 96, 3)):
        """
        初始化谎言检测网络
        :param input_shape_3d: 3D-CNN输入形状 (帧数, 高, 宽, 通道数)
        :param input_shape_2d: 2D-CNN输入形状 (高, 宽, 通道数)
        """
        super(LieDetectionNetwork, self).__init__()
        self.input_shape_3d = input_shape_3d
        self.input_shape_2d = input_shape_2d
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 3D CNN分支
        self.conv3d_branch = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # 计算3D分支展平后的特征维度 修改于2025/4/28 18：59
        dummy_3d = torch.zeros((1, 3, self.input_shape_3d[0], self.input_shape_3d[1], self.input_shape_3d[2]))
        with torch.no_grad():
            dummy_3d_out = self.conv3d_branch(dummy_3d)
        self.flatten_3d_size = dummy_3d_out.view(1, -1).size(1)

        # 2D CNN分支
        self.conv2d_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 计算2D分支展平后的特征维度
        dummy_2d = torch.zeros((1, 3, self.input_shape_2d[0], self.input_shape_2d[1]))
        with torch.no_grad():
            dummy_2d_out = self.conv2d_branch(dummy_2d)
        self.flatten_2d_size = dummy_2d_out.view(1, -1).size(1)

        # 融合后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_3d_size + self.flatten_2d_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_3d, input_2d):
        x3d = self.conv3d_branch(input_3d)
        x3d = x3d.view(x3d.size(0), -1)

        x2d = self.conv2d_branch(input_2d)
        x2d = x2d.view(x2d.size(0), -1)

        combined = torch.cat((x3d, x2d), dim=1)
        output = self.fc(combined)
        return output

    def extract_video_frames(self, video_path, max_frames):
        """从视频中提取帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // max_frames)

        count = 0
        while len(frames) < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                frame = cv2.resize(frame, (self.input_shape_2d[1], self.input_shape_2d[0]))
                frame = frame / 255.0
                frames.append(frame)
            count += 1
        cap.release()
        return frames if len(frames) >= 5 else None

    def prepare_3d_input(self, frames):
        """准备3D CNN输入"""
        # print(f"Number of frames: {len(frames)}")
        # print(f"Expected number of frames: {self.input_shape_3d[0]}")
        # print(f"Frame shape example: {frames[0].shape}")
        stack = np.stack(frames[:self.input_shape_3d[0]])
        # print(f"Stacked shape: {stack.shape}")
        return stack.transpose(3, 0, 1, 2)  # (C, D, H, W)
        # while len(frames) < self.input_shape_3d[0]:
        #     frames.append(frames[-1])
        # return np.stack(frames[:self.input_shape_3d[0]]).transpose(3, 0, 1, 2)  # (C, D, H, W)

    def prepare_2d_input(self, frames):
        """准备2D CNN输入"""
        mid_frame = frames[len(frames) // 2]
        return np.transpose(mid_frame, (2, 0, 1))  # (C, H, W)

    def predict_from_intervals(self, micro_expression_intervals):
        """预测"""

        if not micro_expression_intervals:
            print("警告：没有检测到有效微表情区间！")
            return 0

        self.load_model()
        X_3d = []
        X_2d = []

        for interval in micro_expression_intervals:
            frames = [frame for frame in interval if frame is not None]
            if frames:
                X_3d.append(self.prepare_3d_input(frames))
                X_2d.append(self.prepare_2d_input(frames))

        if not X_3d:
            return 0  # 默认真话

        X_3d = torch.tensor(np.array(X_3d), dtype=torch.float32).to(self.device)
        X_2d = torch.tensor(np.array(X_2d), dtype=torch.float32).to(self.device)

        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_3d, X_2d)
            probs = F.softmax(outputs, dim=1)[:, 1]
            avg_prediction = probs.mean().item()

        return 1 - avg_prediction

    def load_model(self):
        """加载模型"""
        self.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        self.to(self.device)
        print(f"已从 {cnn_model_path} 加载模型")


# 训练模型并保存模型
def save_sequences_to_folder(sequences, base_save_path='saved_sequences'):
    os.makedirs(base_save_path, exist_ok=True)
    for idx, sequence in enumerate(sequences):
        sequence_folder = os.path.join(base_save_path, f'sequence_{idx}')
        os.makedirs(sequence_folder, exist_ok=True)
        for frame_idx, frame in enumerate(sequence):
            save_path = os.path.join(sequence_folder, f'frame_{frame_idx:04d}.jpg')
            cv2.imwrite(save_path, frame)
    print(f"所有序列已保存到：{base_save_path}")


class FaceFileProcessor:
    def __init__(self):
        self.micro_expression_detector = MicroExpressionDetector()
        self.lie_detection_network = LieDetectionNetwork()

    def process_video(self, video_path):
        """
        处理单个mp4视频
        :return:
        """
        intervals = self.micro_expression_detector.detect_micro_expression_intervals(video_path)
        # 根据间隔进行预测
        prediction = self.lie_detection_network.predict_from_intervals(intervals)
        return prediction

    def predict(self, input_path):
        results = {}

        if os.path.isfile(input_path) and input_path.endswith('.mp4'):
            # 输入是一个视频文件
            prediction = self.process_video(input_path)
            video_name = os.path.basename(input_path)
            results[video_name] = prediction

        elif os.path.isdir(input_path):
            # 输入是一个文件夹
            for filename in os.listdir(input_path):
                if filename.endswith('.mp4'):
                    video_path = os.path.join(input_path, filename)
                    prediction = self.process_video(video_path)
                    results[filename] = prediction

        else:
            print(f"输入路径无效：{input_path}")

        return results


if __name__ == "__main__":
    process_path = 'dataset/RLDD_Deceptive/trial_lie_001.mp4'
    process_folder = 'dataset/RLDD_Truthful_test'
    faceFileProcessor = FaceFileProcessor()
    output = faceFileProcessor.predict(process_folder)
    print(output)

