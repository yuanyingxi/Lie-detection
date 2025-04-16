import os
import cv2
import dlib
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv3D, Conv2D, MaxPooling3D, MaxPooling2D,
                                     Flatten, Dense, concatenate, Input, Dropout)
from tensorflow.keras.models import load_model

lstm_model_path = "model/lstm_model.h5"
predictor_path = "shape_predictor_68_face_landmarks.dat"  # 人脸特征点文件
cnn_model_path = "model/lie_detection_model.h5"


class MicroExpressionDetector:
    def __init__(self, reference_size=(96, 112), num_bins=9, window_size=50, overlap=20):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.reference_size = reference_size  # (宽, 高)
        self.num_bins = num_bins
        self.window_size = window_size
        self.overlap = overlap
        self.reference_points = np.array([  # 用于人脸对齐的参考点
            [30.2946, 51.6963], [65.5318, 51.5014],
            [48.0252, 71.7366], [33.5493, 92.3655],
            [62.7299, 92.2041]
        ], dtype=np.float32)
        self.model = None

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
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = self.get_landmarks(frame)
            if landmarks is not None:
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

    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(2, activation='softmax')  # 二分类：微表情 或 非微表情
        ])
        model.compile(optimizer=Adam(1e-3),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_lstm_model(self, truth_videos_folder, lie_videos_folder):
        # 准备训练数据
        video_label_pairs = []

        # 添加真话视频 (标签0)
        for video_file in os.listdir(truth_videos_folder):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(truth_videos_folder, video_file)
                video_label_pairs.append((video_path, 0))

        # 添加谎话视频 (标签1)
        for video_file in os.listdir(lie_videos_folder):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(lie_videos_folder, video_file)
                video_label_pairs.append((video_path, 1))

        # 提取特征和标签
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

        X = np.array(X)[..., np.newaxis]
        y = to_categorical(y, num_classes=2)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        # 构建并训练模型
        self.model = self.build_lstm_model(X_train.shape[1:])
        self.model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=20,
                       batch_size=32)

        # 保存模型
        self.model.save(lstm_model_path)
        print(f"模型已保存到 {lstm_model_path}")

        return self.model

    def detect_micro_expression_intervals(self, video_path):
        # 加载模型
        self.model = load_model(lstm_model_path)

        # 提取特征
        hoofs = self.extract_hoof_features(video_path)
        if len(hoofs) < self.window_size:
            return []  # 返回空列表表示没有检测到微表情

        # 滑动窗口处理
        windows = self.sliding_window(hoofs)
        if not windows:
            return []

        # 预测
        X = np.array(windows)[..., np.newaxis]
        predictions = self.model.predict(X)

        # 提取微表情间隔
        micro_expression_intervals = []
        current_interval = None

        for idx, pred in enumerate(predictions):
            if np.argmax(pred) == 1:  # 预测为微表情
                start_frame = idx * (self.window_size - self.overlap)
                end_frame = start_frame + self.window_size

                if current_interval is None:
                    current_interval = [start_frame, end_frame]
                else:
                    if start_frame <= current_interval[1]:
                        current_interval[1] = end_frame  # 合并重叠区间
                    else:
                        micro_expression_intervals.append(tuple(current_interval))
                        current_interval = [start_frame, end_frame]
            else:
                if current_interval is not None:
                    micro_expression_intervals.append(tuple(current_interval))
                    current_interval = None

        if current_interval is not None:
            micro_expression_intervals.append(tuple(current_interval))

        return micro_expression_intervals


class LieDetectionNetwork:
    def __init__(self, input_shape_3d=(16, 112, 96, 3), input_shape_2d=(112, 96, 3)):
        """
        初始化谎言检测网络
        :param input_shape_3d: 3D-CNN输入形状 (帧数, 高, 宽, 通道数)
        :param input_shape_2d: 2D-CNN输入形状 (高, 宽, 通道数)
        """
        self.input_shape_3d = input_shape_3d
        self.input_shape_2d = input_shape_2d
        self.model = None

    def build_3d_cnn(self):
        """构建3D CNN分支用于时空特征提取"""
        input_layer = Input(shape=self.input_shape_3d)
        x = Conv3D(32, (3, 3, 3), activation='relu')(input_layer)
        x = MaxPooling3D((2, 2, 2))(x)
        x = Conv3D(64, (3, 3, 3), activation='relu')(x)
        x = MaxPooling3D((2, 2, 2))(x)
        x = Conv3D(128, (3, 3, 3), activation='relu')(x)
        x = MaxPooling3D((2, 2, 2))(x)
        x = Flatten()(x)
        return Model(inputs=input_layer, outputs=x)

    def build_2d_cnn(self):
        """构建2D CNN分支用于空间特征提取"""
        input_layer = Input(shape=self.input_shape_2d)
        x = Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        return Model(inputs=input_layer, outputs=x)

    def build_combined_model(self):
        """构建融合3D和2D特征的复合模型"""
        # 构建两个分支
        model_3d = self.build_3d_cnn()
        model_2d = self.build_2d_cnn()

        # 合并特征
        combined = concatenate([model_3d.output, model_2d.output])

        # 添加全连接层
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)

        # 二分类输出层
        output = Dense(2, activation='softmax')(x)

        # 创建完整模型
        model = Model(
            inputs=[model_3d.input, model_2d.input],
            outputs=output
        )

        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def prepare_training_data(self, truth_videos_dir, lie_videos_dir, max_frames=16):
        """
        准备训练数据：从视频文件夹加载并处理数据
        :param truth_videos_dir: 真话视频文件夹路径
        :param lie_videos_dir: 谎话视频文件夹路径
        :param max_frames: 每个视频提取的最大帧数
        :return: (X_3d, X_2d, y) 训练数据和标签
        """
        X_3d, X_2d, y = [], [], []

        # 处理真话视频 (标签0)
        for video_file in os.listdir(truth_videos_dir):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(truth_videos_dir, video_file)
                frames = self.extract_video_frames(video_path, max_frames)
                if frames:
                    X_3d.append(self.prepare_3d_input(frames))
                    X_2d.append(self.prepare_2d_input(frames))
                    y.append(0)

        # 处理谎话视频 (标签1)
        for video_file in os.listdir(lie_videos_dir):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(lie_videos_dir, video_file)
                frames = self.extract_video_frames(video_path, max_frames)
                if frames:
                    X_3d.append(self.prepare_3d_input(frames))
                    X_2d.append(self.prepare_2d_input(frames))
                    y.append(1)

        return np.array(X_3d), np.array(X_2d), to_categorical(y, num_classes=2)

    def extract_video_frames(self, video_path, max_frames):
        """从视频中提取帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算采样间隔
        interval = max(1, total_frames // max_frames)

        count = 0
        while len(frames) < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                # 调整大小并归一化
                frame = cv2.resize(frame, (self.input_shape_2d[1], self.input_shape_2d[0]))
                frame = frame / 255.0
                frames.append(frame)
            count += 1

        cap.release()
        return frames if len(frames) >= 5 else None  # 至少需要5帧

    def prepare_3d_input(self, frames):
        """准备3D CNN输入数据"""
        # 如果帧数不足，重复最后一帧
        while len(frames) < self.input_shape_3d[0]:
            frames.append(frames[-1])
        return np.array(frames[:self.input_shape_3d[0]])

    def prepare_2d_input(self, frames):
        """准备2D CNN输入数据：使用关键帧"""
        return frames[len(frames) // 2]  # 使用中间帧作为关键帧

    def train(self, truth_videos_dir, lie_videos_dir, epochs=20, batch_size=8):
        """
        训练谎言检测模型
        :param truth_videos_dir: 真话视频文件夹
        :param lie_videos_dir: 谎话视频文件夹
        :param epochs: 训练轮数
        :param batch_size: 批大小
        """
        # 准备数据
        X_3d, X_2d, y = self.prepare_training_data(truth_videos_dir, lie_videos_dir)

        # 构建模型
        self.model = self.build_combined_model()

        # 训练
        self.model.fit(
            [X_3d, X_2d],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            shuffle=True
        )

        # 保存模型
        self.save_model()

    def predict_from_intervals(self, micro_expression_intervals):
        """
        从微表情间隔预测谎言
        :param micro_expression_intervals: 微表情间隔列表，每个元素是帧列表
        :return: 预测结果 (1: 谎言, 0: 真话)
        """
        self.load_model()

        # 准备输入数据
        X_3d = []
        X_2d = []
        for interval in micro_expression_intervals:
            frames = [frame for frame in interval if frame is not None]
            if frames:
                X_3d.append(self.prepare_3d_input(frames))
                X_2d.append(self.prepare_2d_input(frames))

        if not X_3d:
            return 0  # 默认返回真话

        # 预测
        predictions = self.model.predict([np.array(X_3d), np.array(X_2d)])
        avg_prediction = np.mean(predictions[:, 1])  # 取谎言概率的平均值

        return 1 if avg_prediction > 0.5 else 0

    def save_model(self):
        """保存模型"""
        if self.model is not None:
            self.model.save(cnn_model_path)
            print(f"模型已保存到 {cnn_model_path}")

    def load_model(self):
        """加载模型"""
        self.model = load_model(cnn_model_path)
        print(f"已从 {cnn_model_path} 加载模型")


def test_model_accuracy(truth_videos_folder, lie_videos_folder):
    micro_expression_detector = MicroExpressionDetector()
    print("start train lstm")
    micro_expression_detector.train_lstm_model(truth_videos_folder=truthful_folder, lie_videos_folder=deceptive_folder)
    lie_detection_network = LieDetectionNetwork(input_shape_3d=(50, 96, 112, 3), input_shape_2d=(96, 112, 3))
    print("start train cnn")
    lie_detection_network.train(truth_videos_dir=truthful_folder, lie_videos_dir=deceptive_folder)

    # 准备测试数据
    test_videos = []
    true_labels = []

    # 添加真话视频 (标签0)
    for video_file in os.listdir(truth_videos_folder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(truth_videos_folder, video_file)
            print("addend" + video_path)
            test_videos.append(video_path)
            true_labels.append(0)

    # 添加谎话视频 (标签1)
    for video_file in os.listdir(lie_videos_folder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(lie_videos_folder, video_file)
            print("addend" + video_path)
            test_videos.append(video_path)
            true_labels.append(1)

    if not test_videos:
        print("没有找到测试视频")
        return

    # 进行预测
    correct = 0
    total = len(test_videos)

    for video_path, true_label in zip(test_videos, true_labels):
        try:
            # 检测微表情间隔
            micro_expression_intervals = micro_expression_detector.detect_micro_expression_intervals(video_path)

            # 从视频中提取帧用于谎言检测
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (lie_detection_network.input_shape_2d[1],
                                           lie_detection_network.input_shape_2d[0]))
                frame = frame / 255.0
                frames.append(frame)
            cap.release()

            # 准备微表情间隔对应的帧
            interval_frames = []
            for start, end in micro_expression_intervals:
                if end > len(frames):
                    end = len(frames)
                interval_frames.append(frames[start:end])

            # 预测谎言
            prediction = lie_detection_network.predict_from_intervals(interval_frames)

            # 计算准确率
            if prediction == true_label:
                correct += 1

            print(f"视频: {os.path.basename(video_path)}, 真实标签: {true_label}, 预测结果: {prediction}")
        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {str(e)}")
            total -= 1

    accuracy = correct / total if total > 0 else 0
    print(f"\n测试完成，准确率: {accuracy:.2%} ({correct}/{total})")


if __name__ == "__main__":
    deceptive_folder = "dataset/RLDD_Deceptive"   # 路径为谎言视频
    truthful_folder = "dataset/RLDD_Truthful"       # 路径为真实视频
    test_model_accuracy(truthful_folder, deceptive_folder)




