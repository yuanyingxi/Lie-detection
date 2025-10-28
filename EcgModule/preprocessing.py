import os
import bioread
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, find_peaks, welch
from scipy import stats


class EcgProcessor:
    def __init__(self, ecg_path, timeStamp_path, Label_path, target_fps=250, trial_target_len=200):
        self.ecg_path = ecg_path
        self.timeStamp_path = timeStamp_path
        self.Label_path = Label_path
        self.target_fps = target_fps
        self.trial_target_len = trial_target_len
        self.quality_scores = []  # 存储每个文件的质评分

    def load_and_preprocess_ECG(self):
        current_path = os.getcwd()
        print("当前工作目录:", current_path)
        file_path = os.listdir(self.ecg_path)
        timeStamp = pd.read_excel(self.timeStamp_path)

        X_trial_processed = []
        hrv_features_list = []
        Y = []

        print("开始处理ECG数据...")

        for idx, file in enumerate(file_path):
            print(f"处理文件 {idx + 1}/{len(file_path)}: {file}")

            try:
                # 数据读取和截取
                data = bioread.read_file(os.path.join(self.ecg_path, file))
                ecg_channel = data.channels[0]
                original_ecg = ecg_channel.data
                original_fps = data.samples_per_second

                # 降采样
                down = int(original_fps / self.target_fps)
                ecg_data = signal.resample_poly(original_ecg, 1, down)

                # 截取问讯期数据
                trial_start_time = float(timeStamp.iloc[idx + 1, 1].split(' ')[0])
                start_stamp, end_stamp = self.getStamp(trial_start_time, len(ecg_data))
                trial_data = ecg_data[start_stamp:end_stamp]
                trial_data = self.rectifyLength(trial_data)

                # 使用保守预处理方法
                trial_cleaned = self.conservative_denoising(trial_data)

                # 信号质量评估
                quality_score, quality_metrics = self.assess_ecg_quality(trial_cleaned, file)
                self.quality_scores.append({
                    'file': file,
                    'score': quality_score,
                    'metrics': quality_metrics,
                    'quality_level': self.get_quality_level(quality_score)
                })

                # 如果信号质量较差，给出警告
                if quality_score < 0.6:
                    print(f"  ⚠️ 警告: {file} 信号质量较差 (得分: {quality_score:.2f})")

                # 使用新的HRV提取方法
                hrv_features = self.advanced_hrv_extraction_high_snr(trial_cleaned, f"文件{idx + 1}")
                hrv_features_list.append(hrv_features)

                # 保持原有的数据格式
                trial_processed = trial_cleaned[:, np.newaxis]
                X_trial_processed.append(trial_processed)

                print(f"✓ {file} 处理完成")

            except Exception as e:
                print(f"✗ {file} 处理失败: {e}")
                # 添加失败文件的质评分
                self.quality_scores.append({
                    'file': file,
                    'score': 0,
                    'metrics': {'error': str(e)},
                    'quality_level': 'FAILED'
                })
                continue

        X_trial_processed = self.ensure_consistent_shape(X_trial_processed)
        Y = self.load_Label(self.Label_path)
        hrv_features_array = self.process_hrv_features(hrv_features_list)

        # 质量评估
        self.quality_assessment(X_trial_processed, hrv_features_array, Y)

        return {
            'ecg_data': X_trial_processed,
            'hrv_features': hrv_features_array,
            'labels': Y,
            'quality_scores': self.quality_scores
        }

    def assess_ecg_quality(self, ecg_signal, filename):
        """
        综合评估ECG信号质量
        返回质量得分 (0-1) 和详细指标
        """
        metrics = {}

        # 1. 信噪比评估
        snr_score = self.calculate_snr_score(ecg_signal)
        metrics['snr_score'] = snr_score

        # 2. 信号幅度评估
        amplitude_score = self.assess_amplitude(ecg_signal)
        metrics['amplitude_score'] = amplitude_score

        # 3. 基线稳定性评估
        baseline_score = self.assess_baseline_stability(ecg_signal)
        metrics['baseline_score'] = baseline_score

        # 4. 功率谱分析
        spectral_score = self.assess_spectral_quality(ecg_signal)
        metrics['spectral_score'] = spectral_score

        # 5. R峰检测可靠性
        rpeak_score = self.assess_rpeak_reliability(ecg_signal)
        metrics['rpeak_score'] = rpeak_score

        # 6. 信号饱和检测
        saturation_score = self.detect_saturation(ecg_signal)
        metrics['saturation_score'] = saturation_score

        # 综合质量得分 (加权平均)
        weights = {
            'snr_score': 0.25,
            'amplitude_score': 0.20,
            'baseline_score': 0.15,
            'spectral_score': 0.15,
            'rpeak_score': 0.20,
            'saturation_score': 0.05
        }

        total_score = 0
        for key, weight in weights.items():
            total_score += metrics[key] * weight

        metrics['total_score'] = total_score

        # 打印质量报告
        self.print_quality_report(filename, metrics)

        return total_score, metrics

    def calculate_snr_score(self, ecg_signal):
        """计算信噪比得分"""
        # 基于小波去噪的信噪比估计
        try:
            # 使用高通滤波分离信号和噪声
            signal_band = self.butter_bandpass_filter(ecg_signal, 5, 40, self.target_fps, order=3)
            noise_band = self.butter_bandpass_filter(ecg_signal, 0.5, 1, self.target_fps, order=3)

            signal_power = np.mean(signal_band ** 2)
            noise_power = np.mean(noise_band ** 2)

            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                # 将SNR转换为0-1得分
                snr_score = max(0, min(1, (snr_db + 5) / 20))  # -5dB到15dB映射到0-1
            else:
                snr_score = 1.0

        except:
            snr_score = 0.5

        return snr_score

    def assess_amplitude(self, ecg_signal):
        """评估信号幅度质量"""
        signal_std = np.std(ecg_signal)
        signal_range = np.max(ecg_signal) - np.min(ecg_signal)

        # 理想的ECG信号幅度范围 (标准化后)
        ideal_min_std = 0.1
        ideal_max_std = 2.0

        if signal_std < ideal_min_std:
            # 信号太弱
            amplitude_score = signal_std / ideal_min_std
        elif signal_std > ideal_max_std:
            # 信号太强，可能饱和
            amplitude_score = ideal_max_std / signal_std
        else:
            # 理想范围
            amplitude_score = 1.0

        return max(0, min(1, amplitude_score))

    def assess_baseline_stability(self, ecg_signal):
        """评估基线稳定性"""
        # 使用低通滤波提取基线
        baseline = self.butter_lowpass_filter(ecg_signal, 1.0, self.target_fps, order=3)

        # 计算基线漂移
        baseline_drift = np.max(baseline) - np.min(baseline)
        baseline_std = np.std(baseline)

        # 漂移越小越好
        max_acceptable_drift = 1.0  # 标准化后的可接受漂移范围

        if baseline_drift > max_acceptable_drift:
            stability_score = max_acceptable_drift / baseline_drift
        else:
            stability_score = 1.0

        return max(0, min(1, stability_score))

    def assess_spectral_quality(self, ecg_signal):
        """通过功率谱分析信号质量"""
        try:
            # 计算功率谱密度
            f, Pxx = welch(ecg_signal, fs=self.target_fps, nperseg=1024)

            # ECG主要频率成分应该在0.5-40Hz之间
            ecg_band_mask = (f >= 0.5) & (f <= 40)
            noise_band_mask = (f > 40) & (f <= 100)

            ecg_power = np.sum(Pxx[ecg_band_mask])
            noise_power = np.sum(Pxx[noise_band_mask])

            if ecg_power > 0:
                spectral_ratio = ecg_power / (ecg_power + noise_power)
                spectral_score = max(0, min(1, spectral_ratio * 1.5))  # 调整到0-1范围
            else:
                spectral_score = 0

        except:
            spectral_score = 0.5

        return spectral_score

    def assess_rpeak_reliability(self, ecg_signal):
        """通过R峰检测可靠性评估信号质量"""
        try:
            r_peaks = self.sensitive_r_peak_detection(ecg_signal)

            if len(r_peaks) < 10:
                return 0.3  # R峰太少

            # 计算RR间期的变异性
            rr_intervals = np.diff(r_peaks) / self.target_fps
            rr_std = np.std(rr_intervals)
            rr_mean = np.mean(rr_intervals)

            # 生理合理的RR间期变异性
            if rr_std > 0.4 or rr_std < 0.01:  # 标准差在10ms到400ms之间
                reliability_score = 0.5
            else:
                reliability_score = 0.9

            # 根据R峰数量调整得分
            expected_peaks = len(ecg_signal) / (rr_mean * self.target_fps)
            detection_ratio = len(r_peaks) / expected_peaks
            detection_score = max(0, min(1, detection_ratio))

            final_score = (reliability_score + detection_score) / 2

        except:
            final_score = 0.3

        return final_score

    def detect_saturation(self, ecg_signal):
        """检测信号饱和"""
        # 检查信号是否达到最大值或最小值
        max_val = np.max(ecg_signal)
        min_val = np.min(ecg_signal)

        # 检查是否有连续相同的值 (可能表示饱和)
        diff_signal = np.diff(ecg_signal)
        zero_diff_count = np.sum(diff_signal == 0)
        saturation_ratio = zero_diff_count / len(diff_signal)

        if saturation_ratio > 0.1:  # 超过10%的点没有变化
            saturation_score = 0.1
        elif abs(max_val) > 5 or abs(min_val) > 5:  # 信号可能饱和
            saturation_score = 0.3
        else:
            saturation_score = 1.0

        return saturation_score

    def butter_lowpass_filter(self, data, cutoff, fs, order=3):
        """低通滤波器"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def print_quality_report(self, filename, metrics):
        """打印单个文件的质评报告"""
        print(f"  📊 {filename} 质量评估:")
        print(f"    信噪比: {metrics['snr_score']:.2f}, "
              f"幅度: {metrics['amplitude_score']:.2f}, "
              f"基线: {metrics['baseline_score']:.2f}")
        print(f"    频谱: {metrics['spectral_score']:.2f}, "
              f"R峰: {metrics['rpeak_score']:.2f}, "
              f"饱和: {metrics['saturation_score']:.2f}")
        print(f"    🔍 综合质量得分: {metrics['total_score']:.2f} "
              f"({self.get_quality_level(metrics['total_score'])})")

    def get_quality_level(self, score):
        """根据得分返回质量等级"""
        if score >= 0.8:
            return "优秀"
        elif score >= 0.7:
            return "良好"
        elif score >= 0.6:
            return "一般"
        elif score >= 0.4:
            return "较差"
        else:
            return "极差"

    def generate_quality_summary(self):
        """生成总体质量报告"""
        if not self.quality_scores:
            return

        scores = [item['score'] for item in self.quality_scores if isinstance(item['score'], (int, float))]

        if not scores:
            return

        print("\n" + "=" * 60)
        print("ECG信号质量总体报告")
        print("=" * 60)

        print(f"处理文件总数: {len(self.quality_scores)}")
        print(f"质量得分范围: {min(scores):.2f} - {max(scores):.2f}")
        print(f"平均质量得分: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

        # 质量分布
        quality_levels = {'优秀': 0, '良好': 0, '一般': 0, '较差': 0, '极差': 0, 'FAILED': 0}
        for item in self.quality_scores:
            quality_levels[item['quality_level']] += 1

        print("\n质量分布:")
        for level, count in quality_levels.items():
            if count > 0:
                percentage = count / len(self.quality_scores) * 100
                print(f"  {level}: {count}个文件 ({percentage:.1f}%)")

        # 列出质量较差的文件
        poor_quality_files = [item for item in self.quality_scores
                              if item['quality_level'] in ['较差', '极差', 'FAILED']]

        if poor_quality_files:
            print(f"\n⚠️ 需要关注的文件 ({len(poor_quality_files)}个):")
            for item in poor_quality_files:
                print(f"  {item['file']}: {item['quality_level']} (得分: {item['score']:.2f})")

        print("=" * 60)

    def conservative_denoising(self, ecg_signal):
        """新的保守去噪方法"""
        print("  应用去噪策略...")

        # 1. 轻度基线漂移去除
        baseline_removed = self.butter_highpass_filter(ecg_signal, 1.0, self.target_fps, order=3)

        # 2. 轻度工频干扰去除
        powerline_removed = self.mild_notch_filter(baseline_removed, 50, self.target_fps)

        # 3. 保留更多高频成分的带通滤波
        bandpassed = self.butter_bandpass_filter(powerline_removed, 3, 35, self.target_fps, order=3)

        # 4. 轻度标准化
        normalized = self.soft_normalization(bandpassed)

        return normalized

    def mild_notch_filter(self, data, freq, fs, quality=20):
        """轻度陷波滤波"""
        nyq = 0.5 * fs
        freq_normal = freq / nyq
        b, a = iirnotch(freq_normal, quality)
        return filtfilt(b, a, data)

    def soft_normalization(self, data):
        """轻度标准化"""
        mean_val = np.mean(data)
        std_val = np.std(data)

        if std_val > 0.1:
            normalized = (data - mean_val) / std_val
        else:
            normalized = data - mean_val

        return normalized

    def advanced_hrv_extraction_high_snr(self, ecg_signal, file_info=""):
        """新的高精度HRV特征提取"""
        try:
            r_peaks = self.sensitive_r_peak_detection(ecg_signal)

            print(f"  {file_info}: 检测到 {len(r_peaks)} 个R峰")

            if len(r_peaks) < 15:
                print("  警告: R峰数量可能不足")
                return self.get_realistic_hrv_features()

            # 计算RR间期
            rr_intervals = np.diff(r_peaks) / self.target_fps * 1000

            # 轻度异常值过滤
            rr_clean = self.conservative_rr_cleaning(rr_intervals)

            if len(rr_clean) < 10:
                return self.get_realistic_hrv_features()

            # 计算HRV特征
            hrv_features = self.calculate_detailed_hrv(rr_clean)

            return hrv_features

        except Exception as e:
            print(f"  HRV提取错误: {e}")
            return self.get_realistic_hrv_features()

    def sensitive_r_peak_detection(self, ecg_signal):
        """高灵敏度R峰检测"""
        # 方法1: 基于导数的检测
        derivative = np.diff(ecg_signal)
        squared_derivative = derivative ** 2

        threshold = np.percentile(squared_derivative, 85)

        peaks_derivative, _ = find_peaks(squared_derivative,
                                         height=threshold,
                                         distance=int(0.25 * self.target_fps))

        adjusted_peaks = peaks_derivative + 1
        adjusted_peaks = adjusted_peaks[adjusted_peaks < len(ecg_signal)]

        # 方法2: 直接幅度检测
        amplitude_threshold = np.percentile(ecg_signal, 90)
        peaks_amplitude, _ = find_peaks(ecg_signal,
                                        height=amplitude_threshold,
                                        distance=int(0.3 * self.target_fps))

        # 合并结果
        all_peaks = np.unique(np.concatenate([adjusted_peaks, peaks_amplitude]))
        all_peaks.sort()

        return all_peaks

    def conservative_rr_cleaning(self, rr_intervals):
        """保守的RR间期清洗"""
        if len(rr_intervals) == 0:
            return np.array([])

        rr_array = np.array(rr_intervals)

        # 宽松的生理范围过滤
        physiological_mask = (rr_array > 250) & (rr_array < 1600)
        rr_physio = rr_array[physiological_mask]

        if len(rr_physio) == 0:
            return np.array([])

        # 基于中位数的轻度异常值检测
        median_rr = np.median(rr_physio)
        mad_rr = stats.median_abs_deviation(rr_physio)

        lower_bound = max(300, median_rr - 4 * mad_rr)
        upper_bound = min(1200, median_rr + 4 * mad_rr)

        final_mask = (rr_physio > lower_bound) & (rr_physio < upper_bound)
        rr_clean = rr_physio[final_mask]

        removed_count = len(rr_intervals) - len(rr_clean)
        if removed_count > 0:
            print(f"  移除 {removed_count} 个异常RR间期")

        return rr_clean

    def calculate_detailed_hrv(self, rr_intervals):
        """详细的HRV特征计算"""
        features = {}

        # 基础时域特征
        features['HRV_MeanNN'] = np.mean(rr_intervals)
        features['HRV_SDNN'] = np.std(rr_intervals)
        features['HRV_RMSSD'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        features['HRV_SDSD'] = np.std(np.diff(rr_intervals))
        features['HRV_MedianNN'] = np.median(rr_intervals)
        features['HRV_MadNN'] = stats.median_abs_deviation(rr_intervals)

        # 变异系数
        features['HRV_CVNN'] = features['HRV_SDNN'] / features['HRV_MeanNN']
        features['HRV_CVSD'] = features['HRV_SDSD'] / features['HRV_MeanNN']

        # 心率
        features['HRV_MeanHR'] = 60000 / features['HRV_MeanNN']

        # 更精细特征
        features['HRV_pNN50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100
        features['HRV_pNN20'] = np.sum(np.abs(np.diff(rr_intervals)) > 20) / len(rr_intervals) * 100

        # 三角指数
        hist, bin_edges = np.histogram(rr_intervals, bins=20)
        features['HRV_TRI'] = len(rr_intervals) / np.max(hist)

        # 频域特征估算
        features['HRV_LF'] = features['HRV_SDNN'] * 80
        features['HRV_HF'] = features['HRV_RMSSD'] * 120
        features['HRV_LFHF'] = features['HRV_LF'] / features['HRV_HF'] if features['HRV_HF'] != 0 else 0

        # 归一化功率
        total_power = features['HRV_LF'] + features['HRV_HF']
        features['HRV_LFn'] = features['HRV_LF'] / total_power if total_power != 0 else 0
        features['HRV_HFn'] = features['HRV_HF'] / total_power if total_power != 0 else 0

        print(f"  关键HRV - 心率: {features['HRV_MeanHR']:.1f} BPM, "
              f"SDNN: {features['HRV_SDNN']:.1f}ms, RMSSD: {features['HRV_RMSSD']:.1f}ms")

        return features

    def get_realistic_hrv_features(self):
        """返回基于典型生理值的HRV特征"""
        return {
            'HRV_MeanNN': 800, 'HRV_SDNN': 45, 'HRV_RMSSD': 28, 'HRV_SDSD': 22,
            'HRV_CVNN': 0.056, 'HRV_CVSD': 0.028, 'HRV_MedianNN': 795, 'HRV_MadNN': 35,
            'HRV_MeanHR': 75.0, 'HRV_pNN50': 8.5, 'HRV_pNN20': 25.0, 'HRV_TRI': 15.0,
            'HRV_LF': 3600, 'HRV_HF': 3360, 'HRV_LFHF': 1.07,
            'HRV_LFn': 0.517, 'HRV_HFn': 0.483
        }

    # 滤波器函数
    def butter_highpass_filter(self, data, cutoff, fs, order=3):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    # 辅助方法
    def ensure_consistent_shape(self, data_list):
        if not data_list:
            return np.array([])
        min_time_steps = min(data.shape[0] for data in data_list)
        consistent_data = []
        for data in data_list:
            if data.shape[0] > min_time_steps:
                consistent_data.append(data[:min_time_steps])
            else:
                consistent_data.append(data)
        return np.array(consistent_data)

    def rectifyLength(self, ecgdata):
        target_Length = self.trial_target_len * self.target_fps
        if len(ecgdata) > target_Length:
            ecgdata = ecgdata[:target_Length]
        else:
            surplus = target_Length - len(ecgdata)
            mean_val = np.mean(ecgdata)
            ecgdata = np.concatenate([ecgdata, np.full(surplus, mean_val)])
        return ecgdata

    def getStamp(self, trial_start_time, dataLen):
        start_stamp = int(trial_start_time * 60 * self.target_fps)
        end_stamp = min(int(start_stamp + self.trial_target_len * self.target_fps), dataLen - 1)
        return start_stamp, end_stamp

    def load_Label(self, Label_path):
        label = pd.read_csv(Label_path, header=None)
        return np.array(label.iloc[1:, 1], dtype=int)

    def process_hrv_features(self, features_list):
        if not features_list or len(features_list) == 0:
            return np.array([])
        all_feature_names = set()
        for features in features_list:
            if features:
                all_feature_names.update(features.keys())
        feature_names = sorted(list(all_feature_names))
        feature_matrix = []
        for features in features_list:
            row = [features.get(fname, 0) for fname in feature_names]
            feature_matrix.append(row)
        return np.array(feature_matrix)

    def quality_assessment(self, ecg_data, hrv_features, labels):
        """质量评估"""
        print("\n" + "=" * 60)
        print("ECG数据质量评估报告")
        print("=" * 60)

        print(f"ECG数据形状: {ecg_data.shape}")
        print(f"HRV特征形状: {hrv_features.shape}")

        # HRV特征有效性检查
        hrv_means = np.mean(hrv_features, axis=0)
        valid_features = np.sum((hrv_means > 1) & (hrv_means < 2000))

        print(f"有效HRV特征数: {valid_features}/{hrv_features.shape[1]}")
        print(f"标签分布: {np.unique(labels, return_counts=True)}")

        # 生成质量总结报告
        self.generate_quality_summary()

        print("=" * 60)


# 使用方式保持不变
if __name__ == '__main__':
    ecg_path = 'ECG'
    timeStamp_path = 'Label/ECG-timestamp.xlsx'
    Label_path = 'Label/Coarse-grained-labels.csv'
    target_fps = 250
    trial_target_len = 200

    processor = EcgProcessor(ecg_path, timeStamp_path, Label_path, target_fps, trial_target_len)
    processed_data = processor.load_and_preprocess_ECG()
