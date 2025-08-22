import os
import uuid
from abc import abstractmethod
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
from django.conf import settings
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.request import Request

from EcgModule.predict import load_ecg_data
from detector.views.Error import CustomAPIException
from detector.views.models import LieDetector


# TODO: 上传文件视图基类
# APIView 是 DRF 的类视图基类，提供 RESTful 方法处理（GET/POST等）
class BaseUploadView(APIView):
    # parser_class 是一个元组，包含了 MultiPartParser  (处理文件上传) 解析器类
    parser_class = (MultiPartParser, )
    modality = None  # 必须被子类覆盖

    # 保存文件, 返回文件路径
    def save(self, file_obj):
        try:
            # 生成唯一文件名
            file_id = uuid.uuid4().hex
            ext = os.path.splitext(file_obj.name)[1].lower()
            new_filename = f"{self.modality}_{datetime.now().strftime('%Y%m%d')}_{file_id}{ext}"

            # 创建存储目录
            save_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', self.modality)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, new_filename)

            # 保存文件
            with open(save_path, 'wb+') as f:
                for chunk in file_obj.chunks():
                    f.write(chunk)  # 分块写入文件
            return save_path

        except Exception as e:
            raise RuntimeError(f"文件保存失败: {str(e)}")

    # 处理文件, 子类必须实现
    @abstractmethod
    def process_file(self, file_obj):
        pass

    # 上传文件
    def post(self, request: Request, *args, **kwargs):
        try:
            file = request.FILES.get('file')  # 从 FILES 字典获取文件对象
            # file = self.save(upload_file)  # 获取保存路径
            result = self.process_file(file)  # 处理文件
            return Response({"status": "success", "data": result}, status=200)

        except CustomAPIException as e:
            return Response({"status": "error", "data": {"error": e.detail, "code": e.status_code}}, status=200)
        except Exception as e:
                return Response({"status": "error", "data": {"error": "服务器内部错误", "code": 500}}, status=500)

# TODO: EEG 处理
class EEGUploadView(BaseUploadView):
    modality = 'eeg'

    # 处理上传的 EEG 文件
    def process_file(self, file):
        if file.name.endswith('.csv'):
            content = file.read()
            file_data = pd.read_csv(BytesIO(content))  # shape: (T, ch)
            print(file_data.shape)
        try:
            eegDetector = LieDetector().eegLoader
            output = eegDetector.predict(file_data)

            """ 构建响应数据 """
            sequence = np.linspace(0, 100, file_data.shape[0])
            data = [
                    [[sequence[t], file_data.values[t][ch]] for t in range(file_data.shape[0])]
                    for ch in range(file_data.shape[1])
                ]
            Response_data = {
                "raw": {
                    "electrodes": file_data.columns.tolist(),
                    "data": data
                },
                "logo": self.modality,  # logo 用于用于确定当前正在处理的模态，modality 用于显示最终处理了哪些模态
                "output": output,
                "modality": self.modality,
                "confidence": 1 - abs(output - round(output)),
                "result": "说谎" if output < 0.5 else "诚实"
            }

            return Response_data

        except Exception as e:
            print(e)
            raise CustomAPIException(f"EEG 文件格式不正确，处理失败")


# TODO: ECG 处理
class ECGUploadView(BaseUploadView):
    modality = 'ecg'

    # 去除时间列
    def drop_time_cols(self, df: pd.DataFrame):
        time_col = [col for col in df.columns if 'time' in col.lower()]
        df = df.drop(columns=time_col)
        return df

    # 处理上传的 ECG 文件
    def process_file(self, file):
        if file.name.endswith('.acq'):
            content = file.read()
            file_data = BytesIO(content)
        try:
            ecgDetector = LieDetector().ecgLoader
            ecg_signal, sampling_rate = load_ecg_data(file_data)
            output = ecgDetector.predict_proba(ecg_signal, sampling_rate)

            """ 构建响应数据 """
            file_data = ecg_signal[::16]
            sequence = np.linspace(0, 100, file_data.shape[0])
            data = [
                [sequence[t], file_data[t]] for t in range(file_data.shape[0])
            ]
            Response_data = {
                "raw": data,
                "logo": self.modality,
                "output": output,
                "modality": self.modality,
                "confidence": 1 - abs(output - round(output)),
                "result": "诚实" if output < 0.5 else "说谎"
            }

            return Response_data

        except Exception as e:
            print(e)
            raise CustomAPIException(f"ECG 文件格式不正确，处理失败")


# TODO: Video 处理
class VideoUploadView(BaseUploadView):
    file_type = 'video'

    # 处理上传的 Video 文件
    def process_file(self, file_obj):
        try:
            faceDetector = LieDetector().faceLoader
            output = faceDetector.predict(file_obj)

            Response_data = {
                "logo": self.modality,
                "output": 1,
                "modality": self.modality,
                "confidence": 1 - abs(output - round(output)),
                "result": "说谎" if output < 0.5 else "诚实"
            }

            return Response_data

        except Exception as e:
            return {'error': 'Video 处理失败', 'detail': str(e)}


